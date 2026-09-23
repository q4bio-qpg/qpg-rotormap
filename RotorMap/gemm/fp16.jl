# =============================================================================
# fp16.jl -- the fp16 batched GEMM + per-row top-k stage (cuBLAS), self-
# contained: carries the ORIGINAL fp16 kernels from legacy gemmtopk.jl.
#
#   rowtopk_merge_kernel! / launch_rowtopk_merge!      one thread per row
#   seg_rowtopk_merge_kernel! / launch_seg_rowtopk_merge!   PS threads per row
#   gemm_topk! / gemm_topk    chunked fp16 GEMM (CUDA.mul!) + running top-k
#   randn_fp16!               test-fixture generator
#   test_rowtopk_merge_kernel / test_seg_rowtopk_merge_kernel / spot_check
#
# NOT co-includable with gemm/topk_kernels.jl (same names: TOPK_UNROLL,
# seg_rowtopk_merge_kernel!, ...).  Reason both copies exist: the fp8-era
# kernels in topk_kernels.jl (with _scanval + NTuple helpers) do not
# GPU-compile for Float16 outputs (InvalidIRError), so the fp16 flow keeps
# its original kernels -- byte-for-byte the legacy behavior.
# REQUIRES: nothing (CUDA only).
# =============================================================================
using CUDA
using LinearAlgebra
using Random

# once (descending), then two-pointer-merges it into the running top-K in
# D_val/D_loc. Fresh indices are chunk-local (1..width) and are shifted by
# `offset` (= start column - 1) into global column ids; sentinel entries
# (value typemin, index 0) survive as 0. The scan is unrolled TOPK_UNROLL×
# with the next batch of loads issued before the current batch is processed
# (latency hiding); values outside 1:width come in as typemin sentinels and
# never enter the reservoir.
# --------------------------------------------------------------------------
const TOPK_UNROLL = 4

@inline function _tup_argmin(t::NTuple{K, T}) where {K, T}
    mv = t[1]
    mp = 1
    for j in 2:K                     # static bounds → fully unrolled selects
        vj = t[j]
        less = vj < mv
        mv = less ? vj : mv
        mp = less ? j : mp
    end
    (mv, mp)
end

@inline _tup_replace(t::NTuple{K, T}, pos::Int, v) where {K, T} =
    ntuple(j -> j == pos ? v : t[j], Val(K))

@inline function _tup_insert!(lk, li, v, idx::Int32, min_pos)
    lk = _tup_replace(lk, min_pos, v)
    li = _tup_replace(li, min_pos, idx)
    mv, mp = _tup_argmin(lk)
    return (lk, li, mv, mp)
end

# `C` may be a dense device matrix OR a strided (ld ≠ M) row/column-ranged
# device view: indexing is CARTESIAN `C[row, col]`, which costs the same
# multiply-add as the original manual linear indexing for a contiguous matrix
# (stride(C, 2) == M) and stays correct for views of the engine's double
# buffers (tail batches nb < rows_cap, ragged last chunks wi < w).
function rowtopk_merge_kernel!(out_vals::CuDeviceMatrix{T}, out_inds::CuDeviceMatrix{Int32},
                               C, offset::Int32, ::Val{K}) where {T, K}
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M = size(C, 1)
    row <= M || return nothing
    U = TOPK_UNROLL

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(T), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(T)             # current reservoir minimum (register)
    min_pos = 1                      # argmin(lk)

    width = size(C, 2)
    c = 1
    # rolling window of U in-flight loads; nᵢ holds column c+i-1
    @inbounds begin
        n1 = 1 <= width ? C[row, 1] : typemin(T)
        n2 = 2 <= width ? C[row, 2] : typemin(T)
        n3 = 3 <= width ? C[row, 3] : typemin(T)
        n4 = 4 <= width ? C[row, 4] : typemin(T)
    end
    @inbounds while c <= width
        # issue the whole next batch of U loads before processing (latency hiding)
        # (original 0-based linear C[row + (c+U-1+j)*M] == 1-based C[row, c+U+j])
        m1 = c + U <= width ? C[row, c + U] : typemin(T)
        m2 = c + U + 1 <= width ? C[row, c + U + 1] : typemin(T)
        m3 = c + U + 2 <= width ? C[row, c + U + 2] : typemin(T)
        m4 = c + U + 3 <= width ? C[row, c + U + 3] : typemin(T)
        if n1 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n1, Int32(c), min_pos); end
        if n2 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n2, Int32(c + 1), min_pos); end
        if n3 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n3, Int32(c + 2), min_pos); end
        if n4 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n4, Int32(c + 3), min_pos); end
        n1 = m1; n2 = m2; n3 = m3; n4 = m4
        c += U
    end

    # sort the reservoir descending (once per thread)
    @inbounds for i in 2:K
        v = lk[i]
        d = li[i]
        j = i
        while j > 1 && lk[j-1] < v
            lk = _tup_replace(lk, j, lk[j-1]);  li = _tup_replace(li, j, li[j-1])
            lk = _tup_replace(lk, j - 1, v);    li = _tup_replace(li, j - 1, d)
            j -= 1
        end
    end

    # merge into the running top-K. Reads only from registers, writes straight
    # to global — no aliasing.
    rv = ntuple(j -> out_vals[row, j], Val(K))
    ri = ntuple(j -> out_inds[row, j], Val(K))
    pa = 1; pb = 1
    @inbounds for j in 1:K
        va = pa <= K ? rv[pa] : typemin(T)
        vb = pb <= K ? lk[pb] : typemin(T)
        if va >= vb
            out_vals[row, j] = va
            out_inds[row, j] = ri[pa]
            pa += 1
        else
            out_vals[row, j] = vb
            idx = li[pb]
            out_inds[row, j] = idx == Int32(0) ? Int32(0) : idx + offset
            pb += 1
        end
    end
    return nothing
end

function launch_rowtopk_merge!(D_val, D_loc, C, offset::Integer, k::Integer;
                               threads::Integer=128)
    M = size(C, 1)
    @assert size(C, 2) >= 1 "chunk must be non-empty"
    @cuda threads=threads blocks=cld(M, threads) rowtopk_merge_kernel!(
        D_val, D_loc, C, Int32(offset), Val(Int(k)))
    return nothing
end

# --------------------------------------------------------------------------
# Segmented per-row top-k: PS threads per row instead of one.
#
# rowtopk_merge_kernel! assigns ONE thread per row, so its grid (and hence
# its memory-level parallelism) scales with M: at M = 2^13 it launches only
# 64 blocks and the scan runs ~4.7x below the bandwidth the same kernel
# reaches at M = 2^17 (160 vs 747 GiB/s of C read).  The flowtopk engine
# needs exactly that small M (one streamed batch = one (2^13 x 2^21) chunked
# GEMM), so this variant decouples parallelism from M: each row is scanned
# by PS threads, thread (row, s) covering the segment of W/PS columns
# [s*W/PS + 1, (s+1)*W/PS] with its own full-K register reservoir (same
# threshold/min-replace machinery, same unrolled scan).  The memory pattern
# is UNCHANGED: threads of a warp are 32 consecutive rows reading the same
# column (C is column-major -- coalescing is across rows, not columns), so
# global reads stay fully coalesced while the thread count grows PS x.
#
# Exactness: a true chunk top-K value has at most K-1 larger values in the
# whole row, hence in its own segment, so it survives that segment's
# K-reservoir; the union of the PS sorted segment lists contains the true
# chunk top-K.  After the scan each thread sorts its reservoir (same
# insertion sort) and stages it to shared memory; the block's first warp
# then, per row, walks the running top-K (preloaded to shared before any
# output write, since merge writes can lap the read cursor) against the
# chunk top-K produced by PS-way max-extraction over the segment heads --
# the same descending `va >= vb` merge as rowtopk_merge_kernel!.  Sentinel
# entries (typemin, 0) survive as (typemin, 0); ragged and empty segments
# (W < PS, ragged last chunk) degenerate to sentinel-only lists, exactly
# like the 1-thread kernel's guard columns.  Block shape: 32 rows x PS
# segments = 32*PS threads, cld(M, 32) blocks; one sync_threads between the
# scan/stage phases and the merge (no early returns -- invalid rows skip the
# work but reach the barrier).
# --------------------------------------------------------------------------
function seg_rowtopk_merge_kernel!(out_vals::CuDeviceMatrix{T}, out_inds::CuDeviceMatrix{Int32},
                                   C, offset::Int32, ::Val{K}, ::Val{PS}) where {T, K, PS}
    tid = threadIdx().x
    mt = ((tid - Int32(1)) & Int32(31)) + Int32(1) # row within the block's 32-row tile
    s0 = (tid - Int32(1)) >> Int32(5)            # this thread's segment (0-based)
    row = (blockIdx().x - Int32(1)) * Int32(32) + mt
    M = size(C, 1)
    valid = row <= M
    U = TOPK_UNROLL

    width = size(C, 2)
    segw = cld(width, PS)                        # segment width in columns
    seg_lo = s0 * segw + 1                       # my segment's first column (1-based)
    count = max(0, min(width, (s0 + 1) * segw) - seg_lo + 1)  # 0: empty segment

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(T), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(T)
    min_pos = 1

    if valid && count > 0
        # scan my segment: same software-pipelined, threshold-filtered scan as
        # rowtopk_merge_kernel!, with columns shifted into the segment; the
        # insertion index is segment-local, mapped to a global column below
        c = 1
        @inbounds begin
            n1 = count >= 1 ? C[row, seg_lo] : typemin(T)
            n2 = count >= 2 ? C[row, seg_lo + 1] : typemin(T)
            n3 = count >= 3 ? C[row, seg_lo + 2] : typemin(T)
            n4 = count >= 4 ? C[row, seg_lo + 3] : typemin(T)
        end
        @inbounds while c <= count
            # issue the whole next batch of U loads before processing
            m1 = c + U <= count     ? C[row, seg_lo + c + U - 1] : typemin(T)
            m2 = c + U + 1 <= count ? C[row, seg_lo + c + U]     : typemin(T)
            m3 = c + U + 2 <= count ? C[row, seg_lo + c + U + 1] : typemin(T)
            m4 = c + U + 3 <= count ? C[row, seg_lo + c + U + 2] : typemin(T)
            if n1 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n1, Int32(c), min_pos); end
            if n2 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n2, Int32(c + 1), min_pos); end
            if n3 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n3, Int32(c + 2), min_pos); end
            if n4 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n4, Int32(c + 3), min_pos); end
            n1 = m1; n2 = m2; n3 = m3; n4 = m4
            c += U
        end

        # sort the reservoir descending (once per thread)
        @inbounds for i in 2:K
            v = lk[i]
            d = li[i]
            j = i
            while j > 1 && lk[j-1] < v
                lk = _tup_replace(lk, j, lk[j-1]);  li = _tup_replace(li, j, li[j-1])
                lk = _tup_replace(lk, j - 1, v);    li = _tup_replace(li, j - 1, d)
                j -= 1
            end
        end
    end

    # shared memory: staged segment lists + heads, preloaded running list
    o = 0
    shv = CuDynamicSharedArray(T, (PS, 32, K));            o += PS * 32 * K * sizeof(T)
    shi = CuDynamicSharedArray(Int32, (PS, 32, K), o);     o += PS * 32 * K * sizeof(Int32)
    shp = CuDynamicSharedArray(Int16, (PS, 32), o);        o += PS * 32 * sizeof(Int16)
    shr_v = CuDynamicSharedArray(T, (32, K), o);           o += 32 * K * sizeof(T)
    shr_i = CuDynamicSharedArray(Int32, (32, K), o)        # (<= 48 KiB total for K=20)

    # stage my segment's sorted list (sentinels when the row/segment is empty)
    if valid
        @inbounds for j in 1:K
            shv[s0 + 1, mt, j] = lk[j]
            shi[s0 + 1, mt, j] = li[j]
        end
    end
    # preload the running top-K before any output write (merge writes can lap
    # the read cursor)
    if valid && tid <= Int32(32)
        @inbounds for j in 1:K
            shr_v[mt, j] = out_vals[row, j]
            shr_i[mt, j] = out_inds[row, j]
        end
    end
    sync_threads()

    # the block's first warp merges, per row: one fused pass that walks the
    # running top-K (preloaded above) against the chunk top-K produced by
    # PS-way max-extraction over the segment heads -- the same descending
    # `va >= vb` two-pointer decision as rowtopk_merge_kernel!, with the
    # chunk stream's replayable state living in (shv, shi, shp)
    if valid && tid <= Int32(32)
        for s in 1:PS
            @inbounds shp[s, mt] = Int16(1)
        end
        pa = 1
        for j in 1:K
            va = pa <= K ? @inbounds(shr_v[mt, pa]) : typemin(T)
            bv = typemin(T); bs = 0; bli = Int32(0)
            for s in 1:PS
                h = @inbounds shp[s, mt]
                if h <= K
                    v = @inbounds shv[s, mt, h]
                    if bv < v                     # ties keep the lowest segment
                        bv = v; bs = s; bli = @inbounds shi[s, mt, h]
                    end
                end
            end
            if va >= bv                           # running wins (or both done)
                @inbounds out_vals[row, j] = va
                @inbounds out_inds[row, j] = pa <= K ? shr_i[mt, pa] : Int32(0)
                pa += 1
            else
                @inbounds out_vals[row, j] = bv
                @inbounds out_inds[row, j] = bli == Int32(0) ? Int32(0) :
                                             offset + Int32((bs - 1) * segw) + bli
                @inbounds shp[bs, mt] += Int16(1)
            end
        end
    end
    return nothing
end

function launch_seg_rowtopk_merge!(D_val::CuMatrix{T}, D_loc::CuMatrix{Int32},
                                   C, offset::Integer, k::Integer;
                                   segs::Integer = 8) where {T}
    M = size(C, 1)
    @assert size(C, 2) >= 1 "chunk must be non-empty"
    PS = Int(segs)
    K = Int(k)
    @assert 1 <= PS && 32 * PS <= 1024 "segs must be in 1:32"
    shmem = PS * 32 * K * (sizeof(T) + sizeof(Int32)) + PS * 32 * sizeof(Int16) +
            32 * K * (sizeof(T) + sizeof(Int32))
    @assert shmem <= 48 * 1024 "seg_rowtopk_merge_kernel! needs $(shmem) B of shared memory " *
                               "(> 48 KiB); lower `segs` (K=20, fp16 fits segs <= 10)"
    @cuda threads = (32 * PS) blocks = cld(M, 32) shmem = shmem seg_rowtopk_merge_kernel!(
        D_val, D_loc, C, Int32(offset), Val(K), Val(PS))
    return nothing
end

# --------------------------------------------------------------------------
# Double-buffered batched GEMM + top-k pipeline.
#
# `w` is the chunk width (columns of B per chunk); buffer usage is 2·M·w fp16.
# `overlap=false` runs everything in order on the default stream (useful as a
# benchmark baseline; results are identical — the kernels are deterministic).
# --------------------------------------------------------------------------
function gemm_topk!(D_val::CuMatrix{T}, D_loc::CuMatrix{Int32},
                    A::CuMatrix{T}, B::CuMatrix{T};
                    k::Integer=20, w::Integer=2^13, overlap::Bool=true,
                    threads::Integer=128, max_chunks::Integer=typemax(Int)) where {T}
    M, KA = size(A)
    KB, N = size(B)
    @assert KA == KB "inner dimensions of A and B must match"
    @assert size(D_val) == (M, k) && size(D_loc) == (M, k)
    @assert 1 <= k <= w <= N

    # reset the running top-k; the kernel merges into it chunk by chunk
    fill!(D_val, typemin(T))
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    w = Int(w)
    nb = min(cld(N, w), Int(max_chunks))
    bufs = [CuMatrix{T}(undef, M, w), CuMatrix{T}(undef, M, w)]

    # with overlap=false everything runs sequentially on the default stream and
    # the event waits are trivially satisfied — same results, no hiding.
    sg = overlap ? CuStream(; flags=CUDA.STREAM_NON_BLOCKING) : CUDA.default_stream()
    st = overlap ? CuStream(; flags=CUDA.STREAM_NON_BLOCKING) : sg
    evg = (CuEvent(), CuEvent())   # gemm of the chunk using buffer b is done
    evt = (CuEvent(), CuEvent())   # top-k of the chunk using buffer b is done

    blocks = cld(M, threads)
    for i in 1:nb
        b = mod1(i, 2)
        lo = (i - 1) * w + 1
        wi = min(N, i * w) - lo + 1
        Bv = @view B[:, lo:lo+wi-1]
        Cv = @view bufs[b][:, 1:wi]

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(evt[b])       # buffer b's previous top-k finished
            mul!(Cv, A, Bv)                  # cuBLAS fp16 GEMM (tensor cores)
            CUDA.record(evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(evg[b])                # C chunk is ready
            launch_rowtopk_merge!(D_val, D_loc, Cv, lo - 1, k; threads)
            CUDA.record(evt[b])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

function gemm_topk(A::CuMatrix{T}, B::CuMatrix{T}; k::Integer=20, kwargs...) where {T}
    D_val = CuMatrix{T}(undef, size(A, 1), k)
    D_loc = CuMatrix{Int32}(undef, size(A, 1), k)
    gemm_topk!(D_val, D_loc, A, B; k, kwargs...)
    return D_val, D_loc
end

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# fills an fp16 matrix with standard normals, chunk by chunk (fp32 generation,
# broadcast-converted; avoids any fp16 RNG edge cases)
function randn_fp16!(X::CuMatrix{Float16}; chunk::Integer=2^11)
    M, N = size(X)
    tmp = CuMatrix{Float32}(undef, M, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(tmp)
        X[:, lo:lo+wi-1] .= @view tmp[:, 1:wi]
    end
    return X
end

# exact multiset check against a CPU sort, over several shapes including ragged
# chunk sizes (w ∤ N), a single row, and the exact full-matrix chunk size
function test_rowtopk_merge_kernel(; k::Integer=20)
    CUDA.seed!(123)
    for (M, N, w) in ((64, 100, 37), (257, 128, 128), (1000, 333, 111), (1, 4096, 512), (511, 2048, 2048))
        C = CuMatrix{Float16}(randn(Float32, M, N))
        D_val = CUDA.fill(typemin(Float16), M, k)
        D_loc = CUDA.zeros(Int32, M, k)
        for i in 1:cld(N, w)
            lo = (i - 1) * w + 1
            wi = min(N, i * w) - lo + 1
            buf = CuMatrix{Float16}(@view C[:, lo:lo+wi-1])
            launch_rowtopk_merge!(D_val, D_loc, buf, lo - 1, k)
        end
        Ch = Array(C)
        Dv = Array(D_val)
        Di = Array(D_loc)
        for r in 1:M
            ref = sort(Ch[r, :], rev=true)[1:k]
            got = Dv[r, :]
            @assert issorted(got; rev=true) "row $r of ($M,$N,$w): output not sorted"
            @assert sort(got; rev=true) == ref "row $r of ($M,$N,$w): wrong top-$k values"
            @assert all(>(0), Di[r, :]) && Ch[r, Di[r, :]] == got "row $r of ($M,$N,$w): wrong locations"
        end
    end
    @info "unit tests passed: rowtopk_merge_kernel! matches CPU reference"
    return nothing
end

# exact multiset check for the segmented (PS-threads-per-row) variant, over
# the same shapes as the 1-thread test plus: strided parent views (engine-
# style ld = w buffers viewed to wi < w), w == k (exact fit), an empty-
# segment case (w = 25 with segs = 8 -> segw = 4, segment 8 empty), and a
# real-geometry chunk
function test_seg_rowtopk_merge_kernel(; k::Integer=20, segs::Integer=8)
    CUDA.seed!(321)
    for (M, N, w) in ((64, 100, 37), (257, 128, 128), (1000, 333, 111), (1, 4096, 512),
                      (511, 2048, 2048), (2048, 2^14, 2^13), (33, 37, 37), (64, 25, 25),
                      (64, 20, 20))
        C = CuMatrix{Float16}(randn(Float32, M, N))
        D_val = CUDA.fill(typemin(Float16), M, k)
        D_loc = CUDA.zeros(Int32, M, k)
        for i in 1:cld(N, w)
            lo = (i - 1) * w + 1
            wi = min(N, i * w) - lo + 1
            parent = CuMatrix{Float16}(undef, M, w)   # engine-style buffer, ld = w
            pview = @view parent[:, 1:wi]         # strided when wi < w
            copyto!(pview, @view C[:, lo:lo+wi-1])
            launch_seg_rowtopk_merge!(D_val, D_loc, pview, lo - 1, k; segs)
        end
        Ch = Array(C)
        Dv = Array(D_val)
        Di = Array(D_loc)
        for r in 1:M
            ref = sort(Ch[r, :], rev=true)[1:k]
            got = Dv[r, :]
            @assert issorted(got; rev=true) "row $r of ($M,$N,$w): output not sorted"
            @assert sort(got; rev=true) == ref "row $r of ($M,$N,$w): wrong top-$k values"
            @assert all(>(0), Di[r, :]) && Ch[r, Di[r, :]] == got "row $r of ($M,$N,$w): wrong locations"
        end
    end
    @info "unit tests passed: seg_rowtopk_merge_kernel! matches CPU reference"
    return nothing
end

# end-to-end accuracy: recompute a few random rows of C = A*B exactly in fp32
# (chunked gemv, B streamed through a small fp32 buffer) and compare top-k
function spot_check(A::CuMatrix{Float16}, B::CuMatrix{Float16},
                    D_val::CuMatrix{Float16}, D_loc::CuMatrix{Int32}, k::Integer;
                    nrows::Integer=8, seed::Integer=42)
    Random.seed!(seed)
    M, K = size(A)
    N = size(B, 2)
    wref = 2^15                          # fp32 reference chunks (fits easily)
    a32 = CuVector{Float32}(undef, K)
    B32 = CuMatrix{Float32}(undef, K, wref)
    ref = CuVector{Float32}(undef, N)

    maxvalerr = 0.0
    locbad = 0
    for _ in 1:nrows
        r = rand(1:M)
        a32 .= @view A[r, :]                       # fp16 → fp32
        fill!(ref, 0f0)
        for lo in 1:wref:N
            wi = min(N, lo + wref - 1) - lo + 1
            B32[:, 1:wi] .= @view B[:, lo:lo+wi-1]
            # ref chunk = B32ᵀ * a32  (fp32 gemv)
            mul!(view(ref, lo:lo+wi-1), transpose(view(B32, :, 1:wi)), a32)
        end
        refh = Array(ref)
        refv = sort(refh; rev=true)[1:k]
        gotv = Array(@view D_val[r, :])
        goti = Array(@view D_loc[r, :])

        @assert all(1 .<= goti .<= N) "row $r: locations out of range"
        @assert issorted(gotv; rev=true) "row $r: values not sorted"
        maxvalerr = max(maxvalerr, maximum(abs.(refv .- gotv)))
        # a location is "wrong" only if the value stored there disagrees with
        # the claimed top-k value by more than fp16 accumulation noise
        locbad += count(abs.(refh[goti] .- gotv) .> 1.0)
    end
    @assert maxvalerr < 2.0 "top-$k values deviate too much from fp32 reference (max err $maxvalerr)"
    @assert locbad == 0 "$locbad locations are inconsistent with their values"
end
