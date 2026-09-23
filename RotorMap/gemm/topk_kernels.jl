# ==============================================================================
# topk_kernels.jl — fused top-k reduction kernels (scan + running merge).
#
# PURPOSE
#   GPU kernels that scan a C chunk per row into a register-resident top-K
#   reservoir (threshold-filtered, minimum-replace) and merge it into the
#   running top-K in D_val/D_loc, plus their launchers and an exact
#   CPU-reference unit test:
#     rowtopk_merge_kernel!       one thread per row, (Tc, Tv) version
#     seg_rowtopk_merge_kernel!   PS threads per row (parallelism decoupled
#                                 from M), (Tc, Tv) version
#     seg_rowtopk_cmerge_kernel!  complex-magnitude variant: scans the fused
#                                 abs2 score of an (Re, Im) chunk pair
#
# SOURCES
#   legacy/test/gemmtopkfp8.jl: TOPK_UNROLL (deduplicated — gemmtopk.jl
#     defined the identical const), _scanval (all 3 methods),
#     _tup_argmin/_tup_replace/_tup_insert!, rowtopk_merge_kernel! +
#     launch_rowtopk_merge! (the 6-arg row_offset version).
#   legacy/test/flowtopkfp8.jl: seg_rowtopk_merge_kernel! (the _scanval
#     version), launch_seg_rowtopk_merge!, test_seg_rowtopk_merge_kernel.
#   legacy/test/e2ecomplex.jl: _cscore, seg_rowtopk_cmerge_kernel!,
#     launch_seg_rowtopk_cmerge!.
#
# DEPS
#   include fp8_convert.jl (F8 — used by _scanval and the unit test);
#   using CUDA, Random.
#
# NOTES
#   - Deduplication: the single-type-T rowtopk_merge_kernel! (5-arg) and
#     seg_rowtopk_merge_kernel! from the fp16 script are SUPERSEDED and not
#     carried.  The (Tc, Tv) seg kernel is numerically identical for
#     Float16/Float32 inputs (the _scanval widening is exact); only the
#     reservoir/D_val currency is fixed to Float32 by its launcher.
#   - launch_rowtopk_merge!/launch_seg_rowtopk_merge! require Float32 running
#     lists; the fp16 stage (fp16.jl) inlines its own Tv-generic launcher.
#   - Engine/pipeline unit tests (all-types 1-thread test, engine checks) are
#     not carried here — this file holds kernels only.
# ==============================================================================

include(joinpath(@__DIR__, "fp8_convert.jl"))

using CUDA
using Random

# --------------------------------------------------------------------------
# Kernel: fused per-row top-k scan + running merge (the (Tc, Tv)
# generalization of the single-type fp16 kernel: C chunks are read as Tc
# into a Tv reservoir).
#
# One thread per row of the C chunk. Scans the row into a register-resident
# unsorted top-K reservoir (threshold-filtered, minimum-replace), sorts it
# once (descending), then two-pointer-merges it into the running top-K in
# D_val/D_loc. Fresh indices are chunk-local (1..width) and are shifted by
# `offset` into global column ids. The scan is unrolled TOPK_UNROLLx with the
# next batch of loads issued before the current batch is processed.
# IMPORTANT: all sentinels (out-of-range loads, reservoir init, threshold)
# are typemin(Tv) — NOT typemin(Tc) converted, which for Tc=Float16/Tv=Float32
# (-65504 vs -3.4e38) would wrongly enter the reservoir.
# --------------------------------------------------------------------------
const TOPK_UNROLL = 4

# scan value conversion: C chunks are read as Tc; fp8-out chunks are raw
# e4m3 bytes (CuMatrix{UInt8}) decoded on the fly (exact, ~10 integer ops).
@inline _scanval(::Type{Float16}, x::Float16) = Float32(x)
@inline _scanval(::Type{Float32}, x::Float32) = x
@inline _scanval(::Type{UInt8}, x::UInt8) = Float32(reinterpret(F8, x))

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

function rowtopk_merge_kernel!(out_vals::CuDeviceMatrix{Tv}, out_inds::CuDeviceMatrix{Int32},
                               C::CuDeviceMatrix{Tc}, offset::Int32, row_offset::Int32,
                               ::Val{K}) where {Tc, Tv, K}
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M = size(C, 1)
    row <= M || return nothing
    drow = row + row_offset          # this row's slot in the running top-k
    U = TOPK_UNROLL

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(Tv), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(Tv)            # current reservoir minimum (register)
    min_pos = 1                      # argmin(lk)

    width = size(C, 2)
    c = 1
    # rolling window of U in-flight loads; nᵢ holds column c+i-1 (as Tv)
    @inbounds begin
        n1 = 1 <= width ? _scanval(Tc, C[row]) : typemin(Tv)
        n2 = 2 <= width ? _scanval(Tc, C[row + M]) : typemin(Tv)
        n3 = 3 <= width ? _scanval(Tc, C[row + 2 * M]) : typemin(Tv)
        n4 = 4 <= width ? _scanval(Tc, C[row + 3 * M]) : typemin(Tv)
    end
    @inbounds while c <= width
        # issue the whole next batch of U loads before processing (latency hiding)
        m1 = c + U     <= width ? _scanval(Tc, C[row + (c + U - 1) * M]) : typemin(Tv)
        m2 = c + U + 1 <= width ? _scanval(Tc, C[row + (c + U) * M]) : typemin(Tv)
        m3 = c + U + 2 <= width ? _scanval(Tc, C[row + (c + U + 1) * M]) : typemin(Tv)
        m4 = c + U + 3 <= width ? _scanval(Tc, C[row + (c + U + 2) * M]) : typemin(Tv)
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
    rv = ntuple(j -> out_vals[drow, j], Val(K))
    ri = ntuple(j -> out_inds[drow, j], Val(K))
    pa = 1; pb = 1
    @inbounds for j in 1:K
        va = pa <= K ? rv[pa] : typemin(Tv)
        vb = pb <= K ? lk[pb] : typemin(Tv)
        if va >= vb
            out_vals[drow, j] = va
            out_inds[drow, j] = ri[pa]
            pa += 1
        else
            out_vals[drow, j] = vb
            idx = li[pb]
            out_inds[drow, j] = idx == Int32(0) ? Int32(0) : idx + offset
            pb += 1
        end
    end
    return nothing
end

function launch_rowtopk_merge!(D_val::CuMatrix{<:AbstractFloat}, D_loc::CuMatrix{Int32},
                               C::CuMatrix, offset::Integer, k::Integer;
                               threads::Integer = 128, row_offset::Integer = 0)
    M = size(C, 1)
    @assert size(C, 2) >= 1 "chunk must be non-empty"
    @cuda threads = threads blocks = cld(M, threads) rowtopk_merge_kernel!(
        D_val, D_loc, C, Int32(offset), Int32(row_offset), Val(Int(k)))
    return nothing
end

# --------------------------------------------------------------------------
# Segmented per-row top-k: PS threads per row instead of one.
#
# rowtopk_merge_kernel! assigns ONE thread per row, so its grid (and hence
# its memory-level parallelism) scales with M: at M = 2^13 it launches only
# 64 blocks and the scan runs ~4.7x below the bandwidth the same kernel
# reaches at M = 2^17 (160 vs 747 GiB/s of C read).  Streamed-batch engines
# need exactly that small M, so this variant decouples parallelism from M:
# each row is scanned by PS threads, thread (row, s) covering the segment of
# W/PS columns [s*W/PS + 1, (s+1)*W/PS] with its own full-K register
# reservoir (same threshold/min-replace machinery, same unrolled scan).  The
# memory pattern is UNCHANGED: threads of a warp are 32 consecutive rows
# reading the same column (C is column-major -- coalescing is across rows,
# not columns), so global reads stay fully coalesced while the thread count
# grows PS x.
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
#
# This is the (Tc, Tv) split version and SUPERSEDES the single-type-T
# original (same name + arity): C chunks are read as Tc ∈ {Float16, Float32,
# UInt8 = raw e4m3 bytes} via _scanval, the reservoir / shared staging /
# D_val are Tv = Float32.  All sentinels are typemin(Tv) DIRECTLY.  For
# Float16/Float32 inputs it is numerically identical to the old fp16 kernel
# (the _scanval widening is exact, comparisons/sorting/merging see the same
# order); the launcher fixes the Float32 currency.
# --------------------------------------------------------------------------
function seg_rowtopk_merge_kernel!(out_vals::CuDeviceMatrix{Tv}, out_inds::CuDeviceMatrix{Int32},
                                   C, offset::Int32, ::Val{K}, ::Val{PS}) where {Tv, K, PS}
    tid = threadIdx().x
    mt = ((tid - Int32(1)) & Int32(31)) + Int32(1) # row within the block's 32-row tile
    s0 = (tid - Int32(1)) >> Int32(5)              # this thread's segment (0-based)
    row = (blockIdx().x - Int32(1)) * Int32(32) + mt
    M = size(C, 1)
    valid = row <= M
    U = TOPK_UNROLL

    width = size(C, 2)
    segw = cld(width, PS)                        # segment width in columns
    seg_lo = s0 * segw + 1                       # my segment's first column (1-based)
    count = max(0, min(width, (s0 + 1) * segw) - seg_lo + 1)  # 0: empty segment

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(Tv), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(Tv)
    min_pos = 1

    if valid && count > 0
        Tc = eltype(C)
        # scan my segment: software-pipelined, threshold-filtered; the
        # insertion index is segment-local, mapped to a global column below
        c = 1
        @inbounds begin
            n1 = count >= 1 ? _scanval(Tc, C[row, seg_lo]) : typemin(Tv)
            n2 = count >= 2 ? _scanval(Tc, C[row, seg_lo + 1]) : typemin(Tv)
            n3 = count >= 3 ? _scanval(Tc, C[row, seg_lo + 2]) : typemin(Tv)
            n4 = count >= 4 ? _scanval(Tc, C[row, seg_lo + 3]) : typemin(Tv)
        end
        @inbounds while c <= count
            # issue the whole next batch of U loads before processing
            m1 = c + U <= count     ? _scanval(Tc, C[row, seg_lo + c + U - 1]) : typemin(Tv)
            m2 = c + U + 1 <= count ? _scanval(Tc, C[row, seg_lo + c + U])     : typemin(Tv)
            m3 = c + U + 2 <= count ? _scanval(Tc, C[row, seg_lo + c + U + 1]) : typemin(Tv)
            m4 = c + U + 3 <= count ? _scanval(Tc, C[row, seg_lo + c + U + 2]) : typemin(Tv)
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
    shv = CuDynamicSharedArray(Tv, (PS, 32, K));           o += PS * 32 * K * sizeof(Tv)
    shi = CuDynamicSharedArray(Int32, (PS, 32, K), o);     o += PS * 32 * K * sizeof(Int32)
    shp = CuDynamicSharedArray(Int16, (PS, 32), o);        o += PS * 32 * sizeof(Int16)
    shr_v = CuDynamicSharedArray(Tv, (32, K), o);          o += 32 * K * sizeof(Tv)
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

    # the block's first warp merges, per row: PS-way max-extraction over the
    # segment heads against the preloaded running list
    if valid && tid <= Int32(32)
        for s in 1:PS
            @inbounds shp[s, mt] = Int16(1)
        end
        pa = 1
        for j in 1:K
            va = pa <= K ? @inbounds(shr_v[mt, pa]) : typemin(Tv)
            bv = typemin(Tv); bs = 0; bli = Int32(0)
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

function launch_seg_rowtopk_merge!(D_val::CuMatrix{<:AbstractFloat}, D_loc::CuMatrix{Int32},
                                   C, offset::Integer, k::Integer;
                                   segs::Integer = 8)
    M = size(C, 1)
    @assert size(C, 2) >= 1 "chunk must be non-empty"
    PS = Int(segs)
    K = Int(k)
    @assert 1 <= PS && 32 * PS <= 1024 "segs must be in 1:32"
    Tv = eltype(D_val) # Float32 in the fp8 flow; Float16 in the legacy fp16 engine
    shmem = PS * 32 * K * (sizeof(Tv) + sizeof(Int32)) + PS * 32 * sizeof(Int16) +
            32 * K * (sizeof(Tv) + sizeof(Int32))
    @assert shmem <= 48 * 1024 "seg_rowtopk_merge_kernel! needs $(shmem) B of shared memory " *
                               "(> 48 KiB); lower `segs` (K = 20, Float32 reservoir: segs <= 8 fits)"
    @cuda threads = (32 * PS) blocks = cld(M, 32) shmem = shmem seg_rowtopk_merge_kernel!(
        D_val, D_loc, C, Int32(offset), Val(K), Val(PS))
    return nothing
end

# ==============================================================================
# The COMPLEX segmented per-row top-k kernel: the segmented
# seg_rowtopk_merge_kernel! with the scan value changed from the raw C entry
# to the FUSED abs2 score of the (Re, Im) chunk pair:
#     score(j) = Float32(Cr[row, j])^2 + Float32(Ci[row, j])^2
# (the norm/magnitude computation of the complex inner product).  Everything
# else -- PS threads per 32-row tile, the register Float32 reservoir with
# tracked minimum, the software-pipelined threshold-filtered scan, the sort +
# PS-way merge mechanics, the sentinels at typemin(Float32) -- is verbatim
# (scores are >= 0, so the sentinels stay correct).  NB: the GPU may contract
# r*r + i*i into mul+FMA -- tests compare values with a few-ulp tolerance
# (and cross-check locations via values).
# ==============================================================================

@inline function _cscore(::Type{Tc}, Cr, Ci, row, col) where {Tc}
    @inbounds r = Cr[row, col]
    @inbounds im = Ci[row, col]
    rf = Float32(r)
    imf = Float32(im)
    return rf * rf + imf * imf
end

function seg_rowtopk_cmerge_kernel!(out_vals::CuDeviceMatrix{Float32},
                                    out_inds::CuDeviceMatrix{Int32},
                                    Cr, Ci, offset::Int32, ::Val{K}, ::Val{PS}) where {K, PS}
    tid = threadIdx().x
    mt = ((tid - Int32(1)) & Int32(31)) + Int32(1) # row within the block's 32-row tile
    s0 = (tid - Int32(1)) >> Int32(5)              # this thread's segment (0-based)
    row = (blockIdx().x - Int32(1)) * Int32(32) + mt
    M = size(Cr, 1)
    valid = row <= M
    U = TOPK_UNROLL

    width = size(Cr, 2)
    segw = cld(width, PS)                        # segment width in columns
    seg_lo = s0 * segw + 1                       # my segment's first column (1-based)
    count = max(0, min(width, (s0 + 1) * segw) - seg_lo + 1)  # 0: empty segment

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(Float32), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(Float32)
    min_pos = 1

    if valid && count > 0
        Tc = eltype(Cr)
        # scan my segment: software-pipelined, threshold-filtered; each
        # "value" is the abs2 score of one (Cr, Ci) column pair; the
        # insertion index is segment-local, mapped to a global column below
        c = 1
        @inbounds begin
            n1 = count >= 1 ? _cscore(Tc, Cr, Ci, row, seg_lo) : typemin(Float32)
            n2 = count >= 2 ? _cscore(Tc, Cr, Ci, row, seg_lo + 1) : typemin(Float32)
            n3 = count >= 3 ? _cscore(Tc, Cr, Ci, row, seg_lo + 2) : typemin(Float32)
            n4 = count >= 4 ? _cscore(Tc, Cr, Ci, row, seg_lo + 3) : typemin(Float32)
        end
        @inbounds while c <= count
            # issue the whole next batch of U (paired) loads before processing
            m1 = c + U <= count     ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U - 1) : typemin(Float32)
            m2 = c + U + 1 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U)     : typemin(Float32)
            m3 = c + U + 2 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U + 1) : typemin(Float32)
            m4 = c + U + 3 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U + 2) : typemin(Float32)
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
    shv = CuDynamicSharedArray(Float32, (PS, 32, K));      o += PS * 32 * K * sizeof(Float32)
    shi = CuDynamicSharedArray(Int32, (PS, 32, K), o);     o += PS * 32 * K * sizeof(Int32)
    shp = CuDynamicSharedArray(Int16, (PS, 32), o);        o += PS * 32 * sizeof(Int16)
    shr_v = CuDynamicSharedArray(Float32, (32, K), o);     o += 32 * K * sizeof(Float32)
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

    # the block's first warp merges, per row: PS-way max-extraction over the
    # segment heads against the preloaded running list (verbatim)
    if valid && tid <= Int32(32)
        for s in 1:PS
            @inbounds shp[s, mt] = Int16(1)
        end
        pa = 1
        for j in 1:K
            va = pa <= K ? @inbounds(shr_v[mt, pa]) : typemin(Float32)
            bv = typemin(Float32); bs = 0; bli = Int32(0)
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

function launch_seg_rowtopk_cmerge!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                                    Cr, Ci, offset::Integer, k::Integer;
                                    segs::Integer = 8)
    M = size(Cr, 1)
    @assert size(Ci) == size(Cr) "Re/Im chunks must have the same shape"
    @assert size(Cr, 2) >= 1 "chunk must be non-empty"
    PS = Int(segs)
    K = Int(k)
    @assert 1 <= PS && 32 * PS <= 1024 "segs must be in 1:32"
    shmem = PS * 32 * K * (sizeof(Float32) + sizeof(Int32)) + PS * 32 * sizeof(Int16) +
            32 * K * (sizeof(Float32) + sizeof(Int32))
    @assert shmem <= 48 * 1024 "seg_rowtopk_cmerge_kernel! needs $(shmem) B of shared memory " *
                               "(> 48 KiB); lower `segs` (K = 20, Float32 reservoir: segs <= 8 fits)"
    @cuda threads = (32 * PS) blocks = cld(M, 32) shmem = shmem seg_rowtopk_cmerge_kernel!(
        D_val, D_loc, Cr, Ci, Int32(offset), Val(K), Val(PS))
    return nothing
end

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# exact multiset check against a CPU sort for the segmented (Tc, Tv) kernel,
# for all C element types (fp16, fp32 and raw e4m3 bytes), over several shapes
# including ragged chunk sizes (w ∤ N), a single row, the exact full-matrix
# chunk size, engine-style strided parent views (ld = w viewed to wi < w),
# w == k and an empty-segment case (w = 25 with segs = 8 -> segment 8 empty)
function test_seg_rowtopk_merge_kernel(; k::Integer = 20, segs::Integer = 8)
    CUDA.seed!(321)
    for Tc in (Float16, Float32, UInt8)
        for (M, N, w) in ((64, 100, 37), (257, 128, 128), (1000, 333, 111), (1, 4096, 512),
                          (511, 2048, 2048), (2048, 2^14, 2^13), (33, 37, 37), (64, 25, 25),
                          (64, 20, 20))
            Cf = randn(Float32, M, N)
            C8 = Tc === UInt8 ? CuMatrix{F8}(Cf) : nothing
            C = Tc === UInt8 ? reinterpret(UInt8, C8) : CuMatrix{Tc}(Cf)
            Ch = Tc === UInt8 ? Float32.(Array(C8)) : Array(C)
            D_val = CUDA.fill(typemin(Float32), M, k)
            D_loc = CUDA.zeros(Int32, M, k)
            for i in 1:cld(N, w)
                lo = (i - 1) * w + 1
                wi = min(N, i * w) - lo + 1
                parent = CuMatrix{Tc}(undef, M, w)   # engine-style buffer, ld = w
                pview = @view parent[:, 1:wi]        # strided when wi < w
                copyto!(pview, @view C[:, lo:lo+wi-1])
                launch_seg_rowtopk_merge!(D_val, D_loc, pview, lo - 1, k; segs)
            end
            Dv = Array(D_val)
            Di = Array(D_loc)
            for r in 1:M
                ref = sort(Ch[r, :]; rev = true)[1:k]
                got = Dv[r, :]
                @assert issorted(got; rev = true) "row $r of ($M,$N,$w,$Tc): output not sorted"
                @assert sort(got; rev = true) == ref "row $r of ($M,$N,$w,$Tc): wrong top-$k values"
                @assert all(>(0), Di[r, :]) && Ch[r, Di[r, :]] == got "row $r of ($M,$N,$w,$Tc): wrong locations"
            end
        end
    end
    @info "unit tests passed: seg_rowtopk_merge_kernel! matches CPU reference " *
          "(Tc=Float16,Float32,e4m3 bytes; ragged/strided/w==k/empty-segment shapes)"
    return nothing
end
