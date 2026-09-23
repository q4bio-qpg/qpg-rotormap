# ==============================================================================
# ropeflowreal.jl -- ropeflow_v3 with REAL output vectors: the rope encodings
# are emitted as concatenated real/imaginary parts instead of ComplexF32, plus
# an fp16 option for the final output.  Same three-stage channel pipeline
# (fasta_reads -> gather -> encode -> Channel{RopeRealBatch}), same kernel
# semantics, same teardown contracts.
#
# THE REAL LAYOUT (a FIXED contract for consumers -- never varies per call):
#   a fragment's complex embedding is a D = m*4^c vector over s-mer bins (bin
#   = (csmer-1)*m + idm, the v3 convention); its real form is the 2D-vector
#       [ Re(b_1) .. Re(b_D) , Im(b_1) .. Im(b_D) ]
#   i.e. the SPLIT/planar order: ALL REAL PARTS FIRST, then ALL IMAGINARY
#   PARTS.  Chosen over interleaved because the codebase's downstream
#   consumers (SplitComplexMatrix(.Re, .Im), Mapper.jl) want contiguous
#   blocks: row r's Re block is embeds[r, 1:D], its Im block embeds[r, D+1:2D]
#   -- plain contiguous slices, no strides.  Consumer guarantees: the real
#   inner product equals the complex one (<x, y> = Re<z_x, z_y>) and real
#   distance equals complex distance (|x - y|^2 = |z_x - z_y|^2).  (The
#   shared-memory histogram stays interleaved exactly as in kernel_rope_frag;
#   only the final shared->global copy splits it into the two halves.)
#
# THE FP16 OPTION (fp16 = true on the stream / T = Float16 on the encoders):
#   compute and the norms stay Float32; only the OUTPUT storage is Float16 --
#   the conversion happens inside the kernel (no intermediate buffer, one
#   fused pass over shared memory), so the embedding D2H transfer HALVES
#   (~4 KiB/fragment at the default config instead of ~8 KiB; the D2H was the
#   ropeflow_v3 bottleneck).  norms remain Float32 (m, nb) regardless.  The
#   unit-energy invariant survives to ~1e-2 relative (fp16 rounding).
#
# Everything else -- kernel semantics (little-endian s-mer fields, scramble
# key, G-bin zeroing, normalize 0-3, exp(+i*theta*p) phases, direct staging,
# 512 threads, maxregs = 128), pipeline topology, backpressure, pinned buffer
# recycling, early-close teardown, err_out/tasks_out, run modes -- is
# ropeflow_v3 verbatim; that file is included for fasta_reads, ensure_data3,
# the complex CPU reference (_ref_rope_frag) and the harness helpers.
#
# Run modes (ARGV[1]):
#   gen   generate the test files if missing (fastareads_v3's ensure_data3)
#   test  correctness: float64 CPU reference of the exact kernel semantics for
#         the REAL output (all normalize modes, fp32 and fp16 outputs, several
#         s/m/c configs, tail/guard edge ks), end-to-end flow vs the CPU
#         reference on the 164 MB file (batch layouts, heads, fp16 batches,
#         sampled rows, unit energy), early-close teardown, empty file
#   bench parse-only vs full pipeline (fp32 and fp16, no-op and collecting
#         sinks) on the 2.6 GB file, plus the kernel alone
#   all   everything above (default)
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16, min of 3, big file =
# 2.6 GB, k = 20,000, s = 8, m = 4, c = 4 -> dim = 2*m*4^c = 2048,
# batch_size = 2^13):
#
#   fasta_reads drain (parse only)                       0.272 s    9.63 GB/s
#   real stream, no-op consumer fp32 (H2D+kernel+D2H)    0.620 s    4.23 GB/s
#   real stream, no-op consumer fp16 (H2D+kernel+D2H)    0.531 s    4.94 GB/s
#   real stream, collecting sink fp32 (pinned)           0.503 s    5.21 GB/s
#   real stream, collecting sink fp16 (pinned)           0.474 s    5.53 GB/s
#   kernel_rope_frag_real only (2^15 x 20 kb, fp32)      0.021 s   31.08 GB/s
#   kernel_rope_frag_real only (2^15 x 20 kb, fp16)      0.021 s   30.88 GB/s
#
# The fp32 lines match ropeflow_v3's complex pipeline within noise (the real
# fp32 output has the same D2H byte count: 2048 x 4 B = complex 1024 x 8 B);
# fp16 halves the embedding D2H and buys ~6-14% end-to-end (more with the
# no-op consumer, whose D2H is pageable; the pinned collecting sink is
# already copy-bound).  The kernel is unaffected by T: the compute and the
# shared-memory traffic are Float32 either way, the fp16 store is free.
#
# Correctness (all on kau, julia -t 16): kernel vs the same independent
# float64 CPU reference (_ref_rope_frag) wrapped with the [Re; Im] flattening
# -- all normalize modes 0-3 x fp32/fp16 outputs, configs (5,1,4), (16,2,4),
# (8,1,6), tail/guard edge fragment lengths k = 2001 (tail mask, fp32+fp16)
# and 17 (nw < block), the real k = 20000; unit energy via
# |x|^2 = |Re|^2 + |Im|^2 everywhere.  End-to-end stream on the 164 MB file:
# 4 batch layouts fp32, normalize 0/1 x fp32/fp16 (eltype asserted per
# batch), configs (5,1,4)/(16,2,4), heads set-equality, sampled full-value
# rows.  Deterministic early close (tasks joined, err_out clean, pipeline
# reusable); empty file -> 0 batches.  ALL PASS.
# ==============================================================================

include(joinpath(@__DIR__, "ropeflow_v3.jl")) # fasta_reads, RopeEncoder,
# ensure_data3, _ref_rope_frag (the complex CPU reference), _frag_codes,
# _timed_min/_warm_cache/_reps, _V3_SMALL_READS/_V3_BIG_READS (+ the whole
# transitive v2/v3 context)

using Random
using CUDA
using Printf
using ProgressMeter
using RotorMap
using RotorMap.RopeEncoders
using Base.Threads

# ==============================================================================
# The forward-fragment kernel, real output
# ==============================================================================

"""
Real-output counterpart of `kernel_rope_frag` (ropeflow_v3.jl) -- identical
semantics and work distribution; only the output stage differs:

  * `embeds` holds REAL vectors: an (n, 2*m*4^c) row-major-batch matrix where
    row `idr` is the SPLIT concatenation [Re; Im] of the fragment's complex
    histogram -- `embeds[idr, bin] = Re(bin)`, `embeds[idr, bin + m*4^c] =
    Im(bin)` (coalesced per block in both halves);
  * the output eltype is `T` (Float32, or Float16 with the `fp16` option --
    the conversion from the Float32 shared histogram happens right here in
    the kernel, so no intermediate buffer and a half-size D2H);
  * `embeds_norms[t, idr]` (m x n, Float32 always) is unchanged.

See ropeflow_v3.jl for the s-mer fields, scramble key, G-bin convention,
phase convention and normalize modes 0-3 (all verbatim).
"""
function kernel_rope_frag_real(embeds::CuDeviceArray{T}, embeds_norms, dnas,
                               s, m, c, k, W, n, normalize) where {T<:Real}
    idr = blockIdx().x # one block per fragment
    idr > n && return

    t = threadIdx().x
    B = blockDim().x

    nw = Int32(k) - Int32(s) + Int32(1) # number of sliding windows (0: degenerate)
    nbin = m * 4^c

    hist = CuDynamicSharedArray(Float32, 2 * nbin)
    sh_norms = CuDynamicSharedArray(Float32, m, 2 * nbin * sizeof(Float32))
    dna_sh = CuDynamicSharedArray(UInt32, W + 1, (2 * nbin + m) * sizeof(Float32))

    # stage the fragment's words (direct copy: the fragment is self-aligned)
    # and zero the histogram
    g0 = (idr - Int32(1)) * Int32(W) # the fragment's first word in the flat batch
    for wo = t:B:W
        @inbounds dna_sh[wo] = dnas[g0 + wo]
    end
    if t == 1
        @inbounds dna_sh[W+1] = UInt32(0) # guard word for funnel word pairs at
    end                                   # the fragment end (never unmasked)
    for i = t:B:(2*nbin)
        @inbounds hist[i] = 0f0
    end
    sync_threads()

    bmask = UInt32(4^s - 1) # requires s <= 16
    cbmask = UInt32(4^c - 1)
    mixer = 0x9E3779B1 # based on the golden ratio

    # thread t accumulates a consecutive chunk of FORWARD windows; window p
    # (0-based) gets phase exp(+i*theta*p) -- same convention as kernel_best_v2,
    # reached directly (the reversed-stream kernel needed the conj trick).  The
    # chunk length is made odd so that lanes of a warp, reading the staged words
    # at stride W_chunk, touch distinct banks instead of conflicting
    W_chunk = cld(nw, B)
    iszero(W_chunk & 1) && (W_chunk += 1)
    for idm = 1:m
        θ = (m == 1 ? 1f0 : Float32(2 * (idm - 1) / (m - 1) + 1)) * Float32(2π / k)
        ω_x = ComplexF32(cos(θ), sin(θ)) # exp(+i*theta)

        start_w = (t - Int32(1)) * W_chunk + Int32(1) # the thread's first window
        if start_w <= nw
            # software funnel shift over the staged words: the window's 32-bit
            # neighborhood (w0, w1) + bit offset sh; the little-endian s-mer
            # field is v & bmask
            p0 = start_w - Int32(1) # 0-based base position of the window
            wo = p0 >> 4            # staged word containing the window start
            sh = UInt32((p0 & 15) << 1)
            w0 = dna_sh[wo+1]
            w1 = dna_sh[wo+2]
            v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))

            a = θ * (start_w - Int32(1)) # phase of the thread's first window
            ω = ComplexF32(cos(a), sin(a))

            end_w = min(start_w + W_chunk - Int32(1), nw)
            for w = start_w:end_w
                smer = v & bmask # little-endian field of bases [p, p+s)
                lower = smer & cbmask
                upper = smer >> 2c
                scramble_key = (upper * mixer) & cbmask
                csmer = (lower ⊻ scramble_key) + 1

                bin = (csmer - 1) * m + idm # linear index in the (m, 4^c) layout
                CUDA.@atomic hist[2*bin-1] += real(ω)
                CUDA.@atomic hist[2*bin] += imag(ω)

                ω *= ω_x
                if w < end_w
                    sh += UInt32(2)
                    if sh == UInt32(32) # the window start crossed into the next word
                        sh = UInt32(0)
                        w0 = w1
                        wo += Int32(1)
                        w1 = dna_sh[wo+2]
                    end
                    v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))
                end
            end
        end
    end
    sync_threads()

    begin # G-homopolymer bin zeroing, the established `2*allones+1` convention
        if t == 1
            allones = UInt32(0)
            for i = 0:c-1
                allones += UInt32(4^i)
            end
            csmerG = 2 * allones + 1
            for idm = 1:m
                bin = (csmerG - 1) * m + idm
                hist[2*bin-1] = 0f0
                hist[2*bin] = 0f0
            end
        end
    end
    sync_threads()

    if normalize == 0 # one norm shared by all m copies
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                acc += abs2(ComplexF32(hist[2*bin-1], hist[2*bin]))
            end
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc # partial norms, as left by the original
        end
        sync_threads()
        if t == 1
            acc = sh_norms[1]
            for idm = 2:m
                acc += sh_norms[idm]
            end
            acc = sqrt(acc)
            acc += 1e-16
            embeds_norms[1, idr] = acc
            sh_norms[1] = acc
        end
        sync_threads()
        nrm = sh_norms[1]
        for i = t:B:(2*nbin)
            hist[i] /= nrm
        end
    elseif normalize == 1 # per-copy norms
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                acc += abs2(ComplexF32(hist[2*bin-1], hist[2*bin]))
            end
            acc = sqrt(acc)
            acc += 1f-7
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    elseif normalize == 2 # each s-mer treated equally
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                re_im = ComplexF32(hist[2*bin-1], hist[2*bin])
                re_im = ComplexF32(re_im / (abs(re_im) + eps()))
                hist[2*bin-1] = real(re_im)
                hist[2*bin] = imag(re_im)
                acc += abs2(re_im)
            end
            acc = sqrt(acc)
            acc += eps()
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    elseif normalize == 3 # use sqrt
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                re_im = ComplexF32(hist[2*bin-1], hist[2*bin])
                re_im = ComplexF32(re_im / sqrt(abs(re_im) + eps()))
                hist[2*bin-1] = real(re_im)
                hist[2*bin] = imag(re_im)
                acc += abs2(re_im)
            end
            acc = sqrt(acc)
            acc += eps()
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    end

    sync_threads()
    # the ONLY semantic change vs kernel_rope_frag: the final shared->global
    # copy splits the interleaved histogram into the [Re; Im] halves
    for bin = t:B:nbin
        @inbounds embeds[idr, bin] = T(hist[2*bin-1])         # Re half
        @inbounds embeds[idr, bin+nbin] = T(hist[2*bin])      # Im half
    end

    return
end

"""
Real-output counterpart of `encode_frag_batch!`: `dnas` is the flat fragment
batch (nfrag * W words, W = cld(k, 16)), `dest` is (nfrag, dim = 2*m*4^c) with
eltype Float32 or Float16 (the fp16 output option -- converted in-kernel),
`dest_norms` is (m, nfrag) Float32.
"""
function encode_frag_real_batch!(
    dest::CuArray{T,2},
    dest_norms::CuArray{Float32,2},
    re::RopeEncoder,
    dnas::CuVector{UInt32};
    normalize::Int = 0
) where {T<:Union{Float16,Float32}}
    @assert re.s <= 16 "kernel_rope_frag_real keeps the s-mer in a 32-bit funnel window, so it requires s <= 16"
    @assert re.c <= re.s "the compact c-mer must not exceed the s-mer (c <= s)"
    n = size(dest, 1)
    @assert size(dest, 2) == 2 * re.m * 4^re.c "dest must be (n, 2*m*4^c) -- the [Re; Im] concatenation"
    @assert size(dest_norms) == (re.m, n)
    W = cld(re.k, 16)
    @assert length(dnas) == n * W "dnas must hold n * cld(k, 16) words"

    nbin = re.m * 4^re.c
    shmem_bytes = (2 * nbin + re.m) * sizeof(Float32) + (W + 1) * sizeof(UInt32)
    @assert shmem_bytes <= 48 * 1024 "fragment too long for the shared-memory fast path (k = $(re.k) needs $(shmem_bytes) B; k <= ~163k fits at m=4, c=4)"

    n == 0 && return dest, dest_norms
    threads = 512
    # register cap as in kernel_best_v3/kernel_rope_frag: the funnel-shift
    # state would otherwise push past 65536 regs/SM and the launch would fail
    @cuda blocks = n threads = threads shmem = shmem_bytes maxregs = 128 kernel_rope_frag_real(
        dest, dest_norms, dnas, re.s, re.m, re.c, re.k, W, n, normalize
    )
    return dest, dest_norms
end

"""Allocating variant: `dnas` holds nfrag * cld(k, 16) words -> (embeds, norms)."""
function encode_frag_real_batch(::Type{T}, re::RopeEncoder, dnas::CuVector{UInt32};
                                normalize::Int = 0) where {T<:Union{Float16,Float32}}
    n = length(dnas) ÷ cld(re.k, 16)
    dest = CUDA.zeros(T, n, 2 * re.m * 4^re.c)
    dest_norms = CUDA.zeros(Float32, re.m, n)
    encode_frag_real_batch!(dest, dest_norms, re, dnas; normalize)
    return dest, dest_norms
end

# ==============================================================================
# The streamed flow
# ==============================================================================

"""
One streamed batch of REAL rope encodings.  `embeds[r, :]` is fragment r's
encoding, the FIXED split layout `[Re; Im]`: element `bin` (1 <= bin <= D,
D = m*4^c) is Re of complex bin `(bin-1) % m + 1`/s-mer `(bin-1) ÷ m + 1`, and
element `D + bin` its Im part (row r's Re block = embeds[r, 1:D], Im block =
embeds[r, D+1:2D]; real inner products = complex real parts).  `T` is Float32,
or Float16 with the stream's `fp16` option (converted in-kernel, norms stay
Float32).  `norms[t, r]` the kernel's per-copy norms (normalize = 0: row 1 is
the shared norm, rows 2..m hold partials, as always), `heads[r]` the
fragment's fasta header, `first` the 1-based global index of the batch's first
fragment in the stream.  Fresh host matrices -- retain freely.
"""
struct RopeRealBatch{T<:Union{Float16,Float32}}
    embeds::Matrix{T}          # (nb, 2*D = 2*m*4^c), [Re; Im] split layout
    norms::Matrix{Float32}     # (m, nb)
    heads::Vector{String}
    first::Int
end

# A task-level error router: same contract as ropeflow_v3's `_rope_stream_error`
# (InvalidStateException anywhere in the channel network = benign teardown,
# never a data error; real errors print immediately, land in err_out, first
# error wins, and every channel is closed so all stages unwind).  Its own copy
# so the log prefix says ropeflowreal.
function _rope_real_stream_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "ropeflowreal: $(who) failed: ",
            sprint(showerror, err, catch_backtrace()))
    flush(stderr)
    err_out[] === nothing && (err_out[] = err)
    for ch in chs
        try
            close(ch)
        catch
        end
    end
    return false
end

"""
    rope_encode_real_stream(re::RopeEncoder, file::String; k = 20_000,
                            batch_size = 2^13, normalize = 0, fp16 = false,
                            parts = nthreads(), in_cap = 2, out_cap = 2,
                            progress = false, err_out = Ref{Any}(nothing),
                            tasks_out = Ref{Vector{Task}}(Task[]))
                            -> Channel{RopeRealBatch}

Stream-rope-encode the fasta `file` through fastareads_v3's parallel fragment
reader and emit REAL vectors: fragments (exactly k bases, forward 2-bit
packed, header attached) are stacked `batch_size` at a time into pinned
staging buffers, uploaded, encoded by `kernel_rope_frag_real` on the GPU and
handed to the returned channel as
`RopeRealBatch(embeds (nb, 2*m*4^c), norms (m, nb), heads, first)` with
`nb <= batch_size` (only the last batch is partial).  `k` must equal the
encoder's baseline `re.k` (it is both the fragment length and the rope phase
basis).  `fp16 = true` makes `embeds::Matrix{Float16}` (converted IN THE
KERNEL; compute and norms stay Float32; halves the embedding D2H traffic);
`fp16 = false` (default) gives Float32.  Gathering of batch i+1 overlaps the
upload/encode/copy of batch i; `in_cap`/`out_cap` bound the in-flight stages
(backpressure: a slow consumer throttles the parsers).

The fragment order in the stream is the parsers' interleaving (arbitrary --
headers make batches self-describing).  Consume to the end (the channel closes
itself) or `close(ch)` early: the pipeline tears down quietly.  Real failures
land in `err_out[]` after the stream ends short.  Parse errors are reported in
the same slot (it is passed through to `fasta_reads`).  `tasks_out`, when
given, receives the two internal stage tasks, so a caller that stopped early
can `wait` them instead of polling/sleeping.
"""
function rope_encode_real_stream(re::RopeEncoder, file::String;
                                 k::Int = 20_000,
                                 batch_size::Int = 2^13,
                                 normalize::Int = 0,
                                 fp16::Bool = false,
                                 parts::Int = Threads.nthreads(),
                                 in_cap::Int = 2,
                                 out_cap::Int = 2,
                                 progress::Bool = false,
                                 err_out::Ref{Any} = Ref{Any}(nothing),
                                 tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k >= re.s || throw(ArgumentError("k = $k < s = $(re.s): no s-mer windows"))
    k == re.k || throw(ArgumentError("k = $k must equal the encoder baseline re.k = $(re.k)"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1"))
    normalize in (0, 1, 2, 3) || throw(ArgumentError("normalize must be 0..3"))
    T = fp16 ? Float16 : Float32 # the output eltype (converted in-kernel)
    dmn = re.m * 4^re.c
    rdim = 2 * dmn # the [Re; Im] concatenated dimension
    W = cld(k, 16) # UInt32 words per fragment (16 bases per word)

    frag_ch = fasta_reads(file; k, parts, err_out)
    free = Channel{Matrix{UInt32}}(in_cap + 1) # recycled pinned staging buffers
    gathered = Channel{Tuple{Matrix{UInt32},Int,Vector{String}}}(in_cap)
    out = Channel{RopeRealBatch}(out_cap) # RopeRealBatch{Float32} or {Float16}

    for _ in 1:(in_cap + 1)
        buf = Matrix{UInt32}(undef, W, batch_size)
        CUDA.pin(buf) # one-time page-lock, reused by every batch (fast H2D)
        put!(free, buf)
    end

    # ---- gather: fragments -> pinned (W, batch_size) word batches -----------
    t_gather = Threads.@spawn begin
        try
            buf = take!(free)
            nb = 0
            heads = String[]
            for f in frag_ch
                copyto!(buf, nb * W + 1, f.words, 1, W) # linear fill, column-major
                push!(heads, f.header)
                nb += 1
                if nb == batch_size
                    put!(gathered, (buf, nb, heads))
                    buf = take!(free)
                    nb = 0
                    heads = String[]
                end
            end
            if err_out[] === nothing # a healthy end: flush the (partial) tail
                if nb > 0
                    put!(gathered, (buf, nb, heads))
                else
                    put!(free, buf)
                end
            end # on a parse error the tail is dropped; the stream ends short
        catch err
            # benign teardown can only come from encode's death (the only
            # closer of gathered/free), and encode already closed frag_ch;
            # the extra close is belt-and-braces idempotence
            _rope_real_stream_error(err, err_out, gathered, frag_ch; who = "gather") && close(frag_ch)
        finally
            close(gathered)
        end
    end

    # ---- encode: upload -> kernel -> download -> RopeRealBatch --------------
    t_encode = Threads.@spawn begin
        prog = progress ? ProgressUnknown(desc = "Fragments encoded ", dt = 1.0) : nothing
        total = 0
        try
            for (buf, nb, heads) in gathered
                first_idx = total + 1
                dwords = CuArray{UInt32}(undef, W * nb)
                copyto!(dwords, 1, buf, 1, W * nb) # stream-ordered H2D (pinned src)
                dest = CUDA.zeros(T, nb, rdim) # kernel splits hist into [Re; Im]
                dnorms = CUDA.zeros(Float32, re.m, nb) # Float32 norms regardless of T
                encode_frag_real_batch!(dest, dnorms, re, dwords; normalize)
                embeds = Matrix{T}(undef, nb, rdim)
                norms = Matrix{Float32}(undef, re.m, nb)
                copyto!(embeds, dest) # stream-ordered D2H (fp16: half the bytes)
                copyto!(norms, dnorms)
                CUDA.synchronize() # batch done: staging buffer and device pool free
                put!(free, buf)
                total += nb
                put!(out, RopeRealBatch(embeds, norms, heads, first_idx))
                prog === nothing || update!(prog, total)
            end
        catch err
            if _rope_real_stream_error(err, err_out, out, gathered, frag_ch, free; who = "encode")
                # benign teardown (the consumer closed `out`): nobody will
                # drain the upstream stages any more, so stop them -- gather
                # may be blocked in put!(gathered)/take!(free), the parsers in
                # put!(frag_ch); all closes are idempotent
                close(gathered)
                close(free)
                close(frag_ch)
            end
        finally
            close(out) # also the early-close path (close on closed is a no-op)
            prog === nothing || finish!(prog)
        end
    end

    append!(tasks_out[], [t_gather, t_encode]) # consumers may wait these
    return out
end

# ==============================================================================
# Correctness.  The reference is ropeflow_v3's independent float64 CPU encoder
# (_ref_rope_frag, included above) wrapped with the split [Re; Im] flattening
# -- so the real path inherits an implementation that shares nothing with the
# kernel.  (The encoding is intentionally NOT bitwise the kernel_best_v2/v3
# one: the bin bijection and the scramble key differ, see ropeflow_v3.jl.)
# ==============================================================================

"""_ref_rope_frag's complex histogram -> the kernel's split [Re; Im] real vector."""
function _ref_rope_frag_real(codes::Vector{UInt8}, re::RopeEncoder; normalize::Int)
    hist, norms = _ref_rope_frag(codes, re; normalize)
    D = length(hist)
    out = Vector{Float64}(undef, 2 * D)
    @inbounds for i in 1:D
        out[i] = real(hist[i])
        out[D+i] = imag(hist[i])
    end
    return out, norms
end

# GPU batch of random fragments vs the CPU reference (values + norms +, for
# normalize = 0, the unit-energy invariant).  Modes 2/3 divide by |bin| +
# eps, so bins with heavy collision cancellation amplify the float32-vs-
# float64 epsilon difference -- they get a looser tolerance, as they only
# check convention faithfulness, not precision.  Float16 output gets its own
# (still loose) tolerances: fp16 rounding is ~5e-4 relative on top of the
# float32 accumulation, the norms stay Float32 (v3's tolerance).
function _check_frag_kernel_real(re; k = re.k, nfrag = 64, normalize::Int = 0,
                                 T::Type{<:Union{Float16,Float32}} = Float32,
                                 seed = 1234)
    Random.seed!(seed)
    W = cld(k, 16)
    words = Vector{UInt32}(undef, W * nfrag)
    rand!(words)
    if (r = k & 15) > 0 # zero-pad the tail bits, like fastareads_v3's _emit3!
        mask = (UInt32(1) << (2 * r)) - 1
        for f in 1:nfrag
            words[(f - 1) * W + W] &= mask
        end
    end
    dest, dnorms = encode_frag_real_batch(T, re, cu(words); normalize)
    embeds = Array(dest)
    norms = Array(dnorms)
    D = re.m * 4^re.c
    heavy = normalize in (2, 3)
    rtol = 1e-2
    atol = T == Float16 ? (heavy ? 2e-2 : 2e-3) : (heavy ? 1e-2 : 2e-4)
    ertol = T == Float16 ? 1e-2 : 1e-3  # unit-energy tolerance
    nrmrtol = heavy ? 1e-2 : 1e-3       # norms are Float32 regardless of T
    nrm_rows = normalize == 0 ? (1:1) : (1:re.m) # mode 0: only row 1 is the norm
    for f in 1:nfrag
        codes = _frag_codes(@view(words[(f - 1) * W + 1:f * W]), k)
        href, nrmref = _ref_rope_frag_real(codes, re; normalize)
        @assert isapprox(@view(embeds[f, :]), href; rtol, atol) "real kernel vs CPU reference mismatch (frag $f, normalize=$normalize, T=$T, s=$(re.s), m=$(re.m), c=$(re.c), k=$k)"
        for idm in nrm_rows
            @assert isapprox(norms[idm, f], nrmref[idm]; rtol = nrmrtol) "norms mismatch (frag $f, copy $idm, normalize=$normalize)"
        end
        # split layout: |x|^2 = |Re|^2 + |Im|^2 == the complex unit energy
        if normalize == 0
            e = sum(abs2, @view(embeds[f, 1:D]); init = 0.0) +
                sum(abs2, @view(embeds[f, D+1:2D]); init = 0.0)
            @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (frag $f)"
        else
            # copy idm's bins sit at stride m WITHIN each half (bin =
            # (csmer-1)*m + idm -> positions idm, idm+m, ... in 1:D and D+1:2D)
            for idm in 1:re.m
                e = sum(abs2, @view(embeds[f, idm:re.m:D]); init = 0.0) +
                    sum(abs2, @view(embeds[f, D+idm:re.m:2D]); init = 0.0)
                @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (frag $f, copy $idm)"
            end
        end
    end
    return nothing
end

# End-to-end: stream the file and compare against reference fragments
# (matched by header -- the shared channel interleaves parsers arbitrarily).
function _check_stream_real(re, path; k::Int, normalize::Int, batch_size::Int,
                            fp16::Bool, refw::Dict{String,Vector{UInt32}},
                            sample::Int, seed = 7)
    rs = MersenneTwister(seed)
    err = Ref{Any}(nothing)
    nb = seen = 0
    heads_all = String[]
    D = re.m * 4^re.c
    rdim = 2 * D
    for nt in rope_encode_real_stream(re, path; k, batch_size, normalize, fp16, err_out = err)
        nb += 1
        nn = length(nt.heads)
        @assert nt isa RopeRealBatch && eltype(nt.embeds) == (fp16 ? Float16 : Float32) "batch eltype mismatch (fp16 = $fp16)"
        @assert nt.first == seen + 1 "batch first-index mismatch"
        @assert size(nt.embeds) == (nn, rdim) "batch shape mismatch"
        @assert size(nt.norms) == (re.m, nn) "norms shape mismatch"
        # unit energy for EVERY row (normalize 0: whole row; else per copy),
        # computed over the [Re; Im] halves == the complex energy
        ertol = fp16 ? 1e-2 : 1e-3
        for r in 1:nn
            e = if normalize == 0
                sum(abs2, @view(nt.embeds[r, 1:D]); init = 0.0) +
                sum(abs2, @view(nt.embeds[r, D+1:rdim]); init = 0.0)
            else
                # stride m WITHIN each half (see _check_frag_kernel_real)
                maximum(sum(abs2, @view(nt.embeds[r, idm:re.m:D]); init = 0.0) +
                        sum(abs2, @view(nt.embeds[r, D+idm:re.m:rdim]); init = 0.0)
                        for idm in 1:re.m)
            end
            @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (row $r)"
        end
        # sampled rows: full value + norms comparison against the CPU reference
        heavy = normalize in (2, 3)
        rtol = 1e-2
        atol = fp16 ? (heavy ? 2e-2 : 2e-3) : (heavy ? 1e-2 : 2e-4)
        for r in rand(rs, 1:nn, min(sample, nn))
            w = refw[nt.heads[r]]
            href, nrmref = _ref_rope_frag_real(_frag_codes(w, k), re; normalize)
            @assert isapprox(@view(nt.embeds[r, :]), href; rtol, atol) "streamed row vs CPU reference mismatch ($(nt.heads[r]))"
            nrm_cols = normalize == 0 ? (1:1) : (1:re.m) # mode 0: row 1 only
            for idm in nrm_cols
                @assert isapprox(nt.norms[idm, r], nrmref[idm]; rtol = 1e-3) "streamed norms mismatch ($(nt.heads[r]), copy $idm)"
            end
        end
        append!(heads_all, nt.heads)
        seen += nn
    end
    @assert err[] === nothing "stream error: $(err[])"
    @assert seen == length(refw) "fragment count mismatch: $seen vs $(length(refw))"
    @assert nb == cld(seen, batch_size) "batch count mismatch ($nb for batch_size=$batch_size)"
    @assert sort(heads_all) == sort(collect(keys(refw))) "streamed heads mismatch"
    return nb
end

function run_rope_real_test()
    @show CUDA.name(device())
    @show nthreads()

    small, big = ensure_data3()
    k = 20_000
    dir = mktempdir(prefix = "ropeflowreal_")

    # ==========================================================================
    # A. KERNEL vs the float64 CPU reference: all normalize modes, fp32 AND
    #    fp16 outputs, several s/m/c configs, tail/guard edge fragment lengths.
    # ==========================================================================
    @info "A. kernel_rope_frag_real vs the CPU reference"
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    re2k = RopeEncoder(k = 2_000, s = 8, m = 4, c = 4) # short-fragment config
    for normalize in 0:3
        _check_frag_kernel_real(re2k; k = 2_000, normalize)
        @info "  config (s=8, m=4, c=4), k=2000, normalize=$normalize, fp32 OK"
    end
    for normalize in (0, 1, 2, 3) # the fp16 output path, all modes
        _check_frag_kernel_real(re2k; k = 2_000, normalize, T = Float16, seed = 11)
        @info "  config (s=8, m=4, c=4), k=2000, normalize=$normalize, fp16 OK"
    end
    for (s_, m_, c_) in ((5, 1, 4), (16, 2, 4), (8, 1, 6))
        re2 = RopeEncoder(k = 2_000, s = s_, m = m_, c = c_)
        for normalize in (0, 1)
            _check_frag_kernel_real(re2; k = 2_000, normalize, seed = 42)
        end
        @info "  config (s=$s_, m=$m_, c=$c_), k=2000, normalize 0/1 OK"
    end
    _check_frag_kernel_real(RopeEncoder(k = 2_001, s = 8, m = 4, c = 4); k = 2_001, normalize = 0, seed = 1) # 2001 % 16 > 0: tail mask
    _check_frag_kernel_real(RopeEncoder(k = 2_001, s = 8, m = 4, c = 4); k = 2_001, normalize = 1, T = Float16, seed = 4) # tail mask + fp16
    _check_frag_kernel_real(RopeEncoder(k = 17, s = 5, m = 1, c = 2); k = 17, normalize = 0, seed = 2) # nw < threads
    _check_frag_kernel_real(re; k = 20_000, nfrag = 16, normalize = 0, seed = 3) # the real shape
    _check_frag_kernel_real(RopeEncoder(k = 20_000, s = 5, m = 1, c = 5); k = 20_000, nfrag = 16,
                            normalize = 0, seed = 5) # the s5m1c5 production index sweep
    _check_frag_kernel_real(RopeEncoder(k = 20_000, s = 5, m = 1, c = 5); k = 20_000, nfrag = 16,
                            normalize = 0, T = Float16, seed = 6) # ... and its fp16 stream eltype
    @info "  edge ks (2001, 17), fp16+tail, the real k=20000, and s5m1c5 k=20000 OK"

    # ==========================================================================
    # B. FLOW on the small generated file (2^13 reads x ~20 kb, 164 MB):
    #    batch layouts, heads, fp32 AND fp16 batches, unit energy everywhere,
    #    sampled rows vs the CPU reference.  Reference fragments come from one
    #    fasta_reads drain.
    # ==========================================================================
    @info "B. rope_encode_real_stream on the small file vs reference fragments"
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == _V3_SMALL_READS "unexpected reference fragment count"
    @info "  reference: $(length(refw)) fragments (k=$k)"

    for batch_size in (3000, 2^13, 5000, 10^6) # tail, exact fit, uneven, single
        nb = _check_stream_real(re, small; k, normalize = 0, batch_size, fp16 = false, refw, sample = 8)
        @info "  normalize=0, fp32, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_stream_real(re, small; k, normalize = 1, batch_size = 3000, fp16 = false, refw, sample = 8)
    @info "  normalize=1, fp32, batch_size=3000 OK ($nb batches)"
    for normalize in (0, 1) # the fp16 output path through the whole stream
        nb = _check_stream_real(re, small; k, normalize, batch_size = 3000, fp16 = true, refw, sample = 8)
        @info "  normalize=$normalize, fp16, batch_size=3000 OK ($nb batches)"
    end
    # other encoder configs through the whole stream (sampled checks only)
    for (s_, m_, c_) in ((5, 1, 4), (16, 2, 4))
        re2 = RopeEncoder(k = k, s = s_, m = m_, c = c_)
        nb = _check_stream_real(re2, small; k, normalize = 0, batch_size = 3000, fp16 = false, refw, sample = 4)
        @info "  config (s=$s_, m=$m_, c=$c_) end-to-end OK ($nb batches)"
    end

    # ==========================================================================
    # C. teardown: closing the stream early must unwind the whole pipeline
    #    quietly (no hang, no spurious err_out) and leave it reusable.
    # ==========================================================================
    @info "C. early close"
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    ch = rope_encode_real_stream(re, small; k, batch_size = 3000, err_out = err, tasks_out = tks)
    got = 0
    for _ in ch
        got += 1
        got == 2 && break
    end
    close(ch)
    foreach(wait, tks[]) # deterministic: the whole pipeline has unwound
    @assert err[] === nothing "early close raised: $(err[])"
    @assert got == 2
    @assert all(istaskdone, tks[])
    nb = _check_stream_real(re, small; k, normalize = 0, batch_size = 3000, fp16 = false, refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    # ==========================================================================
    # D. empty file -> zero batches, no error.
    # ==========================================================================
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    err = Ref{Any}(nothing)
    n = 0
    for _ in rope_encode_real_stream(re, empty_fasta; k, err_out = err)
        n += 1
    end
    @assert n == 0 && err[] === nothing
    @info "D. empty file OK (0 batches)"

    @info "ALL ROPEFLOWREAL CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks: parse-only vs full pipeline (no-op / collecting consumer, fp32
# and fp16 outputs) on the big file, plus the kernel alone; run under
# `julia -t N`.  The fp16 lines exist to quantify the halved D2H (the
# ropeflow_v3 bottleneck) against the otherwise-identical fp32 pipeline.
# ==============================================================================
function bench_rope_real(; reps = _reps())
    small, big = ensure_data3()
    _warm_cache(big)
    k = 20_000
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4)
    dmn = re.m * 4^re.c
    rdim = 2 * dmn # dim of the [Re; Im] real vectors
    @info "ropeflowreal benchmarks on the big file ($(filesize(big)) bytes, $_V3_BIG_READS reads)" julia_threads = nthreads() reps

    # ---- parse only ----------------------------------------------------------
    _timed_min("fasta_reads drain (parse only)"; bytes = filesize(big), reps) do
        n = 0
        for _ in fasta_reads(big; k)
            n += 1
        end
        n
    end

    # ---- full pipeline, no-op consumers --------------------------------------
    for fp16 in (false, true)
        _timed_min("real stream, no-op consumer, $(fp16 ? "fp16" : "fp32") (H2D+kernel+D2H)"; bytes = filesize(big), reps) do
            n = 0
            for nt in rope_encode_real_stream(re, big; k, normalize = 0, fp16)
                n += length(nt.heads)
            end
            n
        end
    end

    # ---- full pipeline, collecting consumers (pinned host arrays) ------------
    N = Array{Float32}(undef, re.m, _V3_BIG_READS)
    CUDA.pin(N)
    for fp16 in (false, true)
        dev_max = Ref(0)
        E = Array{fp16 ? Float16 : Float32}(undef, _V3_BIG_READS, rdim) # 1.07 / 0.54 GiB
        CUDA.pin(E)
        _timed_min("real stream, collecting sink $(fp16 ? "fp16" : "fp32") (pinned)"; bytes = filesize(big), reps) do
            nrow = 0
            for nt in rope_encode_real_stream(re, big; k, normalize = 0, fp16)
                nn = length(nt.heads)
                dev_max[] = max(dev_max[], CUDA.used_memory())
                copyto!(E, nrow * rdim + 1, nt.embeds, 1, nn * rdim)
                copyto!(N, nrow * re.m + 1, nt.norms, 1, nn * re.m)
                nrow += nn
            end
            nrow
        end
        @printf("  peak device-pool residency during the %s stream: %.2f GiB\n",
                fp16 ? "fp16" : "fp32", dev_max[] / 2^30)
        E = nothing # release the pinned sink before allocating the next one
        GC.gc()
    end

    # ---- the kernel alone ----------------------------------------------------
    nker = 2^15
    words = Vector{UInt32}(undef, nker * cld(k, 16))
    woff = 0
    for f in fasta_reads(big; k) # one parse pass to get real fragments
        copyto!(words, woff + 1, f.words, 1, length(f.words))
        woff += length(f.words)
        woff >= length(words) && break
    end
    @assert woff == length(words)
    dwords = cu(words)
    dnorms = CUDA.zeros(Float32, re.m, nker)
    for fp16 in (false, true)
        dest = CUDA.zeros(fp16 ? Float16 : Float32, nker, rdim)
        _timed_min("kernel_rope_frag_real only ($nker x 20 kb, fwd, $(fp16 ? "fp16" : "fp32"))"; bytes = nker * k, reps) do
            encode_frag_real_batch!(dest, dnorms, re, dwords; normalize = 0)
            CUDA.synchronize()
            nker
        end
    end

    println("  (GB/s = FASTA bytes consumed per second; the stream holds ~in_cap")
    println("   pinned word batches + out_cap embedding batches in flight, and one")
    println("   batch on the device; fp32 output is the same D2H byte count as the")
    println("   complex ropeflow_v3, fp16 halves it)")
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data3()
    mode == "test" && run_rope_real_test()
    mode == "bench" && bench_rope_real()
    mode == "all" && (ensure_data3(); run_rope_real_test(); bench_rope_real())
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
