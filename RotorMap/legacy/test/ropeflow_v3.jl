# ==============================================================================
# ropeflow_v3.jl -- the rope-encoding stage of the fastareads_v3 pipeline:
# fragments from `fasta_reads` (fastareads_v3.jl) are rope-encoded on the GPU
# and the resulting complex vectors are stacked into batches of
# `batch_size` (default 2^13) x dim = m*4^c rows, streamed to the consumer
# through a Channel{RopeBatch}.
#
# WHY A NEW KERNEL (the fastareads_v3 <-> newflow_v2 format mismatch):
#   newflow_v2 feeds kernel_best_v3 (newencoder2bit.jl), whose input is ONE
#   flat 2-bit bitstream storing every read's bases in REVERSE order, reads
#   addressed by global Int32 starts/stops and funnel-shifted at staging
#   because a read may start at any bit offset.  fastareads_v3 emits
#   self-contained fragments instead: exactly k bases, PLAIN FORWARD packing
#   (base j -> word (j-1)>>4 + 1, bit 2*((j-1)&15), cld(k,16) UInt32 words,
#   zero-padded tail), no global offsets, a header String instead of a
#   position.  `kernel_rope_frag` below is the forward-fragment counterpart:
#     * one block per fragment, 512 threads, dynamic shared memory
#       [histogram | norm workspace | staged words] exactly like kernel_best_v3,
#       but staging is a DIRECT copy (each fragment is word-aligned to its own
#       row of the batch, no sub-word alignment, one zero guard word);
#     * windows are walked FORWARD (0-based window p covers bases [p, p+s));
#       the s-mer is the little-endian 2s-bit field `v & bmask` -- i.e. the
#       low 2c bits are the window's LAST c bases reversed and the high part
#       is the FIRST s-c bases reversed.  Relative to kernel_best_v2/v3 the
#       histogram bins are therefore a fixed per-window bijection and the
#       scramble key is computed from a reversed upper field: the encoding is
#       not bitwise the v2 embedding (confirmed as acceptable), but it is the
#       same rope construction -- per-window phasor exp(+i*theta_idm*p) with
#       theta_idm = factor*2pi/k (factor = 1 for m=1, else 2(idm-1)/(m-1)+1,
#       the same convention and float32 trig as kernel_best_v2), scrambled
#       histogram accumulation, G-bin zeroing (the established
#       `2*allones+1` convention) and normalize modes 0-3 verbatim;
#     * the phases run forward, so (unlike the reversed-stream kernel) there
#       is no conjugation bookkeeping -- `omega` starts at cis(theta*p0) and
#       steps by omega_x = cis(theta).
#
# THE STREAM (rope_encode_stream): three stages connected by channels, all
# backpressured, one batch of words (host, pinned) + one batch of embeddings
# (device) alive per in-flight slot:
#
#     fasta_reads(file)               parts parallel parsers (its own tasks)
#       | Channel{FastaFragment}      (header, len == k, words)
#     v
#     gather task                     fills a pinned (cld(k,16), batch_size)
#       | Channel{(buf, nb, heads)}   UInt32 buffer + the batch's headers
#     v
#     encode task                     H2D -> kernel_rope_frag -> D2H, one
#       | Channel{RopeBatch}          CUDA.synchronize per batch, buffer ring
#     v                               (in_cap+1 pinned word buffers recycled
#     consumer                        via a free-list channel)
#
# RopeBatch carries FRESH host matrices (embeds (nb, dim) ComplexF32 -- row r
# is fragment r's encoding; norms (m, nb); heads; first = 1-based global index
# of the batch's first fragment), so the consumer may retain batches freely.
# Stopping: consume to the end (the channel closes itself) or `close(ch)`
# early -- the whole pipeline tears down quietly and deterministically (the
# caller can `wait` the two stage tasks via `tasks_out`); genuine
# errors print immediately and are stored in `err_out[]` (also the
# fastareads_v3 parser's error slot).  fwd only: fasta_reads yields forward
# fragments, no reverse-complement stream is produced (fastareads_v3 spec).
#
# Run modes (ARGV[1]):
#   gen   generate the test files if missing (fastareads_v3's ensure_data3)
#   test  correctness: float64 CPU reference of the exact kernel semantics
#         (all normalize modes, several s/m/c configs, tail/guard edge ks),
#         end-to-end flow vs the CPU reference on the 164 MB file (batch
#         layouts, heads, sampled rows, unit energy), early-close teardown,
#         empty file
#   bench parse-only vs full pipeline vs collecting sink on the 2.6 GB file,
#         plus the kernel alone
#   all   everything above (default)
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16, min of 3, big file =
# 2.6 GB, k = 20,000, s = 8, m = 4, c = 4 -> dim = 1024, batch_size = 2^13):
#
#   fasta_reads drain (parse only)                  0.248 s   10.60 GB/s
#   rope stream, no-op consumer (H2D+kernel+D2H)    0.590 s    4.45 GB/s
#   rope stream, collecting sink (pinned)           0.506 s    5.18 GB/s
#   kernel_rope_frag only (2^15 x 20 kb)            0.021 s   31.04 GB/s
#
# The rope stage roughly halves the parse-only throughput: the kernel itself is
# fast (131k fragments ~= 84 ms of pure GPU, peak device residency ~0.5 GiB),
# the difference is the unavoidable output traffic -- every fragment's 8 KiB
# embedding must reach the host (1.07 GiB per big-file pass, pageable D2H
# ~6 GB/s) plus the gather/H2D/channel funnel (gc ~5%).  Correctness: kernel vs
# a float64 CPU reference of the exact semantics (all normalize modes, 5
# configs, tail/guard edge ks), end-to-end stream vs reference fragments (4
# batch layouts, normalize 0/1, 3 encoder configs, unit energy on every row),
# deterministic early-close teardown (stage tasks joined via tasks_out), empty
# file -- ALL PASS.
# ==============================================================================

include(joinpath(@__DIR__, "fastareads_v3.jl")) # FastaFragment, fasta_reads,
# ensure_data3, _timed_min/_warm_cache/_reps (+ transitively newfasta_v2.jl ->
# newflow.jl -> newfasta.jl -> newencoder2bit.jl: pack_refs-style v2 kernels
# and loaders for context, and the v2 test harness)

using Random
using CUDA
using Printf
using ProgressMeter
using RotorMap
using RotorMap.RopeEncoders
using Base.Threads

# ==============================================================================
# The forward-fragment kernel
# ==============================================================================

"""
One CUDA block per fragment, forward-fragment counterpart of `kernel_best_v3`
(newencoder2bit.jl).  `dnas` is the flat batch: fragment `idr` occupies the
`W` consecutive words `dnas[(idr-1)*W .+ (1:W)]` (W = cld(k, 16)); base j of
the fragment sits at bit 2*(j & 15) of word (j >> 4) + 1 (fastareads_v3 plain
forward packing).  Each thread accumulates a consecutive FORWARD chunk of
s-mer windows into the shared histogram (little-endian s-mer field: low 2c
bits = the window's last c bases reversed; high part = the first s-c bases
reversed), window p gets phasor exp(+i*theta_idm*p) computed with the same
float32 trig as kernel_best_v2, the G-bin is zeroed with the established
`2*allones+1` convention and normalize modes 0-3 replicate the v2/v3 kernels
verbatim.  Output: `embeds[idr, bin]` (an (n, m*4^c) row-major-batch matrix,
coalesced per block) and `embeds_norms[t, idr]` (m x n, same semantics as
always, incl. the normalize==0 partial-rows quirk).
"""
function kernel_rope_frag(embeds, embeds_norms, dnas, s, m, c, k, W, n, normalize)
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
            # field is v & bmask (NB: a fixed bijection of the v2 bins, and a
            # different scramble key -- acceptable per the ropeflow_v3 spec)
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
    for bin = t:B:nbin
        @inbounds embeds[idr, bin] = ComplexF32(hist[2*bin-1], hist[2*bin])
    end

    return
end

"""
Drop-in forward-fragment counterpart of `encode_batch_cuda_all_best_v3!`:
`dnas` is the flat fragment batch (nfrag * W words, W = cld(k, 16)),
`dest` is (nfrag, dim = m*4^c), `dest_norms` is (m, nfrag).
"""
function encode_frag_batch!(
    dest::CuArray{ComplexF32},
    dest_norms::CuArray{Float32},
    re::RopeEncoder,
    dnas::CuVector{UInt32};
    normalize::Int = 0
)
    @assert re.s <= 16 "kernel_rope_frag keeps the s-mer in a 32-bit funnel window, so it requires s <= 16"
    @assert re.c <= re.s "the compact c-mer must not exceed the s-mer (c <= s)"
    n = size(dest, 1)
    @assert size(dest, 2) == re.m * 4^re.c && size(dest_norms) == (re.m, n)
    W = cld(re.k, 16)
    @assert length(dnas) == n * W "dnas must hold n * cld(k, 16) words"

    nbin = re.m * 4^re.c
    shmem_bytes = (2 * nbin + re.m) * sizeof(Float32) + (W + 1) * sizeof(UInt32)
    @assert shmem_bytes <= 48 * 1024 "fragment too long for the shared-memory fast path (k = $(re.k) needs $(shmem_bytes) B; k <= ~163k fits at m=4, c=4)"

    n == 0 && return dest, dest_norms
    threads = 512
    # register cap as in kernel_best_v3: the funnel-shift state would otherwise
    # push past 65536 regs/SM and the launch would fail
    @cuda blocks = n threads = threads shmem = shmem_bytes maxregs = 128 kernel_rope_frag(
        dest, dest_norms, dnas, re.s, re.m, re.c, re.k, W, n, normalize
    )
    return dest, dest_norms
end

"""Allocating variant: `dnas` holds nfrag * cld(k, 16) words -> (embeds, norms)."""
function encode_frag_batch(re::RopeEncoder, dnas::CuVector{UInt32}; normalize::Int = 0)
    n = length(dnas) ÷ cld(re.k, 16)
    dest = CUDA.zeros(ComplexF32, n, re.m * 4^re.c)
    dest_norms = CUDA.zeros(Float32, re.m, n)
    encode_frag_batch!(dest, dest_norms, re, dnas; normalize)
    return dest, dest_norms
end

# ==============================================================================
# The streamed flow
# ==============================================================================

"""
One streamed batch of rope encodings.  `embeds[r, :]` is fragment r's encoding
(a fresh host matrix -- retain freely), `norms[t, r]` the kernel's per-copy
norms (normalize = 0: row 1 is the shared norm, rows 2..m hold partials, as
always), `heads[r]` the fragment's fasta header, `first` the 1-based global
index of the batch's first fragment in the stream.
"""
struct RopeBatch
    embeds::Matrix{ComplexF32} # (nb, dim = m*4^c)
    norms::Matrix{Float32}     # (m, nb)
    heads::Vector{String}
    first::Int
end

# A task-level error router: InvalidStateException anywhere in the channel
# network means teardown (some stage closed a channel), never a data error --
# the parsing logic throws BoundsError & friends.  Returns true for benign
# shutdown; real errors print immediately (async failures are otherwise easy
# to miss), go into err_out (first error wins) and close every channel so all
# stages unwind.
function _rope_stream_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "ropeflow_v3: $(who) failed: ",
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
    rope_encode_stream(re::RopeEncoder, file::String; k = 20_000,
                       batch_size = 2^13, normalize = 0, parts = nthreads(),
                       in_cap = 2, out_cap = 2, progress = false,
                       err_out = Ref{Any}(nothing)) -> Channel{RopeBatch}

Stream-rope-encode the fasta `file` through fastareads_v3's parallel fragment
reader: fragments (exactly k bases, forward 2-bit packed, header attached) are
stacked `batch_size` at a time into pinned staging buffers, uploaded, encoded
by `kernel_rope_frag` on the GPU and handed to the returned channel as
`RopeBatch(embeds (nb, dim), norms (m, nb), heads, first)` with `nb <=
batch_size` (only the last batch is partial).  `k` must equal the encoder's
baseline `re.k` (it is both the fragment length and the rope phase basis).  Gathering of batch i+1 overlaps
the upload/encode/copy of batch i; `in_cap`/`out_cap` bound the in-flight
stages (backpressure: a slow consumer throttles the parsers).

The fragment order in the stream is the parsers' interleaving (arbitrary --
headers make batches self-describing).  Consume to the end (the channel closes
itself) or `close(ch)` early: the pipeline tears down quietly.  Real failures
land in `err_out[]` after the stream ends short.  Parse errors are reported in
the same slot (it is passed through to `fasta_reads`).  `tasks_out`, when
 given, receives the two internal stage tasks, so a caller that stopped early
can `wait` them instead of polling/sleeping.
"""
function rope_encode_stream(re::RopeEncoder, file::String;
                            k::Int = 20_000,
                            batch_size::Int = 2^13,
                            normalize::Int = 0,
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
    dmn = re.m * 4^re.c
    W = cld(k, 16) # UInt32 words per fragment (16 bases per word)

    frag_ch = fasta_reads(file; k, parts, err_out)
    free = Channel{Matrix{UInt32}}(in_cap + 1) # recycled pinned staging buffers
    gathered = Channel{Tuple{Matrix{UInt32},Int,Vector{String}}}(in_cap)
    out = Channel{RopeBatch}(out_cap)

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
            _rope_stream_error(err, err_out, gathered, frag_ch; who = "gather") && close(frag_ch)
        finally
            close(gathered)
        end
    end

    # ---- encode: upload -> kernel -> download -> RopeBatch ------------------
    t_encode = Threads.@spawn begin
        prog = progress ? ProgressUnknown(desc = "Fragments encoded ", dt = 1.0) : nothing
        total = 0
        try
            for (buf, nb, heads) in gathered
                first_idx = total + 1
                dwords = CuArray{UInt32}(undef, W * nb)
                copyto!(dwords, 1, buf, 1, W * nb) # stream-ordered H2D (pinned src)
                dest = CUDA.zeros(ComplexF32, nb, dmn) # kernel ORs into zeroed bins
                dnorms = CUDA.zeros(Float32, re.m, nb)
                encode_frag_batch!(dest, dnorms, re, dwords; normalize)
                embeds = Matrix{ComplexF32}(undef, nb, dmn)
                norms = Matrix{Float32}(undef, re.m, nb)
                copyto!(embeds, dest) # stream-ordered D2H
                copyto!(norms, dnorms)
                CUDA.synchronize() # batch done: staging buffer and device pool free
                put!(free, buf)
                total += nb
                put!(out, RopeBatch(embeds, norms, heads, first_idx))
                prog === nothing || update!(prog, total)
            end
        catch err
            if _rope_stream_error(err, err_out, out, gathered, frag_ch, free; who = "encode")
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
# Correctness.  The reference is a deliberately naive float64 CPU encoder with
# the exact semantics of kernel_rope_frag (little-endian s-mer fields, same
# scramble key and mixer, same G-bin convention, exp(+i*theta*p) phases,
# normalize 0-3 with the kernels' epsilons) -- an independent implementation,
# so shared bugs cannot hide.  (The encoding is intentionally NOT bitwise the
# kernel_best_v2/v3 one: the bin bijection and the scramble key differ, see
# the file header.)
# ==============================================================================

"""fastareads_v3 forward packing -> base codes 0:3 (the obvious decode)."""
function _frag_codes(words::AbstractVector{UInt32}, k::Int)
    codes = Vector{UInt8}(undef, k)
    for j in 1:k
        codes[j] = (words[(j - 1) >> 4 + 1] >> (2 * ((j - 1) & 15))) & UInt32(3)
    end
    return codes
end

function _ref_rope_frag(codes::Vector{UInt8}, re::RopeEncoder; normalize::Int)
    (; k, s, m, c) = re
    nbin = m * 4^c
    hist = zeros(ComplexF64, nbin)
    cbmask = UInt32(4^c - 1)
    nw = k - s + 1
    for p in 0:nw-1
        smer = UInt32(0)
        @inbounds for j in 0:s-1
            smer |= UInt32(codes[p+j+1]) << (2 * j) # little-endian field
        end
        lower = smer & cbmask
        upper = smer >> (2 * c)
        csmer = Int(lower ⊻ ((upper * 0x9E3779B1) & cbmask)) + 1
        for idm in 1:m
            θ = (m == 1 ? 1.0 : 2 * (idm - 1) / (m - 1) + 1) * (2π / k)
            hist[(csmer - 1) * m + idm] += cis(θ * p) # exp(+i*theta*p)
        end
    end
    allones = sum(UInt32(4)^i for i in 0:c-1)
    csmerG = 2 * allones + 1
    for idm in 1:m
        hist[(csmerG - 1) * m + idm] = 0.0
    end

    norms = zeros(m)
    if normalize == 0
        nrm = sqrt(sum(abs2, hist)) + 1e-16
        hist ./= nrm
        norms[1] = nrm
    elseif normalize == 1
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + 1e-7
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    elseif normalize == 2
        for bin in 1:nbin
            hist[bin] = hist[bin] / (abs(hist[bin]) + eps())
        end
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + eps()
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    else # 3
        for bin in 1:nbin
            hist[bin] = hist[bin] / sqrt(abs(hist[bin]) + eps())
        end
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + eps()
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    end
    return hist, norms
end

# GPU batch of random fragments vs the CPU reference (values + norms +, for
# normalize = 0, the unit-energy invariant).  Modes 2/3 divide by |bin| +
# eps, so bins with heavy collision cancellation amplify the float32-vs-
# float64 epsilon difference -- they get a looser tolerance, as they only
# check convention faithfulness, not precision.
function _check_frag_kernel(re; k = re.k, nfrag = 64, normalize::Int = 0, seed = 1234)
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
    dest, dnorms = encode_frag_batch(re, cu(words); normalize)
    embeds = Array(dest)
    norms = Array(dnorms)
    rtol = normalize in (2, 3) ? 1e-2 : 1e-3
    atol = normalize in (2, 3) ? 1e-2 : 2e-4
    nrm_rows = normalize == 0 ? (1:1) : (1:re.m) # mode 0: only row 1 is the norm
    for f in 1:nfrag
        codes = _frag_codes(@view(words[(f - 1) * W + 1:f * W]), k)
        href, nrmref = _ref_rope_frag(codes, re; normalize)
        @assert isapprox(@view(embeds[f, :]), href; rtol, atol) "kernel vs CPU reference mismatch (frag $f, normalize=$normalize, s=$(re.s), m=$(re.m), c=$(re.c), k=$k)"
        for idm in nrm_rows
            @assert isapprox(norms[idm, f], nrmref[idm]; rtol) "norms mismatch (frag $f, copy $idm, normalize=$normalize)"
        end
        if normalize == 0
            @assert isapprox(sum(abs2, @view(embeds[f, :])), 1.0; rtol = 1e-3) "unit energy violated (frag $f)"
        else
            for idm in 1:re.m
                @assert isapprox(sum(abs2, @view(embeds[f, idm:re.m:end])), 1.0; rtol = 1e-3) "unit energy violated (frag $f, copy $idm)"
            end
        end
    end
    return nothing
end

# End-to-end: stream the file and compare against reference fragments
# (matched by header -- the shared channel interleaves parsers arbitrarily).
function _check_stream(re, path; k::Int, normalize::Int, batch_size::Int,
                       refw::Dict{String,Vector{UInt32}}, sample::Int, seed = 7)
    rs = MersenneTwister(seed)
    err = Ref{Any}(nothing)
    nb = seen = 0
    heads_all = String[]
    dim = re.m * 4^re.c
    for nt in rope_encode_stream(re, path; k, batch_size, normalize, err_out = err)
        nb += 1
        nn = length(nt.heads)
        @assert nt.first == seen + 1 "batch first-index mismatch"
        @assert size(nt.embeds) == (nn, dim) "batch shape mismatch"
        @assert size(nt.norms) == (re.m, nn) "norms shape mismatch"
        # unit energy for EVERY row (normalize 0: whole row; else per copy)
        for r in 1:nn
            e = normalize == 0 ? sum(abs2, @view(nt.embeds[r, :])) :
                maximum(sum(abs2, @view(nt.embeds[r, idm:re.m:end])) for idm in 1:re.m)
            @assert isapprox(e, 1.0; rtol = 1e-3) "unit energy violated (row $r)"
        end
        # sampled rows: full value + norms comparison against the CPU reference
        for r in rand(rs, 1:nn, min(sample, nn))
            w = refw[nt.heads[r]]
            href, nrmref = _ref_rope_frag(_frag_codes(w, k), re; normalize)
            rtol = normalize in (2, 3) ? 1e-2 : 1e-3
            @assert isapprox(@view(nt.embeds[r, :]), href; rtol, atol = 2e-4) "streamed row vs CPU reference mismatch ($(nt.heads[r]))"
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

function run_rope_test()
    @show CUDA.name(device())
    @show nthreads()

    small, big = ensure_data3()
    k = 20_000
    dir = mktempdir(prefix = "ropeflow_v3_")

    # ==========================================================================
    # A. KERNEL vs the float64 CPU reference: all normalize modes, several
    #    s/m/c configs, tail/guard edge fragment lengths.
    # ==========================================================================
    @info "A. kernel_rope_frag vs the CPU reference"
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    re2k = RopeEncoder(k = 2_000, s = 8, m = 4, c = 4) # short-fragment config
    for normalize in 0:3
        _check_frag_kernel(re2k; k = 2_000, normalize)
        @info "  config (s=8, m=4, c=4), k=2000, normalize=$normalize OK"
    end
    for (s_, m_, c_) in ((5, 1, 4), (16, 2, 4), (8, 1, 6))
        re2 = RopeEncoder(k = 2_000, s = s_, m = m_, c = c_)
        for normalize in (0, 1)
            _check_frag_kernel(re2; k = 2_000, normalize, seed = 42)
        end
        @info "  config (s=$s_, m=$m_, c=$c_), k=2000, normalize 0/1 OK"
    end
    _check_frag_kernel(RopeEncoder(k = 2_001, s = 8, m = 4, c = 4); k = 2_001, normalize = 0, seed = 1) # 2001 % 16 > 0: tail mask
    _check_frag_kernel(RopeEncoder(k = 17, s = 5, m = 1, c = 2); k = 17, normalize = 0, seed = 2) # nw < threads
    _check_frag_kernel(re; k = 20_000, nfrag = 16, normalize = 0, seed = 3) # the real shape
    @info "  edge ks (2001, 17) and the real k=20000 OK"

    # ==========================================================================
    # B. FLOW on the small generated file (2^13 reads x ~20 kb, 164 MB):
    #    batch layouts, heads, unit energy everywhere, sampled rows vs the CPU
    #    reference.  Reference fragments come from one fasta_reads drain.
    # ==========================================================================
    @info "B. rope_encode_stream on the small file vs reference fragments"
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == _V3_SMALL_READS "unexpected reference fragment count"
    @info "  reference: $(length(refw)) fragments (k=$k)"

    for batch_size in (3000, 2^13, 5000, 10^6) # tail, exact fit, uneven, single
        nb = _check_stream(re, small; k, normalize = 0, batch_size, refw, sample = 8)
        @info "  normalize=0, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_stream(re, small; k, normalize = 1, batch_size = 3000, refw, sample = 8)
    @info "  normalize=1, batch_size=3000 OK ($nb batches)"
    # other encoder configs through the whole stream (sampled checks only)
    for (s_, m_, c_) in ((5, 1, 4), (16, 2, 4))
        re2 = RopeEncoder(k = k, s = s_, m = m_, c = c_)
        nb = _check_stream(re2, small; k, normalize = 0, batch_size = 3000, refw, sample = 4)
        @info "  config (s=$s_, m=$m_, c=$c_) end-to-end OK ($nb batches)"
    end

    # ==========================================================================
    # C. teardown: closing the stream early must unwind the whole pipeline
    #    quietly (no hang, no spurious err_out) and leave it reusable.
    # ==========================================================================
    @info "C. early close"
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    ch = rope_encode_stream(re, small; k, batch_size = 3000, err_out = err, tasks_out = tks)
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
    nb = _check_stream(re, small; k, normalize = 0, batch_size = 3000, refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    # ==========================================================================
    # D. empty file -> zero batches, no error.
    # ==========================================================================
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    err = Ref{Any}(nothing)
    n = 0
    for _ in rope_encode_stream(re, empty_fasta; k, err_out = err)
        n += 1
    end
    @assert n == 0 && err[] === nothing
    @info "D. empty file OK (0 batches)"

    @info "ALL ROPEFLOW_V3 CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks: parse-only vs full pipeline (no-op / collecting consumer) on the
# big file, plus the kernel alone; run under `julia -t N`.
# ==============================================================================
function bench_rope(; reps = _reps())
    small, big = ensure_data3()
    _warm_cache(big)
    k = 20_000
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4)
    dmn = re.m * 4^re.c
    @info "ropeflow_v3 benchmarks on the big file ($(filesize(big)) bytes, $_V3_BIG_READS reads)" julia_threads = nthreads() reps

    # ---- parse only ----------------------------------------------------------
    _timed_min("fasta_reads drain (parse only)"; bytes = filesize(big), reps) do
        n = 0
        for _ in fasta_reads(big; k)
            n += 1
        end
        n
    end

    # ---- full pipeline, no-op consumer ---------------------------------------
    _timed_min("rope stream, no-op consumer (H2D+kernel+D2H)"; bytes = filesize(big), reps) do
        n = 0
        for nt in rope_encode_stream(re, big; k, normalize = 0)
            n += length(nt.heads)
        end
        n
    end

    # ---- full pipeline, collecting consumer (pinned host arrays) -------------
    dev_max = Ref(0)
    E = Array{ComplexF32}(undef, _V3_BIG_READS, dmn) # ~1.07 GiB
    N = Array{Float32}(undef, re.m, _V3_BIG_READS)
    CUDA.pin(E)
    _timed_min("rope stream, collecting sink (pinned)"; bytes = filesize(big), reps) do
        nrow = 0
        for nt in rope_encode_stream(re, big; k, normalize = 0)
            nn = length(nt.heads)
            dev_max[] = max(dev_max[], CUDA.used_memory())
            copyto!(E, nrow * dmn + 1, nt.embeds, 1, nn * dmn)
            copyto!(N, nrow * re.m + 1, nt.norms, 1, nn * re.m)
            nrow += nn
        end
        nrow
    end
    @printf("  peak device-pool residency during the stream: %.2f GiB\n", dev_max[] / 2^30)

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
    dest = CUDA.zeros(ComplexF32, nker, dmn)
    dnorms = CUDA.zeros(Float32, re.m, nker)
    _timed_min("kernel_rope_frag only ($nker x 20 kb, fwd)"; bytes = nker * k, reps) do
        encode_frag_batch!(dest, dnorms, re, dwords; normalize = 0)
        CUDA.synchronize()
        nker
    end

    println("  (GB/s = FASTA bytes consumed per second; the stream holds ~in_cap")
    println("   pinned word batches + out_cap embedding batches in flight, and one")
    println("   batch on the device)")
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data3()
    mode == "test" && run_rope_test()
    mode == "bench" && bench_rope()
    mode == "all" && (ensure_data3(); run_rope_test(); bench_rope())
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
