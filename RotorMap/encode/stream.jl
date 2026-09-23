# ==============================================================================
# stream.jl -- the streaming rope-encode pipeline (reads -> GPU encode ->
# channel of batches), REAL output.
#
# PURPOSE: stream-rope-encode a fasta file into a Channel{RopeRealBatch}:
#   fragments (exactly k bases, forward 2-bit packed, header attached) are
#   stacked `batch_size` at a time into pinned staging buffers, uploaded,
#   encoded by kernel_rope_frag_real (kernel.jl) on the GPU and handed to the
#   consumer.  Three stages connected by channels, all backpressured, one
#   batch of words (host, pinned) + one batch of embeddings (device) alive per
#   in-flight slot:
#
#     fasta_reads(file)               parts parallel parsers (its own tasks)
#       | Channel{FastaFragment}      (header, len == k, words)
#     v
#     gather task                     fills a pinned (cld(k,16), batch_size)
#       | Channel{(buf, nb, heads)}   UInt32 buffer + the batch's headers
#     v
#     encode task                     H2D -> kernel_rope_frag_real -> D2H, one
#       | Channel{RopeRealBatch}      CUDA.synchronize per batch, buffer ring
#     v                               (in_cap+1 pinned word buffers recycled
#     consumer                        via a free-list channel)
#
#   RopeRealBatch carries FRESH host matrices (embeds (nb, 2D) in the split
#   [Re; Im] layout, D = m*4^c -- row r is fragment r's encoding; norms (m,
#   nb) Float32; heads; first = 1-based global index of the batch's first
#   fragment), so the consumer may retain batches freely.  Stopping: consume
#   to the end (the channel closes itself) or `close(ch)` early -- the whole
#   pipeline tears down quietly and deterministically (the caller can `wait`
#   the two stage tasks via `tasks_out`); genuine errors print immediately and
#   are stored in `err_out[]` (also the parser's error slot).  fwd only:
#   fasta_reads yields forward fragments, no reverse-complement stream.
#
# SOURCES: legacy/test/ropeflowreal.jl (RopeRealBatch, _rope_real_stream_error,
#   rope_encode_real_stream, _check_stream_real).
#
# DEPS: includes kernel.jl (kernel_rope_frag_real / encode_frag_real_batch!,
#   and transitively ropeencoder.jl + reference.jl for the self-check).  A
#   consumer must include kernel.jl first (or simply include this file, which
#   does it).  `using CUDA, Random, Base.Threads, ProgressMeter` required.
#   EXTERNAL DEPS: the fasta layer (fasta/reader.jl) must be included first --
#   `rope_encode_real_stream` consumes `fasta_reads(file; k, parts, err_out)`
#   and its `FastaFragment` values (fields `.header`, `.words`) by bare name;
#   nothing is rewritten here (the names already match).
#
# NOTES / DROPPED:
#   * the complex-output stream of the old ropeflow_v3 experiment
#     (kernel_rope_frag, encode_frag_batch!, encode_frag_batch, RopeBatch,
#     _rope_stream_error, rope_encode_stream) -- superseded by this real flow;
#   * the experiment mains / benchmark harnesses run_rope_real_test and
#     bench_rope_real (and the fasta/harness helpers they used:
#     ensure_data3, _timed_min/_warm_cache/_reps, _V3_SMALL_READS/
#     _V3_BIG_READS);
#   * `_check_stream_real` KEPT (judgment call): it is small, pure
#     verification of this stream against the CPU golden reference, and the
#     caller supplies the reference fragments (`refw`), so its body has no
#     fasta-layer dependency.
# ==============================================================================

include(joinpath(@__DIR__, "kernel.jl"))

using CUDA
using Random
using Base.Threads
using ProgressMeter

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

# A task-level error router: same contract as the (superseded, not carried)
# complex-flow `_rope_stream_error` -- InvalidStateException anywhere in the
# channel network = benign teardown, never a data error; real errors print
# immediately, land in err_out, first error wins, and every channel is closed
# so all stages unwind.  Its own copy so the log prefix names this stream.
function _rope_real_stream_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "rope_encode_real_stream: $(who) failed: ",
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

Stream-rope-encode the fasta `file` through the fasta layer's parallel
fragment reader (`fasta_reads` -- EXTERNAL DEP, fasta/reader.jl) and emit REAL
vectors: fragments (exactly k bases, forward 2-bit packed, header attached)
are stacked `batch_size` at a time into pinned staging buffers, uploaded,
encoded by `kernel_rope_frag_real` on the GPU and handed to the returned
channel as
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
# Stream self-check against the CPU golden reference (reference.jl).  The
# caller supplies `refw` (header -> packed words), typically from one
# `fasta_reads` drain of the same file (fasta layer).
# ==============================================================================

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
