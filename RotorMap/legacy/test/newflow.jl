using Revise

include(joinpath(@__DIR__, "newfasta.jl")) # the fused packed FASTA loader (+ transitive:
                                           # newencoder2bit.jl -> newencoder.jl, the v3 kernels)

using Random
using CUDA
using Mmap
using ProgressMeter
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders
using Base.Threads

# ==============================================================================
# Streamed FASTA -> 2-bit pack -> v3-encode workflow, one record-aligned batch
# at a time. This merges the two previous stages of the pipeline into a
# bounded-memory producer/consumer flow:
#
#   newindex2.jl:  load_fasta_mmap (all reads as Vector{Vector{UInt8}}) -> rc append
#                  + jagged_array (8 bits/base) -> one giant encode of everything
#   newfasta.jl:   load_fasta_packed (the WHOLE file's bitstream + starts/stops/
#                  heads materialized on the host) -> cu() upload of everything ->
#                  one giant v3 encode of everything
#   newflow.jl:    per batch i: the parse+pack of batch i+1 runs on a spawned
#                  task and overlaps the upload, encode and sinking of batch i.
#
# The pipeline, per batch i:
#
#     host:  [spawn parse+pack batch i+1]  (multithreaded over record-aligned
#             |                             sub-chunks, like load_fasta_packed)
#             | [upload batch i]     copyto! -> stream-ordered async H2D
#             | [encode batch i]     kernel_best_v3 on the packed batch (fwd + rc)
#             | [CUDA.synchronize]   everything above is host-async up to here
#             | [sink(batch i)]      consume embeddings while batch i+1 parses
#             v fetch(parse i+1)     (typically already finished)
#
# Everything the old flow materialized per dataset is now bounded by the batch
# size: the host holds one batch of packed words (+ rc words + starts/stops/
# heads) plus the mmap'd file (page cache), the device one batch of packed
# input and one of embeddings -- so files with more reads than memory work,
# and the whole-file Int32 offset limit of pack_reads disappears (offsets are
# batch-local and restart at 1).
#
# BATCHING SEMANTICS
#
# Batches are cut at record starts ('>' that begins a line), `batch_reads`
# records each; `trim`/`skipN` may drop records, so a batch can emit fewer
# reads than requested. Each batch's bitstream is exactly `pack_reads` of the
# reads it contains (the accumulator starts fresh per batch): bitwise
# identical to packing those reads alone, asserted in the test. Consequently
# the STREAM output differs from load_fasta_packed's WHOLE-FILE bitstream only
# by the per-batch trailing partial word + zero guard word, which never reach
# an s-mer; the per-batch v3 encodings equal the corresponding slices of the
# whole-file encoding up to the usual atomic-order rounding (asserted).
#
# Like load_fasta_packed, `randomize_N` makes the output depend on the
# chunk/batch layout (rand consumption), so it is not bitwise-reproducible
# across different `batch_reads`/`chunks`.
# ==============================================================================

# ---------------------------------------------------------------------------
# Batch boundary scan: byte offset of the (nrec+1)-th record start at or after
# `from` ('>' that begins a line) -- the exclusive end of the batch holding the
# records numbered from..from+nrec-1 -- or length(raw)+1 if fewer remain.
# Resumes from `from`, so walking the whole file is one linear scan.
# ---------------------------------------------------------------------------
function _next_bound(raw, from::Int, nrec::Int)
    n = length(raw)
    GT = UInt8('>'); NL = UInt8('\n'); CR = UInt8('\r')
    i = from
    found = 0
    while found < nrec + 1
        i = something(findnext(==(GT), raw, i), n + 1)
        i > n && return n + 1
        if i == 1 || raw[i-1] == NL || raw[i-1] == CR
            found += 1
        end
        i += 1
    end
    return i - 1 # position of the '>' opening the record AFTER the batch
end

# byte offsets of chunk boundaries INSIDE a batch window [lo, hi): record
# starts nearest to an even split (windowed variant of newfasta._chunk_bounds);
# returns [lo, ..., hi]
function _chunk_bounds(raw, lo::Int, hi::Int, nchunks::Int)
    GT = UInt8('>'); NL = UInt8('\n'); CR = UInt8('\r')
    bounds = [lo]
    if nchunks > 1
        step = cld(hi - lo, nchunks)
        b = lo + step
        while b < hi
            # first '>' at/after b that begins a line, still inside the window
            i = something(findnext(==(GT), raw, b), hi)
            while i < hi && i > 1 && raw[i-1] != NL && raw[i-1] != CR
                i = something(findnext(==(GT), raw, i + 1), hi)
            end
            i < hi || break # no further record in the window: last chunk covers the rest
            push!(bounds, i)
            b += step
        end
    end
    push!(bounds, hi)
    return bounds
end

# parse one batch window [lo, hi): like load_fasta_packed, scoped to the
# window -- record-aligned sub-chunks parsed in parallel, merged with the
# exact funnel shift; the batch's offsets start at 1 (batch-local)
function _parse_batch(it, lo::Int, hi::Int)
    raw = it.raw
    nch = it.chunks
    if nch <= 0
        nch = nthreads()
        nch > 1 && (nch = clamp(div(hi - lo, 1 << 20), 1, nch)) # >= ~1MB per chunk
    end

    if nch == 1
        words, acc, nbits, rwords, racc, rnbits, starts, stops, heads, _ =
            _parse_chunk_packed(raw, lo, hi; trim = it.trim, skipN = it.skipN,
                                randomize_N = it.randomize_N, rc = it.revcomp)
    else
        bounds = _chunk_bounds(raw, lo, hi, nch)
        nch = length(bounds) - 1
        results = Vector{Any}(undef, nch)
        @sync for ci in 1:nch
            clo, chi = bounds[ci], bounds[ci+1]
            Threads.@spawn results[ci] = _parse_chunk_packed(raw, clo, chi;
                                                             trim = it.trim, skipN = it.skipN,
                                                             randomize_N = it.randomize_N,
                                                             rc = it.revcomp)
        end
        # merge in order: exact bitwise concatenation of the chunk-local streams
        words = UInt32[]; acc = UInt64(0); nbits = 0
        rwords = UInt32[]; racc = UInt64(0); rnbits = 0
        starts = Int32[]; stops = Int32[]; heads = String[]
        nbases = 0
        for ci in 1:nch
            cwords, cacc, cnbits, crwords, cracc, crnbits, cstarts, cstops, cheads, _ = results[ci]
            append!(starts, cstarts .+ Int32(nbases))
            append!(stops, cstops .+ Int32(nbases))
            append!(heads, cheads)
            acc, nbits = _merge_chunk!(words, acc, nbits, cwords, cacc, cnbits)
            if it.revcomp
                racc, rnbits = _merge_chunk!(rwords, racc, rnbits, crwords, cracc, crnbits)
            end
            nbases += isempty(cstops) ? 0 : Int(cstops[end])
        end
    end

    # trailing partial word + one zero guard word (as in pack_reads)
    nbits > 0 && push!(words, acc % UInt32)
    push!(words, UInt32(0))
    if it.revcomp
        rnbits > 0 && push!(rwords, racc % UInt32)
        push!(rwords, UInt32(0))
    end

    return (words = words, starts = starts, stops = stops,
            rwords = it.revcomp ? rwords : nothing, heads = heads)
end

"""
Lazily iterates over record-aligned input batches of a FASTA file, parsed and
2-bit packed on demand (the streaming equivalent of `load_fasta_packed`).
Each `iterate` returns

    (words, starts, stops, rwords, heads), next_pos

where `words` is the batch's packed bitstream (reversed base order + zero
guard word -- the exact `pack_reads` format for the reads in the batch),
`starts`/`stops` are batch-local (1-based, Int32), `rwords` the packed
reverse-complement batch (`nothing` unless `revcomp = true`) and `heads` the
raw header lines of the kept reads. The file is mmap'd once; only the current
batch's data is materialized.
"""
struct FastaBatches
    raw::Vector{UInt8} # the mmap'd file
    batch_reads::Int   # records ('>' lines) per batch, upper bound on kept reads
    chunks::Int        # parallel parse chunks per batch (0 = auto)
    trim::Int
    skipN::Bool
    randomize_N::Bool
    revcomp::Bool
end

function FastaBatches(file::String; batch_reads::Int = 2^13, chunks::Int = 0, trim::Int = 0,
                      skipN::Bool = false, randomize_N::Bool = false, revcomp::Bool = false)
    @assert batch_reads > 0
    raw = filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end
    return FastaBatches(raw, batch_reads, chunks, trim, skipN, randomize_N, revcomp)
end

Base.eltype(::Type{FastaBatches}) = Any # variable-length named tuples

function Base.iterate(it::FastaBatches, pos::Int = 1)
    pos > length(it.raw) && return nothing
    hi = _next_bound(it.raw, pos, it.batch_reads)
    return (_parse_batch(it, pos, hi), hi)
end

"""
    encode_fasta_stream(re, file; normalize=0, batch_reads=2^13, revcomp=true,
                        trim=0, skipN=false, randomize_N=false, chunks=0,
                        progress=true, return_heads=false, sink=nothing)
        -> (n_reads, nbases, nbatches, heads)

Streamed version of `encode_fasta_cuda` (newfasta.jl): reads `file` in
record-aligned batches of `batch_reads` records, and for each batch uploads
the packed bitstream and encodes it with `kernel_best_v3` (forward +, when
`revcomp = true`, the reverse-complement batch, sharing `starts`/`stops`).
The parse of batch i+1 runs on a spawned task and overlaps the upload, encode
and sinking of batch i, so CPU and GPU stay busy concurrently and only one
batch of data is alive at a time (see the file header).

`sink`, when given, is called per batch with a NamedTuple

    (cropes, norms, cropes_rc, norms_rc, heads, starts, stops, first_read, ibatch)

where `cropes::CuArray{ComplexF32}` of size (m, 4^c, n) are the batch's
embeddings (the m*4^c bins are flattened over the first two dims, exactly as
the v3 wrapper lays them out; device-resident views -- copy or consume them
inside the sink, before the next batch's GPU work), `cropes_rc`/`norms_rc` the
reverse-complement batch (`nothing` unless `revcomp = true`), `heads` the
batch's headers, and `first_read` the 1-based index of the batch's first read
in the stream. Typical sinks: collect on host
(`out[:, :, f:f+n-1] .= Array(nt.cropes)`), a ComplexF16 downcast + write, or
a per-batch index search.

Returns `(n_reads = ..., nbases = ..., nbatches = ..., heads = ...)`; `heads`
collects all headers in order (`nothing` unless `return_heads = true`).

Note: with `randomize_N = true` the output depends on the batch/chunk layout
(rand consumption), like `load_fasta_packed`.
"""
function encode_fasta_stream(re::RopeEncoder, file::String;
                             normalize = 0,
                             batch_reads::Int = 2^13,
                             revcomp::Bool = true,
                             trim::Int = 0,
                             skipN::Bool = false,
                             randomize_N::Bool = false,
                             chunks::Int = 0,
                             progress::Bool = true,
                             return_heads::Bool = false,
                             sink = nothing)
    it = FastaBatches(file; batch_reads, chunks, trim, skipN, randomize_N, revcomp)

    prog = progress ? ProgressUnknown(desc = "Streaming batches ", dt = 1.0) : nothing
    heads_all = return_heads ? String[] : nothing
    total_reads = 0
    total_bases = 0
    nbatches = 0

    cur = iterate(it) # blocking parse of batch 1; no GPU work queued before it
    while cur !== nothing
        nt, pos = cur
        ibatch = nbatches + 1

        # queue the next parse now: it overlaps this batch's upload/encode/sink
        tsk = pos <= length(it.raw) ? (Threads.@spawn iterate(it, pos)) : nothing

        n = length(nt.starts)
        dest = dest_norms = rdest = rnorms = nothing
        if n > 0
            dwords = CuArray{UInt32}(undef, length(nt.words))
            copyto!(dwords, nt.words) # stream-ordered async H2D (staged by the driver)
            dstarts = CuArray{Int32}(undef, n)
            copyto!(dstarts, nt.starts)
            dstops = CuArray{Int32}(undef, n)
            copyto!(dstops, nt.stops)

            dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, n) # (m, 4^c, n): the kernel writes
            # the m*4^c bins flattened over the first two dims, as in the v3 wrapper
            dest_norms = CUDA.zeros(Float32, re.m, n)
            encode_batch_cuda_all_best_v3!(dest, dest_norms, re, dwords;
                                           starts = dstarts, stops = dstops, normalize = normalize)
            if revcomp
                drwords = CuArray{UInt32}(undef, length(nt.rwords))
                copyto!(drwords, nt.rwords)
                rdest = CUDA.zeros(ComplexF32, re.m, 4^re.c, n)
                rnorms = CUDA.zeros(Float32, re.m, n)
                encode_batch_cuda_all_best_v3!(rdest, rnorms, re, drwords;
                                               starts = dstarts, stops = dstops, normalize = normalize)
            end
            CUDA.synchronize() # uploads + encodes done; the views below are safe to consume
        end

        if sink !== nothing && n > 0
            sink((cropes = view(dest, :, :, 1:n),
                  norms = view(dest_norms, :, 1:n),
                  cropes_rc = rdest === nothing ? nothing : view(rdest, :, :, 1:n),
                  norms_rc = rnorms === nothing ? nothing : view(rnorms, :, 1:n),
                  heads = nt.heads,
                  starts = nt.starts,
                  stops = nt.stops,
                  first_read = total_reads + 1,
                  ibatch = ibatch))
        end

        total_reads += n
        total_bases += n > 0 ? Int(nt.stops[end]) : 0
        nbatches = ibatch
        heads_all === nothing || append!(heads_all, nt.heads)
        prog === nothing || update!(prog, total_reads)

        cur = tsk === nothing ? nothing : fetch(tsk) # typically already finished
    end
    prog === nothing || finish!(prog)

    return (n_reads = total_reads, nbases = total_bases, nbatches = nbatches, heads = heads_all)
end

# ==============================================================================

# per-batch assertions shared by the tests: the streamed batch must equal the
# corresponding slice of the whole-file encoding (up to the atomic-order
# rounding, as in check_pair3), with matching headers and (rebased) offsets
function _check_batch(nt, heads_ref, starts_ref, stops_ref; cpp = nothing, nrm = nothing,
                      cpp_rc = nothing, nrm_rc = nothing)
    f = nt.first_read
    nn = length(nt.starts)
    sl = f:f+nn-1
    @assert nt.heads == heads_ref[sl] "streamed heads mismatch (batch $(nt.ibatch))"
    @assert nt.starts == starts_ref[sl] .- (starts_ref[f] - Int32(1)) "streamed starts mismatch"
    @assert nt.stops == stops_ref[sl] .- (starts_ref[f] - Int32(1)) "streamed stops mismatch"
    c_ref = Array(view(cpp[], :, :, sl))
    c_str = Array(nt.cropes)
    err = maximum(abs.(c_ref .- c_str))
    @assert isapprox(c_ref, c_str; rtol = 1e-3, atol = 2e-4) "streamed encode mismatch (batch $(nt.ibatch))"
    @assert isapprox(Array(view(nrm[], :, sl)), Array(nt.norms); rtol = 1e-3, atol = 2e-4) "streamed norms mismatch (batch $(nt.ibatch))"
    if cpp_rc !== nothing
        @assert isapprox(Array(view(cpp_rc[], :, :, sl)), Array(nt.cropes_rc); rtol = 1e-3, atol = 2e-4) "streamed rc encode mismatch (batch $(nt.ibatch))"
        @assert isapprox(Array(view(nrm_rc[], :, sl)), Array(nt.norms_rc); rtol = 1e-3, atol = 2e-4) "streamed rc norms mismatch (batch $(nt.ibatch))"
    end
    return err
end

function test()
    @show CUDA.name(device())
    @show nthreads()

    dir = mktempdir(prefix = "newflow_")
    L = 20_000
    n_reads = 2^13

    # ---- 1. generate a FASTA file with the project's Utils (same as newfasta) -
    @info "Generating $n_reads mutated reads of ~$L bp and saving the FASTA"
    ref = generate_reference(2^22; seed = 1234)
    reads, pos = generate_reads(ref, L, n_reads; err = 0.02, seed = 42) # indels -> jagged lengths
    fasta = joinpath(dir, "reads.fasta")
    heads = [">read_$i pos=$(pos[i])" for i in eachindex(reads)]
    @time save_fasta(reads, fasta, heads = heads)
    @show filesize(fasta)

    # ---- 2. reference: the non-streamed path (load_fasta_packed + one big v3) -
    @info "Reference: load_fasta_packed (whole file) + v3 encode (whole batch)"
    @time words, starts, stops, rwords, heads_ref = load_fasta_packed(fasta; revcomp = true, progress = false)
    @time begin
        dnas = cu(words); rdnas = cu(rwords)
        s_g = cu(starts); e_g = cu(stops)
    end

    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)
    batch_reads = 3000 # 8192 records -> batches of 3000 + 3000 + 2192 (exercises the tail)

    # ---- 3. streamed vs whole-batch, all normalize modes -----------------------
    for normalize in 0:3
        cref, nref = encode_batch_cuda_all_best_v3(re, dnas; starts = s_g, stops = e_g, normalize = normalize)
        cref_rc, nref_rc = encode_batch_cuda_all_best_v3(re, rdnas; starts = s_g, stops = e_g, normalize = normalize)
        @info "Streaming vs whole-batch (s=8, m=4, c=4), normalize=$normalize"
        maxerr = 0.0
        res = encode_fasta_stream(re, fasta; normalize, batch_reads,
                                  sink = nt -> (maxerr = max(maxerr,
                                                            _check_batch(nt, heads_ref, starts, stops;
                                                                         cpp = Ref(cref), nrm = Ref(nref),
                                                                         cpp_rc = Ref(cref_rc), nrm_rc = Ref(nref_rc)))),
                                  progress = false)
        @assert res.n_reads == n_reads && res.nbatches == 3
        @assert res.nbases == Int(stops[end])
        @assert res.heads === nothing # return_heads defaults to false
        @info "  normalize=$normalize" batches = res.nbatches max_diff = maxerr
    end

    # return_heads + revcomp = false path
    res_h = encode_fasta_stream(re, fasta; normalize = 0, batch_reads, revcomp = false,
                                return_heads = true, progress = false,
                                sink = nt -> (@assert nt.cropes_rc === nothing && nt.norms_rc === nothing))
    @assert res_h.heads == heads_ref "streamed heads (return_heads) mismatch"
    @assert res_h.n_reads == n_reads

    # ---- 4. other encoder configs through the stream ---------------------------
    for (s_, m_) in ((5, 1), (16, 2))
        re2 = RopeEncoder(k = L, s = s_, m = m_, c = 4)
        cref, nref = encode_batch_cuda_all_best_v3(re2, dnas; starts = s_g, stops = e_g, normalize = 0)
        @info "Streaming vs whole-batch (s=$s_, m=$m_, c=4), forward only"
        maxerr = 0.0
        encode_fasta_stream(re2, fasta; normalize = 0, batch_reads, revcomp = false,
                            sink = nt -> (maxerr = max(maxerr,
                                                       _check_batch(nt, heads_ref, starts, stops;
                                                                    cpp = Ref(cref), nrm = Ref(nref)))),
                            progress = false)
        @info "  s=$s_, m=$m_" max_diff = maxerr
    end

    # ---- 5. one giant batch == the whole-file path ------------------------------
    @info "Streaming with batch_reads > n_reads (single batch)"
    cref, nref = encode_batch_cuda_all_best_v3(re, dnas; starts = s_g, stops = e_g, normalize = 0)
    cref_rc, nref_rc = encode_batch_cuda_all_best_v3(re, rdnas; starts = s_g, stops = e_g, normalize = 0)
    encode_fasta_stream(re, fasta; normalize = 0, batch_reads = 10^6,
                        sink = nt -> _check_batch(nt, heads_ref, starts, stops;
                                                  cpp = Ref(cref), nrm = Ref(nref),
                                                  cpp_rc = Ref(cref_rc), nrm_rc = Ref(nref_rc)),
                        progress = false)

    # ---- 6. synthetic edge cases: each batch == pack_reads of its reads ---------
    # tiny file, batch_reads = 1 (one record per batch): a batch's bitstream
    # must equal pack_reads of exactly the reads in that batch, independently
    # of the whole-file layout -- fwd and rc. ('>' must begin a line -- the same
    # record-boundary contract as load_fasta_packed/_chunk_bounds.)
    syn = joinpath(dir, "synthetic.fasta")
    write(syn,
        "junk before the first record\n",
        ">r1 desc\nACGTacgtNnGT\n",   # lowercase; N/n treated as G
        ">r2\r\nAC\r\nGTAA\r\n",      # CRLF line endings
        ">r3\n",                      # empty sequence -> zero-length read
        ">r4\nACGT\n",                # shorter than trim -> dropped
        ">r5 last\nTTTTTTTT")          # no trailing newline
    for trim in (0, 5)
        seqs_s, heads_s = load_fasta_mmap_fixed(syn; trim = trim)
        first = 1
        nb = 0
        for nt in FastaBatches(syn; batch_reads = 1, revcomp = true, trim) # one record per batch
            nn = length(nt.starts)
            sl = first:first + nn - 1
            lw, ls, le = pack_reads(seqs_s[sl])
            @assert nt.words == lw "batch bitstream != pack_reads of the batch's reads (trim=$trim)"
            @assert nt.starts == ls && nt.stops == le "batch offsets mismatch (trim=$trim)"
            lwrc, _, _ = pack_reads([UInt8(3) .- reverse(a) for a in seqs_s[sl]])
            @assert nt.rwords == lwrc "batch rc bitstream mismatch (trim=$trim)"
            @assert nt.heads == heads_s[sl] "batch heads mismatch (trim=$trim)"
            first += nn
            nb += 1
        end
        @assert first - 1 == length(seqs_s) "read count mismatch (trim=$trim)"
        @info "Synthetic edge cases OK (trim=$trim): $nb single-read batches, each == pack_reads of its reads"
    end

    # empty file -> zero batches, no error
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    res_e = encode_fasta_stream(re, empty_fasta; progress = false)
    @assert res_e.n_reads == 0 && res_e.nbatches == 0 && res_e.nbases == 0
    @info "Empty file OK: $(res_e.nbatches) batches"

    # ---- 7. timings --------------------------------------------------------------
    @info "Warming the page cache"
    open(fasta, "r") do io
        while !eof(io)
            read(io, 1 << 22)
        end
    end

    normalize = 0
    @info "Timing the whole-file path: load_fasta_packed -> cu -> v3 (fwd+rc) -> Array"
    local cropes_full, cropes_rc_full # survive the loop scope
    for _ in 1:3
        @time begin
            w2, s2, e2, rw2, _ = load_fasta_packed(fasta; revcomp = true, progress = false)
            w_g = cu(w2); rw_g = cu(rw2); s_g2 = cu(s2); e_g2 = cu(e2)
            c1, _ = encode_batch_cuda_all_best_v3(re, w_g; starts = s_g2, stops = e_g2, normalize = normalize)
            c2, _ = encode_batch_cuda_all_best_v3(re, rw_g; starts = s_g2, stops = e_g2, normalize = normalize)
            cropes_full = Array(c1)
            cropes_rc_full = Array(c2)
        end
    end

    @info "Timing the streamed path: encode_fasta_stream with a collecting sink (batch_reads=2048)"
    cropes_all = Array{ComplexF32}(undef, re.m, 4^re.c, n_reads)
    cropes_rc_all = similar(cropes_all)
    function write_batch(nt)
        f = nt.first_read
        nn = length(nt.starts)
        cropes_all[:, :, f:f+nn-1] .= Array(nt.cropes)
        cropes_rc_all[:, :, f:f+nn-1] .= Array(nt.cropes_rc)
    end
    for _ in 1:3
        @time encode_fasta_stream(re, fasta; normalize, batch_reads = 2048,
                                  sink = write_batch, progress = false)
    end
    @info "End-to-end max diff whole-file vs streamed (fwd)" max_diff = maximum(abs.(cropes_full .- cropes_all))
    @info "End-to-end max diff whole-file vs streamed (rc)" max_diff = maximum(abs.(cropes_rc_full .- cropes_rc_all))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
