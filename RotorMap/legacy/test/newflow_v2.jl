# ==============================================================================
# newflow_v2.jl -- the streamed FASTA -> 2-bit pack -> v3-encode workflow, with
# the v2 (E1/E2) loader as the parse engine, at 2^17 reads x ~20,000 bp
# (~2.6 GB FASTA).
#
# newflow.jl cut the newfasta.jl whole-file flow into record-aligned BATCHES and
# overlapped the stages per batch, but kept the baseline parse engine
# (FastaBatches: push!-grown chunk streams + a scalar per-byte collect loop).
# newfasta_v2.jl showed that engine to be the pipeline's real ceiling: 1.65 GB/s
# at 16 threads (10.3 GiB alloc/pass, GC-bound, no scaling past 4 threads),
# 5.6x below its own E2 variant (8.5+ GB/s, alloc-free, above the cold-disk
# rate). This file is newflow.jl with FastaBatchesV2 wired in:
#
#   newflow.jl    encode_fasta_stream    (FastaBatches,     E0 parse)
#   newflow_v2.jl encode_fasta_stream_v2 (FastaBatchesV2,   E1/E2 parse)
#
# The pipeline shape is unchanged -- per batch i:
#
#     host:  [spawn parse+pack batch i+1]  (v2: parallel record scan -> parallel
#     |                             chunk parse into pooled buffers -> parallel
#     |                             funnel merge; alloc-free steady state)
#     | [upload batch i]     copyto! -> stream-ordered async H2D
#     | [encode batch i]     kernel_best_v3 on the packed batch (fwd + rc)
#     | [CUDA.synchronize]   everything above is host-async up to here
#     | [sink(batch i)]      consume embeddings while batch i+1 parses
#     v fetch(parse i+1)     (typically already finished)
#
# What the v2 engine changes:
#   * parse throughput scales with threads to 16 (vs the baseline's 4-thread
#     ceiling): the streamed pipeline stops being GC-bound and becomes bound by
#     the DRAM traffic of streaming the file + both packed streams once;
#   * the default batch_reads is 2^15 -- the loader's sweet spot at ~20 kb
#     reads (per-batch scan/prefault/merge overhead amortizes; ~655 MB of
#     batch output per batch);
#   * batches stay record-aligned and bitwise == pack_reads of their reads,
#     and batch-local Int32 offsets keep each batch below the 2^31-base limit,
#     so arbitrarily large files still work -- but the GLOBAL base position
#     exceeds Int32 for such files and is the consumer's business (tracked via
#     `first_read` and Int64 counters; the 2^17 x 20 kb file of the tests is
#     exactly such a case, which is why the big-file checks run batched).
#
# Semantics are identical to encode_fasta_stream: batches are cut at line-start
# '>' records, trim/skipN can drop records, `randomize_N` output depends on the
# batch/chunk layout (rand consumption), `swar = false` selects the E1 scalar
# parse path (randomize_N forces it anyway), keep_heads = false drops headers.
#
# RESULTS (RTX 5090, kau, 2^17 reads x ~20 kb = 2.62 GB, batch_reads = 2^15,
# fwd + rc, pinned collecting sink; full notes in prompts/newflow_v2.md):
# at -t 16 the v2 pipeline runs the file + a 2.15 GB embedding downcast in
# 0.51 s (5.1 GB/s of FASTA consumed) vs 2.33 s for the baseline-engine
# pipeline (4.5x), with parse-only at 0.39 s (6.7 GB/s vs the baseline's 1.09).
# Unlike the baseline engine the pipeline keeps scaling to 16 threads (0.78 s
# at -t 4 -> 0.51 s at -t 16), and device residency stays at one batch
# (~1.7 GiB pool peak) no matter the file size. Correctness: -t 2 and -t 16
# ALL PASS (small-file parity vs the baseline whole-file path for all
# normalize modes/configs/synthetic edge cases; big-file lockstep bitwise
# parity + batch-layout invariance across two different batch cuts).
# ==============================================================================

include(joinpath(@__DIR__, "newfasta_v2.jl")) # FastaBatchesV2, the E1/E2 parse engine
# (+ transitively: newflow.jl -> newfasta.jl -> newencoder2bit.jl, i.e. the
#  baseline FastaBatches / encode_fasta_stream for comparisons, load_fasta_packed,
#  pack_reads and the v3 encode kernels; plus the v2 test harness: ensure_data,
#  _timed_min, _warm_cache, _reps, _v2time_print)

using CUDA
using Printf
using ProgressMeter
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders
using Base.Threads

"""
    encode_fasta_stream_v2(re, file; normalize=0, batch_reads=2^15, revcomp=true,
                           trim=0, skipN=false, randomize_N=false, chunks=0,
                           swar=true, keep_heads=true, progress=true,
                           return_heads=false, sink=nothing)
        -> (n_reads, nbases, nbatches, heads)

`encode_fasta_stream` (newflow.jl) with the v2 loader as the parse engine:
`FastaBatchesV2` (newfasta_v2.jl) cuts the file into record-aligned batches of
`batch_reads` records and parses each with the parallel record scan ->
parallel chunk parse -> parallel funnel-merge pipeline (exact preallocated
outputs, pooled and reused chunk buffers: ~no steady-state allocation, E2
pair-LUT SWAR fast path when `swar = true`, which `randomize_N` disables).
The parse of batch i+1 runs on a spawned task and overlaps the upload, encode
and sinking of batch i, so CPU, PCIe and GPU stay busy concurrently and only
one batch of input (host) plus one of output (device) is alive at a time.

`sink`, when given, is called per batch with a NamedTuple

    (cropes, norms, cropes_rc, norms_rc, heads, starts, stops, first_read, ibatch)

exactly as in `encode_fasta_stream`: `cropes::CuArray{ComplexF32}` of size
(m, 4^c, n) are the batch's embeddings (the m*4^c bins flattened over the
first two dims, device-resident views -- copy or consume them inside the sink,
before the next batch's GPU work), `cropes_rc`/`norms_rc` the
reverse-complement batch (`nothing` unless `revcomp = true`), `heads` the
batch's headers and `first_read` the 1-based index of the batch's first read
in the stream.

Base offsets are batch-local Int32 (a batch must stay below 2^31 bases), but
the stream as a whole has no limit -- consumers track the global position via
`first_read` and their own base counters (Int64; files above 2^31 bases are
exactly the point of streaming).

Returns `(n_reads = ..., nbases = ..., nbatches = ..., heads = ...)`; `heads`
collects all headers in order (`nothing` unless `return_heads = true`).

Note: with `randomize_N = true` the output depends on the batch/chunk layout
(rand consumption), like `load_fasta_packed`.
"""
function encode_fasta_stream_v2(re::RopeEncoder, file::String;
                                normalize = 0,
                                batch_reads::Int = 2^15,
                                revcomp::Bool = true,
                                trim::Int = 0,
                                skipN::Bool = false,
                                randomize_N::Bool = false,
                                chunks::Int = 0,
                                swar::Bool = true,
                                keep_heads::Bool = true,
                                progress::Bool = true,
                                return_heads::Bool = false,
                                sink = nothing)
    it = FastaBatchesV2(file; batch_reads, chunks, trim, skipN, randomize_N, revcomp, swar, keep_heads)

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

# per-batch check against BIG-file references: global segmentation (Int64 --
# the file exceeds the 2^31-base Int32 range, so whole-file Int32 offsets are
# impossible) and the collected reference encodings
function _check_batch_big(nt, gheads, gstarts64, gstops64, Afwd, Anrm, Arc, Anrm_rc;
                          rtol = 1e-3, atol = 2e-4)
    f = nt.first_read
    nn = length(nt.starts)
    sl = f:f+nn-1
    @assert nt.heads == gheads[sl] "streamed heads mismatch (batch $(nt.ibatch))"
    off = gstarts64[f] - 1
    @assert Int64.(nt.starts) == gstarts64[sl] .- off "streamed starts mismatch (batch $(nt.ibatch))"
    @assert Int64.(nt.stops) == gstops64[sl] .- off "streamed stops mismatch (batch $(nt.ibatch))"
    c_str = Array(nt.cropes)
    err = maximum(abs.(c_str .- view(Afwd, :, :, sl)))
    @assert isapprox(c_str, view(Afwd, :, :, sl); rtol, atol) "streamed encode mismatch (batch $(nt.ibatch))"
    @assert isapprox(Array(nt.norms), view(Anrm, :, sl); rtol, atol) "streamed norms mismatch (batch $(nt.ibatch))"
    @assert isapprox(Array(nt.cropes_rc), view(Arc, :, :, sl); rtol, atol) "streamed rc encode mismatch (batch $(nt.ibatch))"
    @assert isapprox(Array(nt.norms_rc), view(Anrm_rc, :, sl); rtol, atol) "streamed rc norms mismatch (batch $(nt.ibatch))"
    return err
end

function test()
    @show CUDA.name(device())
    @show nthreads()

    small, big = ensure_data()
    L = 20_000

    # ==========================================================================
    # A. SMALL FILE (2^13 reads, 164 MB): whole-file references are legal here,
    #    so the streamed v2 flow is checked end to end against the BASELINE
    #    whole-file path (load_fasta_packed + one giant v3 encode).
    # ==========================================================================
    n_reads = 2^13
    @info "A. Small file: streamed v2 flow vs the baseline whole-file path ($n_reads reads)"
    words, starts, stops, rwords, heads_ref = load_fasta_packed(small; revcomp = true, progress = false)
    dnas = cu(words); rdnas = cu(rwords)
    s_g = cu(starts); e_g = cu(stops)

    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)
    batch_reads = 3000 # 8192 records -> 3000 + 3000 + 2192 (exercises the tail)

    # normalize=0 references, reused by several checks below
    cref0, nref0 = encode_batch_cuda_all_best_v3(re, dnas; starts = s_g, stops = e_g, normalize = 0)
    cref0_rc, nref0_rc = encode_batch_cuda_all_best_v3(re, rdnas; starts = s_g, stops = e_g, normalize = 0)

    # ---- A1. all normalize modes, fwd + rc ----------------------------------
    for normalize in 0:3
        cref, nref = encode_batch_cuda_all_best_v3(re, dnas; starts = s_g, stops = e_g, normalize)
        cref_rc, nref_rc = encode_batch_cuda_all_best_v3(re, rdnas; starts = s_g, stops = e_g, normalize)
        maxerr = 0.0
        res = encode_fasta_stream_v2(re, small; normalize, batch_reads,
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

    # ---- A2. single giant batch == the whole-file path -----------------------
    @info "  single batch (batch_reads > n_reads)"
    encode_fasta_stream_v2(re, small; normalize = 0, batch_reads = 10^6,
                           sink = nt -> _check_batch(nt, heads_ref, starts, stops;
                                                     cpp = Ref(cref0), nrm = Ref(nref0),
                                                     cpp_rc = Ref(cref0_rc), nrm_rc = Ref(nref0_rc)),
                           progress = false)

    # ---- A3. swar = false (the E1 scalar path) through the flow -------------
    @info "  swar=false (E1 scalar parse) through the flow"
    maxerr = 0.0
    encode_fasta_stream_v2(re, small; normalize = 0, batch_reads, swar = false,
                           sink = nt -> (maxerr = max(maxerr,
                                                      _check_batch(nt, heads_ref, starts, stops;
                                                                   cpp = Ref(cref0), nrm = Ref(nref0),
                                                                   cpp_rc = Ref(cref0_rc), nrm_rc = Ref(nref0_rc)))),
                           progress = false)
    @info "    swar=false OK" max_diff = maxerr

    # ---- A4. keep_heads = false through the flow ----------------------------
    @info "  keep_heads=false through the flow"
    maxerr = 0.0
    res_kh = encode_fasta_stream_v2(re, small; normalize = 0, batch_reads, keep_heads = false,
                                    sink = nt -> begin
                                        f = nt.first_read
                                        nn = length(nt.starts)
                                        @assert isempty(nt.heads)
                                        maxerr2 = maximum(abs.(Array(nt.cropes) .- Array(view(cref0, :, :, f:f+nn-1))))
                                        maxerr = max(maxerr, maxerr2)
                                    end,
                                    progress = false)
    @assert res_kh.n_reads == n_reads && res_kh.nbatches == 3
    @info "    keep_heads=false OK" max_diff = maxerr

    # ---- A5. other encoder configs through the stream ------------------------
    for (s_, m_) in ((5, 1), (16, 2))
        re2 = RopeEncoder(k = L, s = s_, m = m_, c = 4)
        cref, nref = encode_batch_cuda_all_best_v3(re2, dnas; starts = s_g, stops = e_g, normalize = 0)
        maxerr = 0.0
        encode_fasta_stream_v2(re2, small; normalize = 0, batch_reads, revcomp = false,
                               sink = nt -> (maxerr = max(maxerr,
                                                          _check_batch(nt, heads_ref, starts, stops;
                                                                       cpp = Ref(cref), nrm = Ref(nref)))),
                               progress = false)
        @info "  config (s=$s_, m=$m_, c=4), forward only" max_diff = maxerr
    end

    # ---- A6. revcomp = false + return_heads ----------------------------------
    res_h = encode_fasta_stream_v2(re, small; normalize = 0, batch_reads, revcomp = false,
                                   return_heads = true, progress = false,
                                   sink = nt -> (@assert nt.cropes_rc === nothing && nt.norms_rc === nothing))
    @assert res_h.heads == heads_ref "streamed heads (return_heads) mismatch"
    @assert res_h.n_reads == n_reads
    @info "  revcomp=false + return_heads OK"

    # ---- A7. synthetic edge cases: each batch == pack_reads of its reads -----
    # tiny file, batch_reads = 1 (one record per batch): a batch's bitstream
    # must equal pack_reads of exactly the reads in that batch, independently
    # of the whole-file layout -- fwd and rc. ('>' must begin a line -- the
    # record-boundary contract of load_fasta_packed / _chunk_bounds.)
    dir = mktempdir(prefix = "newflow_v2_")
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
        for swar in (false, true)
            first = 1
            nb = 0
            for nt in FastaBatchesV2(syn; batch_reads = 1, revcomp = true, swar, trim) # one record per batch
                nn = length(nt.starts)
                sl = first:first + nn - 1
                lw, ls, le = pack_reads(seqs_s[sl])
                @assert nt.words == lw "batch bitstream != pack_reads of the batch's reads (trim=$trim, swar=$swar)"
                @assert nt.starts == ls && nt.stops == le "batch offsets mismatch (trim=$trim, swar=$swar)"
                lwrc, _, _ = pack_reads([UInt8(3) .- reverse(a) for a in seqs_s[sl]])
                @assert nt.rwords == lwrc "batch rc bitstream mismatch (trim=$trim, swar=$swar)"
                @assert nt.heads == heads_s[sl] "batch heads mismatch (trim=$trim, swar=$swar)"
                first += nn
                nb += 1
            end
            @assert first - 1 == length(seqs_s) "read count mismatch (trim=$trim, swar=$swar)"
            @info "  synthetic edge cases OK (trim=$trim, swar=$swar): $nb single-read batches == pack_reads of their reads"
        end
    end

    # ---- A8. empty file -> zero batches, no error ----------------------------
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    res_e = encode_fasta_stream_v2(re, empty_fasta; progress = false)
    @assert res_e.n_reads == 0 && res_e.nbatches == 0 && res_e.nbases == 0
    @info "  empty file OK: $(res_e.nbatches) batches"

    # ==========================================================================
    # B. BIG FILE (2^17 reads x ~20 kb, ~2.6 GB): exceeds the 2^31-base Int32
    #    offset limit, so EVERYTHING runs batched -- this is the scale the
    #    streamed flow exists for. The baseline loader provides the reference
    #    segmentation (bitwise lockstep), the collected v2 encodings provide
    #    the reference for a second pass with a different batch layout.
    # ==========================================================================
    n_big = 2^17
    batch_big = 2^15 # the loader's sweet spot at ~20 kb reads: 131072 -> 4 batches
    @info "B. Big file: $(filesize(big)) bytes, $n_big reads x ~$L bp, batch_reads=$batch_big"

    # ---- B1. lockstep loader parity + global segmentation (Int64 bases) -----
    gstarts64 = Int64[]; gstops64 = Int64[]; gheads_big = String[]
    nbases_big = Int64(0)
    nb = 0
    for (ntb, ntv) in zip(FastaBatches(big; batch_reads = batch_big, revcomp = true),
                          FastaBatchesV2(big; batch_reads = batch_big, revcomp = true))
        nb += 1
        @assert ntb.words == ntv.words "lockstep words mismatch (batch $nb)"
        @assert ntb.rwords == ntv.rwords "lockstep rwords mismatch (batch $nb)"
        @assert ntb.starts == ntv.starts && ntb.stops == ntv.stops "lockstep offsets mismatch (batch $nb)"
        @assert ntb.heads == ntv.heads "lockstep heads mismatch (batch $nb)"
        append!(gstarts64, Int64.(ntb.starts) .+ nbases_big)
        append!(gstops64, Int64.(ntb.stops) .+ nbases_big)
        append!(gheads_big, ntb.heads)
        nbases_big += isempty(ntb.stops) ? 0 : Int64(ntb.stops[end])
    end
    @assert nb == cld(n_big, batch_big) "unexpected batch count"
    @assert length(gstarts64) == n_big && gstarts64[1] == 1 && gstops64[end] == nbases_big
    @info "  lockstep OK: $nb batches, $n_big reads, $nbases_big bases (global, Int64)"

    # ---- B2. v2 stream, collecting sink (batch_reads = 2^15) ----------------
    # per-batch device residency: must stay at ONE batch, not the dataset
    dev_max = Ref(0)
    Afwd = Array{ComplexF32}(undef, re.m, 4^re.c, n_big) # ~1.07 GiB each
    Anrm = Array{Float32}(undef, re.m, n_big)
    Arc = similar(Afwd)
    Anrm_rc = similar(Anrm)
    res = encode_fasta_stream_v2(re, big; normalize = 0, batch_reads = batch_big,
                                 sink = nt -> begin
                                     dev_max[] = max(dev_max[], CUDA.used_memory())
                                     f = nt.first_read
                                     nn = length(nt.starts)
                                     sl = f:f+nn-1
                                     @assert nt.heads == gheads_big[sl] "big heads mismatch (batch $(nt.ibatch))"
                                     off = gstarts64[f] - 1
                                     @assert Int64.(nt.starts) == gstarts64[sl] .- off "big starts mismatch (batch $(nt.ibatch))"
                                     @assert Int64.(nt.stops) == gstops64[sl] .- off "big stops mismatch (batch $(nt.ibatch))"
                                     Afwd[:, :, sl] .= Array(nt.cropes)
                                     Anrm[:, sl] .= Array(nt.norms)
                                     Arc[:, :, sl] .= Array(nt.cropes_rc)
                                     Anrm_rc[:, sl] .= Array(nt.norms_rc)
                                 end,
                                 progress = false)
    @assert res.n_reads == n_big && res.nbatches == nb && res.nbases == nbases_big
    @info "  collecting pass OK" n_reads = res.n_reads nbatches = res.nbatches nbases = res.nbases

    # ---- B3. v2 stream with a DIFFERENT batch layout (3 uneven batches) -----
    # cuts the file at other record boundaries: embeddings must match the
    # collected reference (up to atomic-order rounding), offsets/heads bitwise
    batch3 = cld(n_big, 3) # 131072 -> 43691 + 43691 + 43690 (uneven tail)
    maxerr = 0.0
    res2 = encode_fasta_stream_v2(re, big; normalize = 0, batch_reads = batch3,
                                  sink = nt -> (maxerr = max(maxerr,
                                                             _check_batch_big(nt, gheads_big, gstarts64,
                                                                              gstops64, Afwd, Anrm, Arc, Anrm_rc))),
                                  progress = false)
    @assert res2.n_reads == n_big && res2.nbatches == 3 && res2.nbases == nbases_big
    @info "  layout invariance OK (batch_reads=$batch3 -> 3 uneven batches)" max_diff = maxerr
    @printf("  peak device-pool residency during the stream: %.2f GiB (whole dataset would be %.2f GiB)\n",
            dev_max[] / 2^30, (2 * re.m * 4^re.c * n_big * 8 + 4 * n_big * L / 4) / 2^30)

    @info "ALL CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# big-file benchmarks: parse-only vs full pipeline, baseline vs v2
# ==============================================================================
function bench(; reps = _reps())
    small, big = ensure_data()
    _warm_cache(big)
    L = 20_000
    n_big = 2^17
    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)
    bytes = filesize(big)
    @info "Pipeline benchmarks on the big file ($bytes bytes, $n_big reads)" julia_threads = nthreads() reps

    # ---- parse only -----------------------------------------------------------
    @info "-- parse only (batch_reads = 2^15, fwd + rc) --"
    _timed_min("E0  FastaBatches stream (baseline parse)"; bytes, reps) do
        s = 0
        for nt in FastaBatches(big; batch_reads = 2^15, revcomp = true)
            s += length(nt.words) + length(nt.rwords)
        end
        s
    end
    _v2time_reset!()
    _timed_min("E2  FastaBatchesV2 stream (v2 parse)"; bytes, reps) do
        s = 0
        for nt in FastaBatchesV2(big; batch_reads = 2^15, revcomp = true)
            s += length(nt.words) + length(nt.rwords)
        end
        s
    end
    _v2time_print()

    # ---- full pipeline with a collecting sink ---------------------------------
    @info "-- full pipeline, collecting sink (fwd + rc embeddings to host) --"
    # the collecting sink preallocates PINNED host buffers and D2H-copies into
    # them at linear offsets: pinned-dest copies DMA at ~34 GB/s vs ~5 GB/s for
    # `Array(device_view)` into fresh pageable memory, and no intermediate copy
    # is made (see the memory-rate measurements in prompts/newflow_v2.md)
    Afwd = Array{ComplexF32}(undef, re.m, 4^re.c, n_big)
    Arc = similar(Afwd)
    CUDA.pin(Afwd); CUDA.pin(Arc) # one-time page-lock (~100 ms/GB), reused by every rep
    nb_el = re.m * 4^re.c
    function collect_sink(nt)
        f = nt.first_read
        nn = length(nt.starts)
        off = (f - 1) * nb_el
        copyto!(Afwd, off + 1, parent(nt.cropes), 1, nn * nb_el)
        copyto!(Arc, off + 1, parent(nt.cropes_rc), 1, nn * nb_el)
        return nothing
    end
    t_base = _timed_min("E0  encode_fasta_stream (baseline pipeline)"; bytes, reps) do
        res = encode_fasta_stream(re, big; normalize = 0, batch_reads = 2^15,
                                  progress = false, sink = collect_sink)
        res.n_reads
    end
    _v2time_reset!()
    t_v2 = _timed_min("E2  encode_fasta_stream_v2 (v2 pipeline)"; bytes, reps) do
        res = encode_fasta_stream_v2(re, big; normalize = 0, batch_reads = 2^15,
                                     progress = false, sink = collect_sink)
        res.n_reads
    end
    _v2time_print()
    @printf("  %-48s %7.2fx vs baseline\n", "v2 pipeline speedup", t_base / t_v2)

    # ---- pipeline without a sink: upload + encode + sync only -----------------
    @info "-- pipeline with sink = nothing (upload + encode + sync only) --"
    _timed_min("E0  encode_fasta_stream, sink = nothing"; bytes, reps) do
        res = encode_fasta_stream(re, big; normalize = 0, batch_reads = 2^15, progress = false)
        res.n_reads
    end
    _timed_min("E2  encode_fasta_stream_v2, sink = nothing"; bytes, reps) do
        res = encode_fasta_stream_v2(re, big; normalize = 0, batch_reads = 2^15, progress = false)
        res.n_reads
    end

    println("  (GB/s = FASTA bytes consumed per second; device residency is one batch:")
    @printf("   input ~%.0f MiB + embeddings ~%.0f MiB fwd+rc, vs ~%.1f GiB for the whole dataset)\n",
            2 * (n_big * L ÷ 4) / 2^20, 2 * re.m * 4^re.c * 2^15 * 8 / 2^20,
            (2 * re.m * 4^re.c * n_big * 8 + 2 * n_big * L / 4) / 2^30)
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data()
    mode == "test" && test()
    mode == "bench" && bench()
    mode == "all" && (ensure_data(); test(); bench())
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
