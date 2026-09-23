# ==============================================================================
# e2e_fp8.jl -- END-TO-END HUMAN-GENOME MAPPING on the fp8 flow: the
# indexreal human-reference index (a REAL rope-encoded database, fwd +
# revcomp) loaded as the fp8 GEMM database, the sample reads (mutated
# genome samples) streamed through the rope -> fp8 GEMM -> top-k pipeline,
# and every read's top-k hits checked against its PROVENANCE (ground truth).
#
# PROBLEM
#   The raw fp8 flow study matches its streamed fragment embeddings against a
#   RANDOM database B.  This experiment swaps that B for the real thing and
#   scores the pipeline end to end:
#     index .bin (human GRCh38.p14, k = 20,000, kstep = 2,000 -> embeds
#                 (2*n_frag, 2*m*4^c) fp16, fwd+rc, with heads/starts/strand
#                 location meta; by default the fp8 save: B = e4m3^(
#                 rdim x 2*n_frag) resident on the GPU as a pure H2D)
#     reads fasta (n = 131,072 reads x k = 20,000, err = 0.05)
#        -> rope fp16 stream -> fp8 GEMM + per-row top-k (k = 20)
#        -> GROUND TRUTH: read r is CORRECTLY MAPPED iff its true window
#           [start, start+len-1] in record src INTERSECTS at least one of its
#           k returned database windows (plus the rate, the first-hit rank
#           histogram and the average intersection LENGTH of the BEST hit).
#
# DESIGN
#   1. Whole-matrix load: ALL index rows become database columns -- the
#     forward AND the revcomp halves (an rc hit is the same genomic location,
#     so the intersection test needs no strand special-casing).
#   2. Index -> GPU: the fp8-format save IS B (pure H2D); the legacy fp16-row
#     layout still loads via the same code path (GPU transpose + quantize).
#   3. normalize = 0 ASSUMED (asserted at load).
#   4. Provenance parsing: sample headers AND colon eval headers are
#     understood (parse_read_head, reads/provenance.jl / common/harness.jl).
#   5. Shared runner: `map_and_count` (common/harness.jl) is the whole
#     scoring loop; the tiny self-test and the production run differ only in
#     the inputs.
#
# SELF-TEST (mode `test`; production files not needed)
#   A tiny 1M-base two-record reference is generated ONCE (cached; see
#   common/harness.jl + common/testref.jl), turned into an fp8-format .bin by
#   a miniature of the production save, 1000 reads of k = 20,000 sampled at
#   err = 0, and pushed through the SAME load -> fp8 -> stream -> score path.
#   At err = 0 every read must map (asserted), rank histogram dominated by
#   rank 1.
#
# ENTRY POINT
#   This file is the experiment entry point.  The flow, loader, provenance
#   scorer and tiny-fixture builders moved to the layers; everything below
#   the include list is verbatim from the legacy experiment (same env
#   variable names, same run_* signatures, same mode dispatch).
#
# SOURCE: legacy/test/e2ehuman.jl (env consts E2H_*, run_e2e_test /
#         run_e2e_bench / run_e2e_human + bottom mode dispatch).
#         Full measured RESULTS (RTX 5090 sweeps: kstep/encoder/error-rate
#         matrices, eval reads) live in that file's header.
#
# INCLUDES (canonical order)
#   common/{treearrays,dna,util,testref}.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl, search/engine_fp8.jl,
#   index/{build,load}.jl, reads/provenance.jl, common/harness.jl
#   (harness.jl provides parse_read_head/map_and_count and the _e2h_* tiny
#   fixture builders + E2H_TEST_* consts; index/load.jl provides load_index,
#   index_database, index_to_f8, E2H_FORMAT/E2H_FP8_FORMAT; engine_fp8.jl
#   provides TopKEngine/_gemm_chunk!/upload_fp16_as_f8!/
#   launch_seg_rowtopk_merge!/_timed_gpu/topk_flow/rope_topk_stream).
#   NEVER co-include experiments/e2e_fp16.jl in the same session
#   (engine_fp8/engine_fp16 name clashes).
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (1M reference -> index -> err=0 reads -> all mapped)
#   bench stage timings: rope-only, quantize/gemm/topk microbenches,
#         compute-only, end-to-end (+ mapping score)
#   run   production: human index + n131072_e0.05 reads -> mapping score
#   all   test + bench + run (default)
#   flags: --k=20 (top-k size), --w=65536 (B-column chunk), --batch=8192,
#          --segs=8, --engine=lt (:lt is the production engine), --reps=1
#   env:   E2HUMAN_FASTA / E2HUMAN_INDEX / E2HUMAN_READS
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/e2e_fp8.jl test
#   julia --project=. -t 16 RotorMap/experiments/e2e_fp8.jl run --engine=lt
#   julia --project=. -t 16 RotorMap/experiments/e2e_fp8.jl    # test + run
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

# --- layers (canonical order) -------------------------------------------------
inc("common", "treearrays.jl")
inc("common", "dna.jl")
inc("common", "util.jl")
inc("common", "testref.jl")
inc("fasta", "pack.jl")
inc("fasta", "loader.jl")
inc("fasta", "reader.jl")
inc("encode", "ropeencoder.jl")
inc("encode", "encoder_v3.jl")
inc("encode", "reference.jl")
inc("encode", "kernel.jl")
inc("encode", "stream.jl")
inc("gemm", "fp8_convert.jl")
inc("gemm", "fp8_ptx.jl")
inc("gemm", "fp8_lt.jl")
inc("gemm", "topk_kernels.jl")
inc("search", "engine_fp8.jl")
inc("index", "build.jl")
inc("index", "load.jl")
inc("reads", "provenance.jl")
inc("common", "harness.jl") # last: CALLS index/encode at runtime

using Random
using Printf
using Base.Threads

# Configuration (env overrides)
# ------------------------------------------------------------------------------
const E2H_FASTA = get(ENV, "E2HUMAN_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna")
# the index matrix: by DEFAULT the fp8 save (indexreal's INDEX_FP8
# production layout: transposed e4m3 database columns -- B verbatim, a pure
# H2D at load time).  The classic fp16 row layout still loads: the format tag
# inside the .bin dispatches (see index_database).
const E2H_INDEX_BIN = get(ENV, "E2HUMAN_INDEX",
                          string(splitext(E2H_FASTA)[1], ".indexreal_fp8.bin"))
# the mutated reads (sample `gen` output; the existing k20000 batch on /share)
const E2H_READS = get(ENV, "E2HUMAN_READS",
                      string(splitext(E2H_FASTA)[1],
                             ".sample_n1048576_k20000_e0.1_s42.fasta"))

# E2H_FORMAT / E2H_FP8_FORMAT (index/load.jl) and the tiny-fixture consts
# E2H_TEST_BASES/E2H_TEST_N/E2H_TEST_SEED (common/harness.jl) are layer-provided.

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end: build the 1M reference (once), index it, sample 1000 err = 0
reads, run the FULL load -> fp8 -> stream -> score path, and assert every
read is correctly mapped (its window shares >= 95% of its bases with an index
window -- any miss means broken location bookkeeping, not noise).
"""
function run_e2e_test(; kfrag::Int = 20_000, n::Int = E2H_TEST_N, ktop::Int = 20,
                      w::Int = 2^16, batch_size::Int = 2^13, segs::Int = 8,
                      engine::Symbol = :lt, seed::Int = E2H_TEST_SEED)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    B = index_database(d)
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    w_test = min(w, prevpow(2, N)) # the tiny N (≈10^3) admits no production w
    res = map_and_count(re, readsf, B, d.heads, d.starts, d.k; ktop,
                        w = w_test, batch_size, segs, engine)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped"
    @info "TINY E2E PASSED: $n/$n correctly mapped" first_hit_ranks = res.rank_hist
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

"""
    run_e2e_bench(; ktop, w, batch_size, segs, engine, reps)

Stage timings of the production mapping flow: the fp8 index -> resident B
upload, the rope stage alone (fasta read + 2-bit pack + rope encode, no-op
consumer; GB/s of fasta), per-chunk microbenches (quantize+upload, fp8 GEMM,
segmented top-k), the GEMM+top-k stage alone (compute-only: the materialized
embedding batches replayed through the engine), and the full end-to-end
pipeline (including the ground-truth score).  The rope stage is a separate
TASK pipeline, so in the pipelined flow it hides behind the compute stage
(e2e ~= compute-only); the GEMM and top-k overlap each other through the
double-buffered chunk streams, so their split is visible only in the
microbenches.
"""
function run_e2e_bench(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, reps::Int = 1)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2H_READS) ||
        error("reads fasta not found: $E2H_READS (generate with reads/sample.jl)")
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    t_B = @elapsed B = index_database(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    (ds, dm, dc) = (d.s, d.m, d.c) # bench follows the index's encoder config
    @info "fp8 database" dims = size(B) gib = round(length(B) / 2^30; digits = 2) seconds = round(t_B; digits = 2) load_seconds = round(t_load; digits = 2)
    d = nothing
    GC.gc(); CUDA.reclaim()

    re = RopeEncoder(k = dk, s = ds, m = dm, c = dc)
    N = size(B, 2)
    rdim = size(B, 1)
    chunks = cld(N, w)

    # warm the page cache: on the network /share a cold 2.6 GB read runs at
    # ~0.6 GB/s and would pollute the rope-stage timing with disk bandwidth
    @info "warming the reads fasta page cache..."
    read(E2H_READS)

    # ---- rope stage alone (no-op consumer) + materialize for the replay ----
    # (the first pass pays the pipeline's JIT/first-call cost (~4 s: GPU rope
    # kernel, channels, parser paths), so it is the warmup AND the
    # materialization pass; `reps` timed passes follow, min reported)
    batches = RopeRealBatch{Float16}[]
    t_jit = @elapsed for nt in rope_encode_real_stream(re, E2H_READS; k = re.k,
                                                       batch_size,
                                                       normalize = 0, fp16 = true)
        push!(batches, nt)
    end
    nreads = sum(nt -> size(nt.embeds, 1), batches)
    bytes = filesize(E2H_READS)
    t_rope = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for nt in rope_encode_real_stream(re, E2H_READS; k = re.k, batch_size,
                                              normalize = 0, fp16 = true)
                n += length(nt.heads)
            end
            @assert n == nreads
        end
        t_rope = min(t_rope, t)
    end
    @printf("rope stage  (fasta read + 2-bit pack + rope encode): %.2f s  (%.2f GB/s of fasta, %d reads -> %d batches; JIT pass %.2f s)\n",
            t_rope, bytes / 1e9 / t_rope, nreads, length(batches), t_jit)

    # ---- per-chunk microbenches at the real geometry -----------------------
    eng = TopKEngine(B; k = ktop, w, rows_cap = batch_size, segs, engine)
    emb1 = batches[1].embeds
    @assert size(emb1) == (batch_size, rdim)
    Bp1 = byteptr(pointer(B))
    CUDA.stream!(eng.sg) do
        upload_fp16_as_f8!(eng, emb1)
        _gemm_chunk!(eng, eng.buf1, Bp1, w) # warm (JIT + the lt heuristic)
    end
    CUDA.device_synchronize()
    tq = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            upload_fp16_as_f8!(eng, emb1)
        end
    end
    tg = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            _gemm_chunk!(eng, eng.buf1, Bp1, w)
        end
    end
    D_val = CuMatrix{Float32}(undef, batch_size, ktop)
    D_loc = CuMatrix{Int32}(undef, batch_size, ktop)
    launch_seg_rowtopk_merge!(D_val, D_loc, eng.buf1, 0, ktop; segs) # warm (JIT)
    CUDA.device_synchronize()
    tt = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            launch_seg_rowtopk_merge!(D_val, D_loc, eng.buf1, 0, ktop; segs)
        end
    end
    @printf("  quantize+upload: %.2f ms/batch | fp8 gemm: %.2f ms/chunk (%.1f TFLOPS) | seg top-k: %.2f ms/chunk (%.0f GiB/s)\n",
            tq * 1e3, tg * 1e3, 2.0 * batch_size * rdim * w / tg / 1e12,
            tt * 1e3, batch_size * w * sizeof(eltype(eng.buf1)) / tt / 2^30)

    # ---- compute-only: GEMM+top-k stage, rope excluded (replayed batches) --
    t_cmp = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for bat in topk_flow(batches, eng; out_cap = 2)
                n += size(bat.vals, 1)
            end
            @assert n == nreads
        end
        t_cmp = min(t_cmp, t)
    end
    flops = 2.0 * nreads * rdim * N
    @printf("gemm+topk stage (quantize+H2D -> %d chunks/batch, gemm || topk): %.2f s  (%.1f TFLOPS sustained over 2^%.1f FLOP)\n",
            chunks, t_cmp, flops / t_cmp / 1e12, log2(flops))

    # ---- end-to-end: the full pipelined flow (and the mapping score) -------
    t_e2e = Inf
    correct = total = 0
    for _ in 1:reps
        res = map_and_count(re, E2H_READS, B, heads, starts, dk; ktop, w,
                            batch_size, segs, engine)
        t_e2e = min(t_e2e, res.seconds)
        correct = res.correct
        total = res.total
    end
    @printf("END-TO-END (fasta -> rope -> fp8 gemm -> topk -> score): %.2f s  (%d/%d = %.2f%% correctly mapped)\n",
            t_e2e, correct, total, 100 * correct / total)
    hides = t_e2e < t_rope + t_cmp ? "hides completely behind the compute stage" : "is (partially) exposed"
    @printf("stage budget: rope-only %.2f s + compute-only %.2f s vs e2e %.2f s -> the rope stage %s\n",
            t_rope, t_cmp, t_e2e, hides)
    eng = B = batches = nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

"""
Production run: load the human index (whole matrix, fwd + rc), build the fp8
database on the GPU, stream the sample reads through the fp8 flow and
report THE number of correctly mapped records (plus rate, first-hit rank
histogram and best top-k intersection).
"""
function run_e2e_human(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2H_READS) ||
        error("reads fasta not found: $E2H_READS (generate with reads/sample.jl)")
    @info "loading the human index" E2H_INDEX_BIN
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    @info "building the fp8 database on GPU"
    t_B = @elapsed B = index_database(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    w = min(w, prevpow(2, N)) # small test indexes admit no production w
    @info "fp8 database built" dims = size(B) gib = round(length(B) / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # the 12.7 GiB host index matrix can go
    GC.gc(); CUDA.reclaim()
    res = map_and_count(re, E2H_READS, B, heads, starts, dk; ktop, w,
                        batch_size, segs, engine, progress = true)
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
    res.inter_cnt > 0 && @printf(
        "  best top-%d intersection: avg %d bases over %d mapped reads = %.2f%% of the %d-base fragment\n",
        ktop, round(Int, res.inter_sum / res.inter_cnt), res.inter_cnt,
        100 * res.inter_sum / res.inter_cnt / dk, dk)
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

# ==============================================================================
# Mode dispatch -- verbatim legacy bottom-of-file code, wrapped in main() and
# called under the old PROGRAM_FILE guard (so runtests.jl can include this
# file and call the run_* functions directly)
# ==============================================================================
function main()
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    ktop = parse(Int, getflag("k", "20"))
    w = parse(Int, getflag("w", string(2^16)))
    batch = parse(Int, getflag("batch", string(2^13)))
    segs = parse(Int, getflag("segs", "8"))
    engine = Symbol(getflag("engine", "lt"))
    reps = parse(Int, getflag("reps", "1"))
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2e_test(; ktop, w, batch_size = batch, segs, engine)
    mode == "bench" && run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps)
    mode == "run" && run_e2e_human(; ktop, w, batch_size = batch, segs, engine)
    mode == "all" && (run_e2e_test(; ktop, w, batch_size = batch, segs, engine);
                      run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps);
                      run_e2e_human(; ktop, w, batch_size = batch, segs, engine))
    mode in ("test", "bench", "run", "all") ||
        error("unknown mode $mode (use test|bench|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
