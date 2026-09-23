# ==============================================================================
# e2e_compact.jl -- END-TO-END MAPPING with a RAM-RESIDENT index: the e2e_fp8
# flow for databases TOO BIG FOR VRAM, with AUTO VRAM RESIDENCY when the
# index fits (see below -- a fitting index runs the resident engine and pays
# NO per-batch sweep, i.e. e2e_fp8 speed).  In the streaming path the whole
# fp8 database B stays in HOST memory (plain Julia Matrix{F8}, no pinning);
# every read batch sweeps it
# through the PCIe link once, `w` columns at a time, into a TRIPLE-BUFFERED
# device ring whose slots feed the same double-buffered fp8 GEMM + seg-top-k
# pipeline as the resident engine (the compact engine's batch loop with one
# extra stage bolted in front).
#
# PIPELINE (per chunk i of a batch sweep)
#   h2d stream : wait ring slot b free -> cudaMemcpyAsync the contiguous host
#                column range into slot b -> record ev_h2d[b]
#   sg stream  : wait ev_h2d[b] (+ C buffer b2's previous top-k) -> fp8 GEMM
#                D = A * B_chunk -> record evg[b2] AND ev_dev[b]
#   st stream  : wait evg[b2] -> seg_rowtopk_merge into the running top-k ->
#                record evt[b2]
# All waits are GPU-side (events); the host loop only enqueues.  Ring depth 3.
# B's leading dimension is rdim both for the resident parent and for each
# (rdim, w) ring slot, so the engine dispatch is untouched.
#
# WHY NO PINNED MEMORY (measured on kau, RTX 5090 + Gen5 x16): RLIMIT_MEMLOCK
# is 8 MiB soft AND hard here, so cudaHostRegister/cudaHostAlloc beyond a few
# MiB fails.  Pageable H2D of 128 MiB chunks measures ~20 GB/s; at the
# default batch the sweep is mildly H2D-bound, at --batch=2^15 it is
# compute-bound -- WITH OR WITHOUT pinning, i.e. the 8 MiB memlock cap costs
# nothing that matters.  Results are bitwise what the resident engine would
# produce; only the wall time carries the extra H2D.  Total H2D per read
# batch = the whole database once, so larger --batch amortizes it linearly.
#
# AUTO RESIDENCY (the design contract): the ring exists ONLY for databases
# too big for VRAM.  When the host database plus the engine's working set
# fits in the device memory free right now (`resident_fits`), the flow
# uploads B ONCE and delegates to the resident engine (`topk_flow`,
# engine_fp8.jl) instead of sweeping: bitwise the same results at e2e_fp8's
# speed -- a fitting index must not pay the per-batch PCIe sweep.
# --resident=off forces the ring, =on forces the upload (fails fast when it
# cannot fit); the decision is logged either way.  The decision rule is
# `sizeof(Bh)` + the engine's whole working set (staging + C chunk buffers
# + 96 MiB temps/slack) <= free VRAM after a pool reclaim -- conservative:
# a wrong guess must only ever cost speed (the ring), never an OOM.  The
# exact formula + a worked example live on `resident_fits`'s docstring;
# MEASURED head-to-head numbers live in the RESULTS block of
# search/engine_compact.jl's header.
#
# SCORING is the harness map_and_count verbatim (same provenance parsing,
# same best-of-top-k intersection metric) with the batch producer swapped:
# the `stream` keyword it grew makes rope_topk_stream replaceable.
#
# ENTRY POINT
#   The compact engine (CompactEngine, batch_gemm_topk_compact!,
#   topk_flow_compact, rope_topk_stream_compact, index_database_host) lives
#   in search/engine_compact.jl.  This file keeps the env path consts and the
#   run entry points, verbatim from the legacy experiment.
#
# SOURCE: legacy/test/e2ecompact.jl (env consts E2C_INDEX/E2C_READS,
#         run_e2e_compact_test / run_e2e_compact + dispatch).  Full measured
#         RESULTS (resident-vs-streamed head-to-heads, eval-read provenance
#         geometry note) live in that file's header.
#
# INCLUDES (canonical order)
#   common/{treearrays,dna,util,testref}.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl,
#   search/engine_fp8.jl, search/engine_compact.jl, index/{build,load}.jl,
#   reads/provenance.jl, common/harness.jl
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (the 1M reference fixture -> index -> err = 0
#         reads -> the COMPACT chunked path -> all mapped)
#   run   production: the maize B73 index (default) + n131072_k4000_e0.3
#         reads (default) -> mapping score
#   all   test + run (default)
#   flags: --k=20, --w=65536 (B-column chunk width = ring slot width),
#          --batch=8192, --segs=8, --engine=lt (:mma needs w | N),
#          --resident=auto|on|off (auto = the resident engine when the index
#          + working set fits in free VRAM; on/off force either side; the
#          `test` mode default runs BOTH paths and cross-checks them)
#   env:   E2COMPACT_INDEX (default: the maize kstep-250 s5m1c5 fp8 index,
#          33 GiB -- does not fit 32 GiB VRAM, the reason this script exists)
#          E2COMPACT_READS (default: the sample n131072_k4000_e0.3_s42
#          batch; point it at e.g. maksym's maize_1X_15.fasta for the eval)
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/e2e_compact.jl test
#   julia --project=. -t 16 RotorMap/experiments/e2e_compact.jl run --engine=lt
#   julia --project=. -t 16 RotorMap/experiments/e2e_compact.jl run --batch=32768
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
inc("search", "engine_compact.jl") # provides CompactEngine, index_database_host,
#                                    topk_flow_compact, rope_topk_stream_compact,
#                                    resident_fits (auto-residency policy)
inc("index", "build.jl")
inc("index", "load.jl")
inc("reads", "provenance.jl")
inc("common", "harness.jl") # last: CALLS index/encode at runtime

using Random
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides): a database too big for VRAM by default
# ------------------------------------------------------------------------------
const E2C_INDEX = get(ENV, "E2COMPACT_INDEX",
                      "/share/q4bio/dandan/rotormap/data/" *
                      "GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_primary12." *
                      "indexreal_fp8_k4000_s5m1c5_kstep250.bin")
const E2C_READS = get(ENV, "E2COMPACT_READS",
                      "/share/q4bio/dandan/rotormap/data/" *
                      "GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_primary12." *
                      "sample_n131072_k4000_e0.3_s42.fasta")

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end through the COMPACT path: e2ehuman's 1M fixture (reference ->
index -> 1000 err = 0 reads), scored by map_and_count with the chunk-streaming
producer.  At err = 0 every read must map, every first hit at rank 1 -- the
ring/event bookkeeping pinned down exactly.  `resident = nothing` (default)
runs BOTH engines on the fixture -- the streaming ring AND the auto-residency
resident delegation -- and asserts they agree (they are bitwise-equal flows
over the same bytes); `false`/`true` run just one of them.  The fixture is
too small to auto-fit anything, so the cross-check is always explicit.
"""
function run_e2e_compact_test(; kfrag::Int = 20_000, n::Int = E2H_TEST_N, ktop::Int = 20,
                              w::Int = 2^16, batch_size::Int = 2^13, segs::Int = 8,
                              engine::Symbol = :lt, seed::Int = E2H_TEST_SEED,
                              resident::Union{Nothing,Bool} = nothing)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    Bh = index_database_host(d) # search/engine_compact.jl
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(Bh, 2)
    w_test = min(w, prevpow(2, N)) # the tiny N (~10^3) admits no production w
    _run(r::Bool) = map_and_count(re, readsf, Bh, d.heads, d.starts, d.k; ktop,
                                  w = w_test, batch_size, segs, engine,
                                  stream = (re_, f, B; kw...) ->
                                      rope_topk_stream_compact(re_, f, B; kw..., resident = r))
    if resident === nothing
        ring = _run(false) # the streaming ring (this script's raison d'être)
        resi = _run(true)  # the auto-residency resident delegation
        @assert resi.total == n "processed $(resi.total) != $n reads"
        @assert resi.correct == n "err = 0 sanity FAILED: only $(resi.correct)/$n " *
                                  "correctly mapped"
        @assert resi.rank_hist == ring.rank_hist "resident delegation diverged " *
                                                 "from the streaming ring"
        @info "TINY COMPACT E2E PASSED: $n/$n correctly mapped, ring == resident" first_hit_ranks =
            resi.rank_hist
        GC.gc(); CUDA.reclaim()
        return resi
    end
    res = _run(resident)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped"
    @info "TINY COMPACT E2E PASSED ($(resident ? "resident" : "ring") path): " *
          "$n/$n correctly mapped" first_hit_ranks = res.rank_hist
    GC.gc(); CUDA.reclaim()
    return res
end

"""
Production run: load the index into HOST RAM first; when the whole database
plus the engine's working set fits in free VRAM (`resident_fits`, or forced
with `resident = true`/`false`), delegate to the resident engine -- one
time upload, NO per-batch PCIe sweep, i.e. e2e_fp8 speed; otherwise stream
it chunkwise through the device ring per read batch.  Either way, report
the mapping score (same metric as e2ehuman).  The reads' headers follow
either provenance format (parse_read_head).
"""
function run_e2e_compact(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                         segs::Int = 8, engine::Symbol = :lt,
                         resident::Union{Nothing,Bool} = nothing)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2C_READS) ||
        error("reads fasta not found: $E2C_READS (generate with reads/sample.jl " *
              "or set E2COMPACT_READS)")
    @info "loading the index (host RAM first; VRAM-resident when it fits)" E2C_INDEX
    t_load = @elapsed d = load_index(E2C_INDEX)
    @info "building the host fp8 database"
    t_B = @elapsed Bh = index_database_host(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    w = min(w, prevpow(2, size(Bh, 2))) # small test indexes admit no production chunk width
    @info "host fp8 database ready" dims = size(Bh) gib = round(length(Bh) / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # keep heads/starts/Bh; the loader NamedTuple can go
    GC.gc()
    # one residency decision for the whole run (the same policy the stream
    # applies under `resident = nothing`) -- picks the report wording too
    go_resident = resident !== nothing ? resident :
                  resident_fits(Bh; rows_cap = batch_size, rdim = size(Bh, 1), w)
    res = map_and_count(re, E2C_READS, Bh, heads, starts, dk; ktop, w,
                        batch_size, segs, engine, progress = true,
                        stream = (re_, f, B; kw...) ->
                            rope_topk_stream_compact(re_, f, B; kw..., resident = go_resident))
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
    res.inter_cnt > 0 && @printf(
        "  best top-%d intersection: avg %d bases over %d mapped reads = %.2f%% of the %d-base fragment\n",
        ktop, round(Int, res.inter_sum / res.inter_cnt), res.inter_cnt,
        100 * res.inter_sum / res.inter_cnt / dk, dk)
    if go_resident
        @printf("  resident path: %.2f GiB uploaded to VRAM once -- no per-batch sweep\n",
                length(Bh) / 2^30)
    else
        swept = res.nb * length(Bh) # every read batch sweeps the whole database
        @printf("  chunk sweep: %d batches x %.2f GiB = %.1f GiB over PCIe in %.1f s (%.1f GB/s effective)\n",
                res.nb, length(Bh) / 2^30, swept / 2^30, res.seconds, swept / res.seconds / 1e9)
    end
    Bh = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

# ==============================================================================
# Mode dispatch -- verbatim legacy bottom-of-file code, wrapped in main() and
# called under the old PROGRAM_FILE guard
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
    rstr = getflag("resident", "auto")
    resident = rstr == "auto" ? nothing :
               rstr == "on" ? true : rstr == "off" ? false :
               error("bad --resident=$rstr (use auto|on|off)")
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2e_compact_test(; ktop, w, batch_size = batch, segs, engine, resident)
    mode == "run" && run_e2e_compact(; ktop, w, batch_size = batch, segs, engine, resident)
    mode == "all" && (run_e2e_compact_test(; ktop, w, batch_size = batch, segs, engine, resident);
                      run_e2e_compact(; ktop, w, batch_size = batch, segs, engine, resident))
    mode in ("test", "run", "all") ||
        error("unknown mode $mode (use test|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
