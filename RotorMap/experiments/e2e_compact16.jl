# ==============================================================================
# e2e_compact16.jl -- END-TO-END MAPPING on the FP16 COMPACT flow: the
# e2e_compact flow (auto GPU-resident / host-RAM chunk ring) for GPUs WITHOUT
# fp8 tensor cores (cc < 8.9), run through the fp16 cuBLAS stack
# (search/engine_fp16.jl + search/engine_compact16.jl) instead of the fp8 one.
#
#   index .bin (fp8 or fp16 layout -- the tag inside dispatches; the fp8
#               layout is WIDENED to a host fp16 database on load, 2 B/elem)
#   reads fasta -> rope fp16 stream -> [auto: RESIDENT engine | host-RAM ring:
#               per-batch sweep of the whole database through a depth-3
#               device ring, w columns at a time] -> fp16 GEMM + seg top-k
#   -> provenance scoring (map_and_count: correctly mapped iff the true
#      window intersects at least one of the ktop returned db windows).
#
# The one-variant rule applies: this script includes search/engine_fp16.jl
# (+ engine_compact16.jl) and must NEVER be co-included with the fp8/complex
# stacks in one session.
#
# SELF-TEST (mode `test`; production files not needed): e2e_compact's tiny
#   fixture (1M reference -> index -> 1000 err = 0 reads) run through BOTH
#   paths -- the streaming ring AND the auto-residency resident delegation --
#   asserting identical rank histograms (they are bitwise-equal flows over
#   the same bytes; the fixture is too small to auto-fit, so the ring is
#   always explicitly exercised).
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (both paths, cross-checked)
#   run   production mapping (defaults: the human k20000 fp8-layout index,
#         widened to a 12.6 GiB host fp16 database, + the n1048576_k20000_e0.1
#         reads)
#   all   test + run (default)
#   flags: --k=20 (top-k size), --w=32768 (chunk/ring-slot width; the fp16
#          optimum), --batch=8192, --segs=8,
#          --resident=auto|on|off (auto = the resident engine when the fp16
#          database + working set fits free VRAM; on/off force either side)
#   env:   E2COMPACT16_INDEX / E2COMPACT16_READS
#
# USAGE
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_compact16.jl test
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_compact16.jl run
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_compact16.jl run --resident=off --batch=32768
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

# --- layers (canonical order; the FP16 variant of e2e_compact.jl's list) ------
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
inc("gemm", "fp8_convert.jl") # F8 + host conversions only (harness fixture
#                               builder + index/load asserts); NO fp8 engines
inc("gemm", "fp16.jl")
inc("search", "engine_fp16.jl")
inc("search", "engine_compact16.jl") # provides index_database_host_f16,
#                                      CompactEngine16, topk_flow_compact16,
#                                      rope_topk_stream_compact16,
#                                      resident_fits16 (auto-residency policy)
inc("index", "build.jl")
inc("index", "load.jl")
inc("reads", "provenance.jl")
inc("common", "harness.jl") # last: CALLS index/encode at runtime

using Random
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides)
# ------------------------------------------------------------------------------
const E2C16_INDEX = get(ENV, "E2COMPACT16_INDEX",
                        "/share/q4bio/dandan/rotormap/data/" *
                        "GCF_000001405.40_GRCh38.p14_primary25.indexreal_fp8.bin")
const E2C16_READS = get(ENV, "E2COMPACT16_READS",
                        "/share/q4bio/dandan/rotormap/data/" *
                        "GCF_000001405.40_GRCh38.p14_primary25." *
                        "sample_n1048576_k20000_e0.1_s42.fasta")

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end through the FP16 COMPACT path: the shared 1M fixture
(reference -> index -> 1000 err = 0 reads) widened to a host fp16 database,
scored by map_and_count with the chunk-streaming producer.  At err = 0 every
read must map, every first hit at rank 1.  `resident = nothing` (default)
runs BOTH engines on the fixture -- the streaming ring AND the auto-residency
resident delegation -- and asserts they agree (bitwise-equal flows over the
same bytes); the fixture is too small to auto-fit anything, so the ring is
always explicitly exercised.
"""
function run_e2e_compact16_test(; kfrag::Int = 20_000, n::Int = E2H_TEST_N,
                                ktop::Int = 20, w::Int = 2^15,
                                batch_size::Int = 2^13, segs::Int = 8,
                                seed::Int = E2H_TEST_SEED,
                                resident::Union{Nothing,Bool} = nothing)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    Bh = index_database_host_f16(d) # search/engine_compact16.jl
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    w_test = min(w, prevpow(2, size(Bh, 2)))
    _run(r) = map_and_count(re, readsf, Bh, d.heads, d.starts, d.k; ktop,
                            w = w_test, batch_size, segs, progress = false,
                            stream = (re_, f, B; kw...) ->
                                rope_topk_stream_compact16(re_, f, B; kw..., resident = r))
    if resident === nothing
        ring = _run(false) # the streaming ring
        resi = _run(true)  # the auto-residency resident delegation
        @assert resi.total == ring.total == n
        @assert resi.correct == ring.correct == n "err = 0 sanity FAILED: only " *
            "$(ring.correct)/$n (ring) / $(resi.correct)/$n (resident) correctly mapped"
        @assert resi.rank_hist == ring.rank_hist "resident delegation diverged " *
            "from the ring: $(ring.rank_hist) vs $(resi.rank_hist)"
        @info "TINY COMPACT16 E2E PASSED: $n/$n correctly mapped, ring == resident" first_hit_ranks =
            ring.rank_hist
        Bh = nothing
        GC.gc(); CUDA.reclaim()
        return ring
    end
    res = _run(resident)
    @assert res.total == n && res.correct == n "err = 0 sanity FAILED: only " *
        "$(res.correct)/$n correctly mapped"
    @info "TINY COMPACT16 E2E PASSED ($(resident ? "resident" : "ring") path): " *
          "$n/$n correctly mapped" first_hit_ranks = res.rank_hist
    Bh = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

"""
Production mapping on the FP16 COMPACT flow: load the index into a host fp16
database (`index_database_host_f16`), decide the residency once (`resident =
nothing` -> auto via `resident_fits16`, or forced), and stream the reads
through the rope -> GEMM + top-k -> provenance-score path.
"""
function run_e2e_compact16(; ktop::Int = 20, w::Int = 2^15, batch_size::Int = 2^13,
                           segs::Int = 8, resident::Union{Nothing,Bool} = nothing)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2C16_READS) ||
        error("reads fasta not found: $E2C16_READS (generate with reads/sample.jl " *
              "or set E2COMPACT16_READS)")
    @info "loading the index (host fp16 database; VRAM-resident when it fits)" E2C16_INDEX
    t_load = @elapsed d = load_index(E2C16_INDEX)
    @info "building the host fp16 database"
    t_B = @elapsed Bh = index_database_host_f16(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    w = min(w, prevpow(2, size(Bh, 2))) # small test indexes admit no production chunk width
    @info "host fp16 database ready" dims = size(Bh) gib = round(length(Bh) * 2 / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # keep heads/starts/Bh; the loader NamedTuple can go
    GC.gc()
    # one residency decision for the whole run (the same policy the stream
    # applies under `resident = nothing`) -- picks the report wording too
    go_resident = resident !== nothing ? resident :
                  resident_fits16(Bh; rows_cap = batch_size, w)
    res = map_and_count(re, E2C16_READS, Bh, heads, starts, dk; ktop, w,
                        batch_size, segs, progress = true,
                        stream = (re_, f, B; kw...) ->
                            rope_topk_stream_compact16(re_, f, B; kw..., resident = go_resident))
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
    res.inter_cnt > 0 && @printf(
        "  best top-%d intersection: avg %d bases over %d mapped reads = %.2f%% of the %d-base fragment\n",
        ktop, round(Int, res.inter_sum / res.inter_cnt), res.inter_cnt,
        100 * res.inter_sum / res.inter_cnt / dk, dk)
    if go_resident
        @printf("  resident path: the whole fp16 database uploaded to VRAM once -- no per-batch sweep\n")
    else
        @printf("  ring path: every batch swept the whole database through the device ring\n")
    end
    Bh = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

# ==============================================================================
# Mode dispatch
# ==============================================================================
function main()
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    ktop = parse(Int, getflag("k", "20"))
    w = parse(Int, getflag("w", string(2^15)))
    batch = parse(Int, getflag("batch", string(2^13)))
    segs = parse(Int, getflag("segs", "8"))
    resident = let r = getflag("resident", "auto")
        r == "auto" ? nothing : r == "on" ? true : r == "off" ? false :
        error("unknown --resident=$r (use auto|on|off)")
    end
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2e_compact16_test(; ktop, w, batch_size = batch, segs, resident)
    mode == "run" && run_e2e_compact16(; ktop, w, batch_size = batch, segs, resident)
    mode == "all" && (run_e2e_compact16_test(; ktop, w, batch_size = batch, segs, resident);
                      run_e2e_compact16(; ktop, w, batch_size = batch, segs, resident))
    mode in ("test", "run", "all") ||
        error("unknown mode $mode (use test|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
