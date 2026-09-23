# ==============================================================================
# e2e_cuts.jl -- END-TO-END HUMAN-GENOME MAPPING with per-COLUMN CUTS: the
# e2e_fp8 flow (the human index as the fp8 database, sample reads
# streamed through rope encoding -> fp8 GEMM) with the per-row top-k
# reduction REPLACED by a per-column threshold ("cut") gather: every inner
# product ABOVE its column's cut saves its value and location.
#
# PROBLEM
#   top-k answers "which k database windows match this read best"; a cut
#   answers "which windows match this read AT ALL" -- a per-column
#   significance bar instead of a per-read quota.  The cut values are COLUMN
#   DEPENDENT (one cut per reference window).  A read's result is a
#   VARIABLE-LENGTH hit list (value, global column id): 0 hits to -- in
#   pathological repetitive regions -- N hits.  Ground truth is unchanged (a
#   read is CORRECTLY MAPPED iff any of its hits' windows intersects its
#   provenance window); the rank histogram is meaningless here and is
#   replaced by hit-count statistics.
#
# DESIGN
#   1. Everything upstream of the reduction is the fp8 flow verbatim (the
#     engine keeps its double-buffered chunk loop, quantize+upload and
#     cuBLASLt GEMM; ONLY the per-chunk consumer kernel changes).
#   2. The cut gather kernel (cuts_gather_kernel!, search/engine_cuts.jl):
#     hits appended to three flat device lists with ONE global Int64 atomic
#     counter; overflow is detected EXACTLY, never truncated silently.
#   3. Batch contract: the engine computes the full padded rows_cap height;
#     pad rows are exact zeros, so cuts >= 0 never produce hits from them.
#   4. Cut vector (cut_vector, search/engine_cuts.jl): a constant (--cut=0.5)
#     or a per-column file (--cutsfile / E2CUTS_FILE env).
#   5. Scoring: the provenance logic per HIT; the report adds hit statistics
#     (hits total and rate, per-read hit-count buckets, max, zero-hit reads)
#     and the run's inner-product VALUE histogram (34 bins of 1/16 over
#     [-1, 1]) -- the calibration data for choosing cut levels.
#
# SELF-TEST (mode `test`; production files not needed; shares e2e_fp8's
#   cached 1M-base two-record fixture and err = 0 reads):
#   A. raw kernel extremes through the REAL engine (no pipeline): cut
#     -1e30 -> total == rows_cap * N EXACTLY, cut +1e30 -> 0 on the SAME
#     counter (per-batch reset witnessed); global column ids in 1:N.
#   B. full pipeline vs an exact CPU reference (production-kernel read
#     encoding -> e4m3 quantize -> float64 GEMM against the DEQUANTIZED B):
#     the per-batch hit SET must match the reference set within the
#     fp16-storage band (atol 5e-3), at BOTH cut levels (bar 0.90, cut 0.5).
#   C. map_and_count_cuts: ALL 1000 err = 0 reads correctly mapped at the
#     0.5 cut; the mapping count at the bar REPORTED (not pass/fail).
#
# ENTRY POINT
#   The cut flow machinery lives in search/engine_cuts.jl (which itself
#   requires search/engine_fp8.jl -- both are included below; engine_fp8 and
#   engine_cuts co-exist fine, only engine_fp8+engine_fp16 clash).  This file
#   keeps its own self-test helper (_e2c_encode_batch) and the run entry
#   points, verbatim from the legacy experiment.
#
# SOURCE: legacy/test/e2ecuts.jl (run_e2c_test / run_e2c_human + dispatch +
#         the _e2c_encode_batch self-test helper).  E2C_MAXHITS,
#         E2C_CUTS_FILE_ENV and cut_vector live in search/engine_cuts.jl.
#         Full measured RESULTS (the similarity landscape, the production cut
#         sweep) live in that file's header.
#
# INCLUDES (canonical order)
#   common/{treearrays,dna,util,testref}.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl,
#   search/engine_fp8.jl, search/engine_cuts.jl, index/{build,load}.jl,
#   reads/provenance.jl, common/harness.jl
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (kernel extremes + CPU-reference hit sets + all mapped)
#   run   production: human index + n131072_e0.05 reads -> cut mapping score
#   all   test + run (default)
#   flags: --cut=0.95 (run mode's constant cut level), --cutsfile= (per-column
#          cut file, beats --cut; E2CUTS_FILE env), --maxhits=67108864,
#          --w=65536, --batch=8192, --engine=lt
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/e2e_cuts.jl test
#   julia --project=. -t 16 RotorMap/experiments/e2e_cuts.jl run --cut=0.95
#   julia --project=. -t 16 RotorMap/experiments/e2e_cuts.jl run --cutsfile=cuts_3M.f32
#   julia --project=. -t 16 RotorMap/experiments/e2e_cuts.jl    # test + run
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
inc("search", "engine_cuts.jl") # requires engine_fp8 above; provides cut_vector,
#                                 E2C_MAXHITS, E2C_CUTS_FILE_ENV ("E2CUTS_FILE")
inc("index", "build.jl")
inc("index", "load.jl")
inc("reads", "provenance.jl")
inc("common", "harness.jl") # last: CALLS index/encode at runtime

using Random
using CUDA
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Self-test helpers
# ------------------------------------------------------------------------------

# Host reference encoding of a batch of reads (packed words -> the PRODUCTION
# kernel -> fp16 rows), for the exact CPU-reference hit-set check.
function _e2c_encode_batch(re::RopeEncoder, words::Vector{Vector{UInt32}};
                           normalize::Int = 0)
    nb = length(words)
    W = cld(re.k, 16)
    rdim = 2 * re.m * 4^re.c
    fw = Vector{UInt32}(undef, W * nb)
    for i in 1:nb
        @assert length(words[i]) == W "packed fragment width mismatch"
        copyto!(fw, (i - 1) * W + 1, words[i], 1, W)
    end
    dest = CUDA.zeros(Float16, nb, rdim)
    dn = CUDA.zeros(Float32, re.m, nb)
    encode_frag_real_batch!(dest, dn, re, cu(fw); normalize)
    emb = Array(dest)
    dest = dn = fw = nothing
    GC.gc(); CUDA.reclaim()
    return emb
end

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end: build the 1M reference (once, e2ehuman's fixture), index
it, sample 1000 err = 0 reads, then (A) check the gather kernel's exact
counts through the real engine at cut extremes, (B) run the full pipeline
at BOTH cut levels -- `bar` (the production operating point, 0.90) and
`cut` (0.5) -- and require the hit SETS to match an exact CPU reference
(quantization modelled, fp16-storage band), and (C) require every read
correctly mapped at `cut` and report the mapping count at `bar` (a data
property of the fixture, not a pass/fail).
"""
function run_e2c_test(; cut::Real = 0.5, bar::Real = 0.90,
                      maxhits::Int = E2C_MAXHITS,
                      kfrag::Int = 20_000, n::Int = E2H_TEST_N, w::Int = 2^16,
                      batch_size::Int = 2^13, engine::Symbol = :lt,
                      seed::Int = E2H_TEST_SEED)
    cut > 0 && bar > 0 ||
        throw(ArgumentError("the self-test needs positive cut levels (got cut = $cut, " *
                            "bar = $bar): the exact-count checks rely on the engine's " *
                            "zeroed pad rows not matching"))
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    B = index_database(d)
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    rdim = size(B, 1)
    w_test = min(w, prevpow(2, N)) # the tiny N (≈10^3) admits no production w

    # ---- A. raw kernel extremes through the real engine (no pipeline) ------
    eng = TopKEngine(B; k = 1, w = w_test, rows_cap = n, engine)
    hv = CuVector{Float32}(undef, eng.rows_cap * N)
    hl = CuVector{Int32}(undef, eng.rows_cap * N)
    hr = CuVector{Int32}(undef, eng.rows_cap * N)
    cnt = CuVector{Int64}(undef, 1)
    hist_d = CuVector{Int64}(undef, 34) # (also exercises the histogram path)
    cuts_d = CuVector{Float32}(undef, N)
    copyto!(cuts_d, fill(-1.0f30, N)) # every finite fp16 value passes
    emb = randn(Float16, eng.rows_cap, rdim)
    CUDA.stream!(eng.sg) do
        upload_fp16_as_f8!(eng, emb)
    end
    t_all = batch_gemm_cuts!(hv, hl, hr, cnt, eng, cuts_d, hist_d)
    @assert t_all == eng.rows_cap * N "all-match count $(t_all) != rows_cap*N = " *
                                      "$(eng.rows_cap * N): elements missed or the " *
                                      "counter not reset"
    h_all = Array(hist_d)
    @assert sum(h_all) == eng.rows_cap * N "histogram total $(sum(h_all)) != rows_cap*N: " *
                                           "histogram misses elements"
    l_all = Array(hl)
    @assert extrema(l_all) == (1, N) "global column ids out of range: $(extrema(l_all))"
    copyto!(cuts_d, fill(1.0f30, N)) # nothing passes
    t_none = batch_gemm_cuts!(hv, hl, hr, cnt, eng, cuts_d, hist_d) # SAME cnt: reset witnessed
    @assert t_none == 0 "no-match count $t_none != 0: stale counter or pad rows not zero"
    @info "A. raw kernel extremes OK" rows_cap = eng.rows_cap N all_matches = t_all no_matches = t_none
    eng = hv = hl = hr = cnt = hist_d = cuts_d = emb = nothing
    GC.gc(); CUDA.reclaim()

    # ---- B. full pipeline vs the exact CPU reference (hit SETS), at BOTH ----
    # cut levels: `cut` (the all-mapped bar) and `bar` (the production default
    # 0.95 -- the correctness witness at the real operating point).  The
    # fixture only guarantees >= 95% base sharing, i.e. inner products down to
    # ~0.9, so ALL-mapped needs a cut below that floor (only 537/1000 err = 0
    # reads clear 0.95) -- at 0.95 the CPU-reference check IS the test.
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(readsf; k = re.k, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == n "unexpected read count"
    Bdeq = Float64.(Float32.(Array(B)))  # the database AS STORED (dequantized)
    for (lbl, lvl) in (("cut", cut), ("bar", bar))
        cuts = fill(Float32(lvl), N)
        atol = 5e-3                      # fp16 C storage + accumulation order
        nb_seen = 0
        maxv = -Inf
        err = Ref{Any}(nothing)
        tks = Ref{Vector{Task}}(Task[])
        for bat in rope_cuts_stream(re, readsf, B, cuts; maxhits, w = w_test,
                                    batch_size = cld(n, 3), # 3 batches incl. ragged tail
                                    engine, err_out = err, tasks_out = tks)
            nb = length(bat.heads)
            @assert bat.first == nb_seen + 1 "batch first index mismatch"
            nb_seen += nb
            @assert bat.total == length(bat.vals) "batch overflow"
            emb_ref = _e2c_encode_batch(re, [refw[h] for h in bat.heads])
            Cref = Float64.(Float32.(F8.(emb_ref))) * Bdeq # (nb, N) exact model
            got = Dict{Tuple{Int,Int},Float32}()
            for p in eachindex(bat.vals)
                r = Int(bat.rows[p])
                c = Int(bat.locs[p])
                @assert 1 <= r <= nb "hit row $r beyond the batch (pad leak)"
                got[(r, c)] = bat.vals[p]
            end
            for ((r, c), v) in got
                @assert Cref[r, c] > lvl - atol "hit below the cut: row $r col $c val $v ref $(Cref[r, c])"
                @assert abs(v - Cref[r, c]) <= atol "value mismatch: $v vs ref $(Cref[r, c]) (row $r col $c)"
            end
            nmiss = count(Cref[r, c] > lvl + atol && !haskey(got, (r, c))
                          for c in 1:N, r in 1:nb)
            @assert nmiss == 0 "$nmiss reference hits missing from the batch (cut = $lvl, atol = $atol)"
            maxv = max(maxv, maximum(Cref))
            @printf("  batch: %d reads, %d hits (%.2f/read) -- matches the CPU reference\n",
                    nb, length(bat.vals), length(bat.vals) / nb)
        end
        foreach(wait, tks[])
        err[] === nothing || error("flow failed: $(err[])")
        @assert nb_seen == n "processed $nb_seen != $n reads"
        @info "B. hit sets match the CPU reference OK ($lbl = $lvl)" atol max_reference_value =
            round(maxv; digits = 4)
    end

    # ---- C. the provenance score: ALL-mapped asserted at `cut`, the mapping
    # count at the production bar REPORTED (a data property, not a pass/fail)
    res = map_and_count_cuts(re, readsf, B, d.heads, d.starts, d.k,
                             fill(Float32(cut), N);
                             maxhits, w = w_test, batch_size, engine)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped at cut $cut"
    @info "TINY E2C PASSED: $n/$n correctly mapped at cut $cut" hits = res.hits hitmax =
        res.hitmax zerohits = res.zerohits buckets = res.buckets
    res_bar = map_and_count_cuts(re, readsf, B, d.heads, d.starts, d.k,
                                 fill(Float32(bar), N);
                                 maxhits, w = w_test, batch_size, engine)
    @info "TINY E2C at the bar: $(res_bar.correct)/$n correctly mapped at cut $bar" hits =
        res_bar.hits zerohits = res_bar.zerohits buckets = res_bar.buckets
    B = d = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

"""
Production run: load the human index (whole matrix, fwd + rc), build the fp8
database on the GPU, build the cut vector (constant `cut` or `cutsfile`),
stream the sample reads through the cut flow and report the number of
correctly mapped records plus the hit statistics (total, rate, per-read
buckets).
"""
function run_e2c_human(; cut::Real = 0.95, cutsfile::Union{String,Nothing} = nothing,
                       maxhits::Int = E2C_MAXHITS, w::Int = 2^16,
                       batch_size::Int = 2^13, engine::Symbol = :lt)
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
    cuts = cut_vector(N; cut, path = cutsfile)
    @info "cut vector" source = (cutsfile === nothing ? "constant $cut" : cutsfile) N min =
        minimum(cuts) max = maximum(cuts)
    @info "fp8 database built" dims = size(B) gib = round(length(B) / 2^30; digits = 2) seconds =
        round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # the 12.7 GiB host index matrix can go
    GC.gc(); CUDA.reclaim()
    res = map_and_count_cuts(re, E2H_READS, B, heads, starts, dk, cuts;
                             maxhits, w, batch_size, engine, progress = true)
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  %d hits over %d x %d = %.3e comparisons (%.3g hits per Ghit); per read: mean %.1f, max %d, zero-hit %d\n",
            res.hits, res.total, N, Float64(res.total) * N,
            res.hits / (Float64(res.total) * N) * 1e9, res.hits / res.total,
            res.hitmax, res.zerohits)
    @printf("  hits/read buckets [0, 1, 2-10, 11-100, 101-1000, >1000]: %s\n",
            res.buckets)
    @printf("  similarity histogram (bins of 1/16 over [-1,1]; first <-1, last >+1):\n    %s\n",
            res.hist)
    B = nothing
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
    cut = parse(Float64, getflag("cut", "0.95"))
    cutsfile = let s = getflag("cutsfile", get(ENV, E2C_CUTS_FILE_ENV, ""))
        isempty(s) ? nothing : s
    end
    maxhits = parse(Int, getflag("maxhits", string(E2C_MAXHITS)))
    w = parse(Int, getflag("w", string(2^16)))
    batch = parse(Int, getflag("batch", string(2^13)))
    engine = Symbol(getflag("engine", "lt"))
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2c_test(; maxhits, w, batch_size = batch, engine)
    mode == "run" && run_e2c_human(; cut, cutsfile, maxhits, w, batch_size = batch, engine)
    mode == "all" && (run_e2c_test(; maxhits, w, batch_size = batch, engine);
                      run_e2c_human(; cut, cutsfile, maxhits, w, batch_size = batch, engine))
    mode in ("test", "run", "all") || error("unknown mode $mode (use test|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
