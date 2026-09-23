# ==============================================================================
# selftest.jl -- single-script, hardware-aware SELF-TEST for the whole
# RotorMap workbench: random DNA creation, rope index construction, index
# sampling (with mutations) and end-to-end mapping, with the GEMM numeric
# path chosen from the GPU's compute capability (fp8 when the hardware
# supports it, fp16 otherwise).
#
# Usage (GPU host, from the repo root):
#
#   julia --project=RotorMap -t 8 RotorMap/selftest.jl
#
# `-t 8` (or more) threads: the CPU stages (mutate, sampling) and the GPU
# encode/gemm pipelines are threaded.  Takes ~1-2 minutes after startup
# (one-time CUDA.jl JIT dominates); fixtures are cached under
# NEWFASTA_V3_DIR (default /tmp) and shared with runtests.jl.
#
# Stages
#   A. random DNA creation (CPU): generate_reference -> save_fasta ->
#      independent re-parse, generate_reads exact windows at err = 0,
#      mutate (err = 0 no-op, err > 0 length-preserving, seeded).
#   B. rope index creation (GPU): the PRODUCTION encode kernel
#      (encode_frag_real_batch!) turns the tiny synthetic reference into an
#      index .bin (fwd + revcomp windows), which load_index re-reads and
#      structurally validates.
#   C. index sampling (CPU): the production sampler
#      (sample_reads, :even draw over the N-free window pool) with
#      provenance headers; err = 0 reads must bitwise equal their claimed
#      windows (two independent fasta parsers agree) and the sampler must be
#      deterministic.  Mutated (err > 0) sampling happens in stage D.
#   D. mapping (GPU): the full load -> resident database -> rope stream ->
#      GEMM + top-k -> provenance-score path: (1) the built-in err = 0
#      guarantee (1000 exact windows, ALL must map), (2) mutated reads
#      (err = SELFTEST_ERR) whose top-k hits must hit each read's true locus
#      at >= SELFTEST_MUTATED_MIN_RATE.
#
# Hardware detection (stage 0)
#   CUDA must be functional (the index + mapping layers are GPU-only).  The
#   numeric GEMM path is picked from the device's compute capability BEFORE
#   any layer is included -- search/engine_fp8.jl and search/engine_fp16.jl
#   are NOT co-includable (RotorMap/README.md's one-variant rule):
#     cc >= 8.9 (Ada RTX 40xx), 9.0 (Hopper) or 12.x (Blackwell RTX 50xx)
#       -> the fp8 flow (e4m3 database, cuBLASLt `:lt` engine; the
#          production path on the RTX 5090);
#     anything older -> the fp16 cuBLAS flow (engine_fp16.jl).
#   The :lt engine additionally needs a cuBLASLt new enough for fp8 (the
#   CUDA runtime pinned in RotorMap/LocalPreferences.toml is fine); the
#   alternative :mma engine would also need nvcc (see gemm/fp8_ptx.jl).
#
# Exit code: 0 = every stage passed; any failed @assert aborts nonzero.
#
# Tunables (consts below or env):
#   NEWFASTA_V3_DIR              fixture cache dir      (default /tmp)
#   SELFTEST_ERR                 stage-D mutation rate  (default 0.05)
#   SELFTEST_MUTATED_MIN_RATE    stage-D pass floor     (default 0.90)
# ==============================================================================

using CUDA
using Printf
using Random: Xoshiro

# --- tunables -----------------------------------------------------------------
const SELFTEST_ERR = parse(Float64, get(ENV, "SELFTEST_ERR", "0.05"))
const SELFTEST_MUTATED_MIN_RATE =
    parse(Float64, get(ENV, "SELFTEST_MUTATED_MIN_RATE", "0.90"))

# ==============================================================================
# stage 0: hardware detection -> numeric path choice
# ==============================================================================

# e4m3 fp8 tensor cores exist on sm_89 (Ada), sm_90 (Hopper) and sm_120+
# (Blackwell).  Everything older takes the fp16 cuBLAS path.  The branch must
# happen BEFORE the includes: the fp8 and fp16 engines are not co-includable.
const _SELFTEST_FP8 = let
    CUDA.functional() ||
        error("no functional CUDA device -- stages B-D (index + mapping) " *
              "are GPU-only; stages A/C alone are not what this script is for")
    cap = CUDA.capability(device())
    cap.major > 8 || (cap.major == 8 && cap.minor >= 9)
end

function _print_hardware()
    dev = device()
    cap = CUDA.capability(dev)
    path = _SELFTEST_FP8 ? "fp8 (e4m3 database, cuBLASLt :lt engine)" :
                           "fp16 (cuBLAS fp16 engine)"
    @printf("hardware: %s, %.1f GiB, compute capability %d.%d\n",
            CUDA.name(dev), CUDA.totalmem(dev) / 2^30, cap.major, cap.minor)
    @printf("numeric path: %s\n", path)
end

# --- the matching layered stack (canonical include order is owned by the
#     experiment files; both provide run_e2e_test and the _e2h_* fixtures) ---
inc(p...) = include(joinpath(@__DIR__, p...))
if _SELFTEST_FP8
    inc("experiments", "e2e_fp8.jl")  # full stack + harness.jl + fp8 TopKEngine
else
    inc("experiments", "e2e_fp16.jl") # full stack + local fp16 fixtures
end
inc("reads", "sample.jl") # CPU-only production sampler (sample_reads); its
                          # re-includes of common/* are no-ops (identical
                          # redefinitions are legal on julia >= 1.12)

# ==============================================================================
# stage A: random DNA creation (CPU)
# ==============================================================================
function selftest_random_dna()
    N = 50_000
    ref = generate_reference(N; seed = 1)
    @assert length(ref) == N && all(c -> 0 <= c <= 3, ref)

    # fasta roundtrip through an INDEPENDENT parser (the harness's scanner)
    path = joinpath(mktempdir(), "selftest_random_ref.fasta")
    save_fasta([ref], path; heads = [">chrT selftest random dna"])
    raw = read(path)
    (heads, lo, hi, nchar) = _e2h_ref_table(raw)
    @assert length(heads) == 1 && heads[1] == ">chrT selftest random dna"
    @assert nchar[1] == N
    @assert _e2h_span_codes(raw, lo[1], hi[1]) == ref "fasta roundtrip changed the sequence"

    # generate_reads at err = 0: reads are EXACT windows, pos are 0-based starts
    k = 1_000
    reads, pos = generate_reads(ref, k, 64; err = 0.0, seed = 2)
    @assert length(reads) == 64 && all(p -> 0 <= p <= N - k, pos)
    @assert all(i -> reads[i] == ref[pos[i]+1:pos[i]+k], 1:64) "err = 0 reads are not exact windows"

    # mutate: err = 0 is a no-op; err > 0 keeps the length, changes bases,
    # and is deterministic in its rng
    @assert mutate(ref, 0.0; rng = Xoshiro(3)) == ref
    m1 = mutate(ref, 0.05; rng = Xoshiro(4))
    m2 = mutate(ref, 0.05; rng = Xoshiro(4))
    m3 = mutate(ref, 0.05; rng = Xoshiro(5))
    @assert length(m1) == N && m1 != ref
    @assert m1 == m2 && m1 != m3 "mutate is not deterministic in rng"
    println("  A. random DNA creation (generate_reference / generate_reads / mutate / fasta) OK")
    return ref
end

# ==============================================================================
# stage B: rope index creation (GPU, production encode kernel)
# ==============================================================================
function selftest_index(; kfrag::Int = 20_000)
    ref = _ensure_e2h_test_ref() # cached 1 Mbp two-record fixture
    t = @elapsed binfile = _e2h_build_index(ref; k = kfrag)
    isfile(binfile) || error("index build produced no .bin: $binfile")
    d = load_index(binfile) # structural asserts + normalize = 0 check inside
    @assert d.k == kfrag && d.n_frag > 0
    @assert d.s == 8 && d.m == 4 && d.c == 4 "unexpected encoder config in the index"
    @assert length(d.heads) == 2 * d.n_frag "location meta does not cover fwd + rc"
    @assert length(d.starts) == 2 * d.n_frag
    @printf("  B. rope index creation OK: %d windows (fwd + rc) -> %s (%.1f s, cached fixtures skip this)\n",
            d.n_frag, binfile, t)
    return (ref = ref, binfile = binfile)
end

# ==============================================================================
# stage C: index sampling with provenance (CPU, the production sample flow)
# ==============================================================================
function selftest_sampling(; n::Int = 32, k::Int = 20_000)
    # a CLEAN synthetic reference: the sampler's default N-free pool needs
    # k-windows without N/IUPAC/junk, and the shared tiny fixture (stage B/D)
    # deliberately sprinkles IUPAC junk into every k-window
    ref = joinpath(mktempdir(), "selftest_clean_ref.fasta")
    save_fasta([generate_reference(200_000; seed = 6)], ref;
               heads = [">chrC selftest clean"])
    # the production sampler: :even draw (seed-free selection) over the
    # N-FREE window pool, provenance headers; at err = 0 every read must
    # bitwise equal its claimed window (two independent parsers agree)
    s = sample_reads(ref; n, k, err = 0.0, seed = 7)
    @assert length(s.reads) == n && length(s.heads) == n
    raw = read(ref)
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    bysrc = Dict(_rec_name(h) => j for (j, h) in enumerate(rheads))
    for i in 1:n
        h = parse_read_head(s.heads[i])
        @assert h !== nothing "unreadable provenance header: $(s.heads[i])"
        @assert h.len == k && haskey(bysrc, h.src)
        codes = _e2h_span_codes(raw, rlo[bysrc[h.src]], rhi[bysrc[h.src]])
        @assert s.reads[i] == codes[h.start:h.start+k-1] "err = 0 read $i != its claimed window"
    end
    # determinism: identical arguments -> bitwise identical output
    s2 = sample_reads(ref; n, k, err = 0.0, seed = 7)
    @assert s2.reads == s.reads && s2.heads == s.heads
    println("  C. index sampling (sample_reads :even, N-free pool, provenance) OK")
    return nothing
end

# ==============================================================================
# stage D: mapping (GPU, the full production path)
# ==============================================================================
function selftest_mapping(binfile::String; ktop::Int = 20,
                          n_mut::Int = 200, err::Real = SELFTEST_ERR,
                          seed::Int = 11)
    d = load_index(binfile)
    B = _SELFTEST_FP8 ? index_database(d) : index_database_f16(d)
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    w = min(_SELFTEST_FP8 ? 2^16 : 2^15, prevpow(2, N))

    # 1) the built-in err = 0 guarantee: 1000 exact windows must ALL map
    #    (reuses the stage B/C fixture builders -- cached, so this is cheap)
    res0 = run_e2e_test()
    @assert res0.correct == res0.total == 1000

    # 2) MUTATED reads: sample-flow provenance sampling at err > 0,
    #    then the production map_and_count scoring (top-k must hit the true
    #    locus); the floor is deliberately generous -- the point is to catch
    #    a broken pipeline, not to measure the error-rate curve
    readsf = _e2h_sample_reads(_E2H_TEST_REF; n = n_mut, k = d.k, err, seed)
    kw = _SELFTEST_FP8 ? (; engine = :lt) : NamedTuple()
    res = map_and_count(re, readsf, B, d.heads, d.starts, d.k;
                        ktop, w, batch_size = 2^13, segs = 8, kw...)
    rate = res.correct / res.total
    @printf("  D. mapping OK: err = 0 -> %d/%d; err = %.2f -> %d/%d (%.1f%%) top-%d hits on the true locus\n",
            res0.correct, res0.total, err, res.correct, res.total,
            100 * rate, ktop)
    @printf("     first-hit rank histogram: %s\n", res.rank_hist)
    @assert res.total == n_mut
    @assert rate >= SELFTEST_MUTATED_MIN_RATE "mutated-read mapping rate " *
        "$(round(rate; digits = 3)) below the self-test floor " *
        "$(SELFTEST_MUTATED_MIN_RATE) (tune SELFTEST_MUTATED_MIN_RATE / SELFTEST_ERR)"
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

# ==============================================================================
# main
# ==============================================================================
function main()
    println("="^78)
    println("RotorMap selftest")
    _print_hardware()
    println("="^78)
    t0 = time()
    selftest_random_dna()
    fx = selftest_index()
    selftest_sampling()
    selftest_mapping(fx.binfile)
    println("="^78)
    @printf("SELFTEST PASSED (all stages) in %.1f s -- %s path on %s\n",
            time() - t0, _SELFTEST_FP8 ? "fp8" : "fp16", CUDA.name(device()))
    return nothing
end

try
    main()
catch e
    println(stderr, "SELFTEST FAILED: ")
    showerror(stderr, e, stacktrace(catch_backtrace()))
    println(stderr)
    rethrow()
end
