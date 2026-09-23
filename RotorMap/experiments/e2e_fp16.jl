# ==============================================================================
# e2e_fp16.jl -- END-TO-END HUMAN-GENOME MAPPING on the fp16 flow: the fp8
# experiment (e2e_fp8.jl) run through the fp16 cuBLAS GEMM + top-k machinery
# instead of the fp8 one -- the REFERENCE-ACCURACY pipeline for the fp8
# mapping numbers.
#
# PROBLEM
#   Identical to e2e_fp8.jl: the human index as the database, the sample
#   reads streamed through the rope -> GEMM -> top-k pipeline, every read
#   scored against its PROVENANCE (correctly mapped iff the true window
#   [start, start+len-1] intersects at least one of the ktop returned
#   database windows).  The ONE change: the GEMM+top-k stage is the fp16
#   machinery (search/engine_fp16.jl), and the database is the index matrix
#   VERBATIM -- the .bin's fp16 rows are fp16 columns of B with NO
#   quantization step at all, so the only numeric loss vs an ideal pipeline
#   is the GEMM's fp16 accumulation + fp16 C storage.  include ONE of
#   e2e_fp8.jl / e2e_fp16.jl per session (the one-variant rule; engine_fp8
#   and engine_fp16 must never be co-included).
#
# SELF-TEST (mode `test`): the SAME tiny fixture REFERENCE as e2e_fp8.jl
# (1M-base two-record reference, built once and shared; each script builds
# its own format-specific .bin: fp8 columns there, fp16 rows here) -> 1000
# err = 0 reads -> every read must be correctly mapped (all at rank 1).
#
# LOCAL HARNESS VARIANTS
#   This file keeps its OWN copies of the tiny-fixture builders (_e2h_* --
#   the FP16 variants, verbatim): they build the fp16-row ("indexreal.v1")
#   .bin instead of harness.jl's fp8 one and differ in the index NamedTuple.
#   common/harness.jl is therefore NOT included here (its _e2h_* would clash
#   with these local copies); parse_read_head comes from reads/provenance.jl,
#   load_index from index/load.jl, index_to_f16/map_and_count from
#   search/engine_fp16.jl.
#
# SOURCE: legacy/test/e2ehuman16.jl (env consts E2H_* incl. E2H_TEST_*, the
#         local _e2h_* variants, run_e2e_test / run_e2e_human + dispatch).
#         Full measured RESULTS (the fp8-vs-fp16 read-level comparison) live
#         in that file's header.
#
# INCLUDES (canonical order)
#   common/{treearrays,dna,util,testref}.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/fp16.jl, search/engine_fp16.jl, index/{build,load}.jl,
#   reads/provenance.jl   (NO harness.jl, NO fp8 gemm files)
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (1M reference -> index -> err=0 reads -> all mapped)
#   run   production: human index + the E2HUMAN_READS file -> mapping score
#   all   test + run (default)
#   flags: --k=20 (top-k size), --w=32768 (B-column chunk; the fp16 optimum),
#          --batch=8192, --segs=8
#   env:   E2HUMAN_FASTA / E2HUMAN_INDEX / E2HUMAN_READS -- the SAME variables
#          e2e_fp8.jl reads (same data, different pipeline precision)
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/e2e_fp16.jl test
#   julia --project=. -t 16 RotorMap/experiments/e2e_fp16.jl run
#   E2HUMAN_READS=<...e0.15...fasta> julia --project=. -t 16 RotorMap/experiments/e2e_fp16.jl run
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
inc("gemm", "fp16.jl")
inc("search", "engine_fp16.jl")
inc("index", "build.jl")
inc("index", "load.jl")
inc("reads", "provenance.jl")

using Random
using CUDA
using Printf
using Mmap
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (the SAME env variables e2ehuman.jl reads)
# ------------------------------------------------------------------------------
const E2H_FASTA = get(ENV, "E2HUMAN_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna")
# the index matrix (indexreal `save` output, next to the reference fasta)
const E2H_INDEX_BIN = get(ENV, "E2HUMAN_INDEX",
                          string(splitext(E2H_FASTA)[1], ".indexreal.bin"))
# the mutated reads (sample `gen` output; the existing k20000 batch on /share)
const E2H_READS = get(ENV, "E2HUMAN_READS",
                      string(splitext(E2H_FASTA)[1],
                             ".sample_n1048576_k20000_e0.1_s42.fasta"))

# tiny self-test fixture consts (SHARED reference with e2e_fp8.jl; each script
# builds its own format-specific .bin)
const E2H_TEST_BASES = 1_000_000
const E2H_TEST_N = 1000
const E2H_TEST_SEED = 42

# E2H_FORMAT ("indexreal.v1") is layer-provided (index/load.jl); both the
# loader and the local _e2h_build_index below key on it.

# ------------------------------------------------------------------------------
# Tiny self-test fixture (the FP16 variants; NOT harness.jl's -- this file
# builds the v1-layout fp16-row .bin and keeps its own _e2h_* copies to avoid
# the harness clash)
# ------------------------------------------------------------------------------

const _E2H_TEST_REF = joinpath(_v3_data_dir(), "e2ehuman_ref1M.fasta")

function _ensure_e2h_test_ref(; total::Int = E2H_TEST_BASES)
    isfile(_E2H_TEST_REF) && filesize(_E2H_TEST_REF) > total + 10_000 &&
        return _E2H_TEST_REF
    lenA = 7 * total ÷ 10
    lenB = total - lenA
    rs = MersenneTwister(2024)
    bases = collect("ACGT")
    iupac = collect("NRYSWKMBDHV")
    randseq(len) = String([bases[rand(rs, 1:4)] for _ in 1:len])
    function sprinkle(s)
        v = collect(s)
        for i in 1:97:length(v)
            v[i] = lowercase(v[i])
        end
        for i in 41:211:length(v)
            v[i] = rand(rs, iupac)
        end
        return String(v)
    end
    wrapped(s, width) = join((s[i:min(i + width, end)] for i in 1:width:length(s)), "\n")
    mkpath(dirname(_E2H_TEST_REF))
    @info "Generating the tiny $(total)-base test reference -> $_E2H_TEST_REF"
    open(_E2H_TEST_REF, "w") do io
        write(io, ">chrA e2ehuman tiny test record A\n",
              wrapped(sprinkle(randseq(lenA)), 70), "\n")
        write(io, ">chrB e2ehuman tiny test record B\n",
              wrapped(sprinkle(randseq(lenB)), 70), "\n")
    end
    return _E2H_TEST_REF
end

function _e2h_ref_table(raw::Vector{UInt8})
    heads = String[]
    lo = Int[]
    hi = Int[]
    nchar = Int[]
    NL = UInt8('\n')
    CR = UInt8('\r')
    n = length(raw)
    s = _next_rec3(raw, 1, n)
    while s <= n
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n)
        cnt = 0
        @inbounds for p in (h+1):(e-1)
            b = raw[p]
            (b == NL || b == CR) || (cnt += 1)
        end
        push!(heads, String(raw[s:hend]))
        push!(lo, h + 1)
        push!(hi, e - 1)
        push!(nchar, cnt)
        s = e
    end
    return (heads, lo, hi, nchar)
end

function _e2h_span_codes(raw::Vector{UInt8}, lo::Int, hi::Int)
    codes = Vector{UInt8}()
    @inbounds for p in lo:hi
        c = raw[p]
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        push!(codes, _LUT3[Int(c) + 1])
    end
    return codes
end

# codes -> forward 2-bit packed words (16 bases per UInt32), the production
# word convention (verbatim common/harness.jl's copy -- the reorganize dropped
# it from this file's local fixture set; restored 2026-09)
function _ref_forward_pack(codes::Vector{UInt8}, L::Int)
    words = fill(UInt32(0), cld(L, 16)) # 16 bases (32 bits) per word
    for pos in 1:L
        words[(pos - 1) >> 4 + 1] |= UInt32(codes[pos]) << (2 * ((pos - 1) & 15))
    end
    return words
end

function _e2h_sample_reads(ref::String; n::Int, k::Int, err::Real = 0.0,
                           seed::Int = E2H_TEST_SEED,
                           out::String = string(splitext(ref)[1],
                                                ".sample_n$(n)_k$(k)_e$(err)_s$(seed).fasta"))
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    w = [max(l - k + 1, 0) for l in rlen]
    total = sum(w)
    total > 0 || error("reference too short for k = $k")
    cum = accumulate(+, w)
    rng = Xoshiro(seed)
    recs = Vector{Int}(undef, n)
    starts = Vector{Int}(undef, n)
    for i in 1:n
        r = rand(rng, Int64(1):total)
        j = searchsortedfirst(cum, r)
        recs[i] = j
        starts[i] = Int(r - (j > 1 ? cum[j-1] : Int64(0)))
    end
    rngs = [Xoshiro(rand(rng, Int64)) for _ in 1:n]
    reads = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        codes = _e2h_span_codes(raw, rlo[recs[i]], rhi[recs[i]])
        win = codes[starts[i]:starts[i]+k-1]
        err > 0 && (win = mutate(win, err; rng = rngs[i]))
        reads[i] = win
    end
    heads = [">read_$(i) start=$(starts[i]) len=$(k) src=$(_rec_name(rheads[recs[i]]))"
             for i in 1:n]
    save_fasta(reads, out; heads)
    @info "sampled $n reads (err = $err) -> $out"
    return out
end

function _e2h_build_index(ref::String; k::Int = 20_000, kstep::Int = max(k ÷ 10, 1),
                          normalize::Int = 0, fp16::Bool = true,
                          binfile::String = string(splitext(ref)[1], ".indexreal.bin"))
    isfile(binfile) && return binfile # shared fixture reference: this script's
    # own v1-layout .bin (e2ehuman.jl builds its fp8-layout sibling)
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4)
    rdim = 2 * re.m * 4^re.c
    W = cld(k, 16)
    T = fp16 ? Float16 : Float32
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    gheads = String[]
    gstarts = Int[]
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        for wi in 0:((rlen[j] - k) ÷ kstep)
            push!(gheads, rheads[j])
            push!(gstarts, wi * kstep + 1)
        end
    end
    nwin = length(gheads)
    fw = Vector{UInt32}(undef, W * nwin)
    rcw = Vector{UInt32}(undef, W * nwin)
    g = 0
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        codes = _e2h_span_codes(raw, rlo[j], rhi[j])
        @assert length(codes) == rlen[j]
        for wi in 0:((rlen[j] - k) ÷ kstep)
            g += 1
            s = wi * kstep
            copyto!(fw, (g - 1) * W + 1, _ref_forward_pack(codes[s+1:s+k], k), 1, W)
            rcc = [UInt8(0x03 - codes[s+k+1-t]) for t in 1:k]
            copyto!(rcw, (g - 1) * W + 1, _ref_forward_pack(rcc, k), 1, W)
        end
    end
    @assert g == nwin
    dest_f = CUDA.zeros(T, nwin, rdim)
    dest_r = CUDA.zeros(T, nwin, rdim)
    dn_f = CUDA.zeros(Float32, re.m, nwin)
    dn_r = CUDA.zeros(Float32, re.m, nwin)
    encode_frag_real_batch!(dest_f, dn_f, re, cu(fw); normalize)
    encode_frag_real_batch!(dest_r, dn_r, re, cu(rcw); normalize)
    embeds = Matrix{T}(undef, 2 * nwin, rdim)
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    ef = Matrix{T}(undef, nwin, rdim)
    er = Matrix{T}(undef, nwin, rdim)
    nf = Matrix{Float32}(undef, re.m, nwin)
    nr = Matrix{Float32}(undef, re.m, nwin)
    copyto!(ef, dest_f)
    copyto!(er, dest_r)
    copyto!(nf, dn_f)
    copyto!(nr, dn_r)
    copyto!(view(embeds, 1:nwin, :), ef)
    copyto!(view(embeds, nwin+1:2*nwin, :), er)
    copyto!(view(norms, :, 1:nwin), nf)
    copyto!(view(norms, :, nwin+1:2*nwin), nr)
    nt = (format = E2H_FORMAT, source = abspath(ref), k, kstep, s = re.s,
          m = re.m, c = re.c, normalize, fp16, n_frag = nwin, embeds, norms,
          heads = vcat(gheads, gheads), starts = vcat(gstarts, gstarts),
          strand = vcat(fill(UInt8(0), nwin), fill(UInt8(1), nwin)))
    save(nt, binfile)
    @info "tiny index built" binfile k kstep nwin cols = 2 * nwin rdim
    return binfile
end

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

function run_e2e_test(; kfrag::Int = 20_000, n::Int = E2H_TEST_N, ktop::Int = 20,
                      w::Int = 2^15, batch_size::Int = 2^13, segs::Int = 8,
                      seed::Int = E2H_TEST_SEED)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    B = index_database_f16(d)  # exact fp8->fp16 widen or fp16-row transpose
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    w_test = min(w, prevpow(2, N))
    res = map_and_count(re, readsf, B, d.heads, d.starts, d.k; ktop,
                        w = w_test, batch_size, segs)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped"
    @info "TINY E2E (fp16) PASSED: $n/$n correctly mapped" first_hit_ranks = res.rank_hist
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

function run_e2e_human(; ktop::Int = 20, w::Int = 2^15, batch_size::Int = 2^13,
                       segs::Int = 8)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2H_READS) ||
        error("reads fasta not found: $E2H_READS (generate with reads/sample.jl)")
    @info "loading the human index" E2H_INDEX_BIN
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    @info "building the fp16 database on GPU (rows -> columns, verbatim)"
    t_B = @elapsed B = index_database_f16(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    w = min(w, prevpow(2, size(B, 2))) # small test indexes admit no production w
    @info "fp16 database built" dims = size(B) gib = round(length(B) * 2 / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # the 12.7 GiB host index matrix can go
    GC.gc(); CUDA.reclaim()
    res = map_and_count(re, E2H_READS, B, heads, starts, dk; ktop, w,
                        batch_size, segs, progress = true)
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
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
    ktop = parse(Int, getflag("k", "20"))
    w = parse(Int, getflag("w", string(2^15)))
    batch = parse(Int, getflag("batch", string(2^13)))
    segs = parse(Int, getflag("segs", "8"))
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2e_test(; ktop, w, batch_size = batch, segs)
    mode == "run" && run_e2e_human(; ktop, w, batch_size = batch, segs)
    mode == "all" && (run_e2e_test(; ktop, w, batch_size = batch, segs);
                      run_e2e_human(; ktop, w, batch_size = batch, segs))
    mode in ("test", "run", "all") ||
        error("unknown mode $mode (use test|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
