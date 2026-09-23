# ==============================================================================
# e2ehuman16.jl -- END-TO-END HUMAN-GENOME MAPPING on the fp16 flow: the
# e2ehuman.jl experiment run through flowtopk.jl (fp16 cuBLAS GEMM + top-k)
# instead of flowtopkfp8.jl (fp8) -- the REFERENCE-ACCURACY pipeline for the
# fp8 mapping numbers.
#
# PROBLEM
#   Identical to e2ehuman.jl (see its header): the indexflowreal human index
#   as the database, the indexsample reads streamed through the rope -> GEMM
#   -> top-k pipeline, every read scored against its PROVENANCE (correctly
#   mapped iff the true window [start, start+len-1] intersects at least one
#   of the ktop returned database windows).  The ONE change: the GEMM+top-k
#   stage is flowtopk.jl's fp16 machinery, and the database is the index
#   matrix VERBATIM -- the .bin's fp16 rows are fp16 columns of B with NO
#   quantization step at all, so the only numeric loss vs an ideal pipeline
#   is the GEMM's fp16 accumulation + fp16 C storage (flowtopk's measured
#   ~3e-3 score noise).  include ONE of e2ehuman.jl / e2ehuman16.jl per
#   session (both re-define flowtopk/flowtopkfp8's structs via their flow
#   includes -- the project's one-variant-per-session rule).
#
# DESIGN
#   Mirrors e2ehuman.jl section by section:
#   1. Whole-matrix load: ALL index rows become database columns (fwd + rc,
#     3,094,162 columns = 12.67 GiB fp16 on the GPU; rc hits are the same
#     genomic location, no strand special-casing).
#   2. Blockwise index->GPU upload: per 2^16-row block H2D + fused GPU
#     transpose into B's contiguous column range -- pure fp16 moves, no
#     arithmetic (vs e2ehuman's extra e4m3 quantize of the same data).
#   3. normalize = 0 ASSUMED (asserted): unit-energy rows => normalized
#     columns; the rope stream runs the same normalize mode.
#   4. Provenance parsing / intersection scoring: verbatim e2ehuman.jl.
#   5. Shared runner: the tiny self-test and the production run differ only
#     in the inputs.  Defaults follow flowtopk's measurements (w = 2^15 is
#     its chunk-width optimum; no engine/cout knobs exist on the fp16 path).
#
# SELF-TEST (mode `test`): the SAME tiny fixture REFERENCE as e2ehuman.jl
# (1M-base two-record reference, built once and shared; each script builds
# its own format-specific .bin: fp8 columns there, fp16 rows here) -> 1000
# err = 0 reads -> every read must be correctly mapped (all at rank 1).
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16; index = the production
# indexflowreal save of GRCh38.p14: 3,094,162 columns x rdim 2048 fp16,
# 67 unique records; k(top) = 20, w = 2^15, batch = 2^13 -> 16 batches):
#
#   index .bin load (page-cache warm)                     2.0 s
#   fp16 database build (2^16-row blocks, H2D+transpose)  8.9 s  -> 11.8 GiB fp16
#   flow END-TO-END: reads -> rope fp16 -> gemm+topk     11.4 s  (131,072 reads;
#     the fp8 flow needed 8.1-8.5 s on the same data -> ~1.4x for the e4m3
#     GEMM at these shapes)
#   err = 0.05 reads: 124,474 / 131,072 = 94.97% mapped (rank-1: 122,441;
#     unmapped 6,598) -- vs the fp8 flow's 124,486 / 94.98% (rank-1 122,429):
#     a TWELVE-read difference.  err = 0.15: 77,920 / 131,072 = 59.45%
#     (rank-1: 55,052; unmapped 53,152) vs fp8's 77,942 / 59.47%: a
#     TWENTY-two-read difference.  At these error rates the any-of-20
#     intersection score is completely insensitive to the e4m3 quantization:
#     the mutation error dominates, and quantization noise only permutes
#     near-ties WITHIN the top-20 (and occasionally across its boundary).
#   tiny self-test: 1000/1000 correctly mapped, all at rank 1 (shared fixture
#     with e2ehuman.jl; as fp8)
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (1M reference -> index -> err=0 reads -> all mapped)
#   run   production: human index + the E2HUMAN_READS file -> mapping score
#   all   test + run (default)
#   flags: --k=20 (top-k size), --w=32768 (B-column chunk; flowtopk's optimum),
#          --batch=8192, --segs=8
#   env:   E2HUMAN_FASTA / E2HUMAN_INDEX / E2HUMAN_READS -- the SAME variables
#          e2ehuman.jl reads (same data, different pipeline precision)
#
# USAGE
#   julia --project=. -t 16 test/e2ehuman16.jl test
#   julia --project=. -t 16 test/e2ehuman16.jl run
#   E2HUMAN_READS=<...e0.15...fasta> julia --project=. -t 16 test/e2ehuman16.jl run
# ==============================================================================

include(joinpath(@__DIR__, "flowtopk.jl")) # the whole fp16 flow: TopKEngine,
# topk_flow, rope_topk_stream, seg_rowtopk_merge_kernel! (+ ropeflowreal /
# fastareads_v3 transitively: RopeEncoder, encode_frag_real_batch!,
# _ref_forward_pack, _next_rec3, _LUT3, Utils, Mmap, Random, CUDA)

using Random
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (the SAME env variables e2ehuman.jl reads)
# ------------------------------------------------------------------------------
const E2H_FASTA = get(ENV, "E2HUMAN_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
# the index matrix (indexflowreal `save` output, next to the reference fasta)
const E2H_INDEX_BIN = get(ENV, "E2HUMAN_INDEX",
                          string(splitext(E2H_FASTA)[1], ".indexflowreal.bin"))
# the mutated reads (indexsample `gen` output; default = the err = 0.05 file)
const E2H_READS = get(ENV, "E2HUMAN_READS",
                      string(splitext(E2H_FASTA)[1],
                             ".indexsample_n131072_k20000_e0.05_s42.fasta"))

const E2H_FORMAT = "indexflowreal.v1" # the .bin format tag (indexflowreal's)

# tiny self-test fixture (SHARED with e2ehuman.jl: same reference + .bin)
const E2H_TEST_BASES = 1_000_000
const E2H_TEST_N = 1000
const E2H_TEST_SEED = 42

# ------------------------------------------------------------------------------
# Ground truth: indexsample's provenance headers
# ------------------------------------------------------------------------------
function parse_read_head(h::AbstractString)
    startswith(h, ">read_") || return nothing
    sp = findfirst(" src=", h)
    sp === nothing && return nothing
    fields = split(h[1:first(sp)-1])
    length(fields) >= 2 || return nothing
    kv = Dict{String,String}()
    for f in fields[2:end]
        eq = findfirst('=', f)
        eq === nothing && return nothing
        kv[String(f[1:eq-1])] = String(f[eq+1:end])
    end
    haskey(kv, "start") && haskey(kv, "len") || return nothing
    start = tryparse(Int, kv["start"])
    len = tryparse(Int, kv["len"])
    (start === nothing || len === nothing) && return nothing
    return (id = String(fields[1]), start = start, len = len,
            src = String(h[last(sp)+1:end]))
end

# ------------------------------------------------------------------------------
# The index .bin -> resident fp16 database
# ------------------------------------------------------------------------------

"""
    load_index(path = E2H_INDEX_BIN)

Load an indexflowreal-format index .bin (Julia-serialized NamedTuple).  This
flow assumes (asserts) `normalize = 0`: unit-energy rows => normalized
database columns.
"""
function load_index(path::String = E2H_INDEX_BIN)
    isfile(path) || error("index .bin not found: $path (generate with " *
                          "indexflowreal.jl `save`)")
    d = Utils.load(path)
    get(d, :format, nothing) == E2H_FORMAT ||
        error("not an $E2H_FORMAT index: $path")
    d.normalize == 0 ||
        error("e2ehuman assumes a normalize = 0 index (got $(d.normalize)): " *
              "unit-energy rows are what makes B's columns normalized")
    rdim = 2 * d.m * 4^d.c
    N = 2 * d.n_frag
    @assert eltype(d.embeds) == (d.fp16 ? Float16 : Float32) "embeds eltype mismatch"
    @assert size(d.embeds) == (N, rdim) "embeds shape mismatch"
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"
    @info "index loaded" file = basename(path) k = d.k kstep = d.kstep n_frag =
        d.n_frag cols = N rdim encoder = (d.s, d.m, d.c) normalize = d.normalize
    return d
end

"""
    index_to_f16(embeds; block = 2^16) -> B::CuMatrix{Float16}

Upload the index matrix (rows = reference-window encodings) as the fp16 GEMM
database B (columns = reference windows): per row-block H2D + GPU transpose.
The fp16 index IS the database -- pure data movement, no quantization (the
fp8 flow's e2ehuman.jl quantizes the very same blocks to e4m3).  Peak device
memory: B + one block.  Runs on the default stream, returns synchronized.
"""
function index_to_f16(embeds::AbstractMatrix{Float16}; block::Int = 2^16)
    N, rdim = size(embeds)
    B = CuMatrix{Float16}(undef, rdim, N)
    for j0 in 1:block:N
        j1 = min(j0 + block - 1, N)
        db = CuArray{Float16}(@view embeds[j0:j1, :]) # (nb, rdim) H2D, contiguous rows
        @views B[:, j0:j1] .= permutedims(db) # fused GPU transpose, pure fp16 moves
    end
    CUDA.synchronize()
    return B
end

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, score the top-k against the provenance
# ------------------------------------------------------------------------------

"""
    map_and_count(re, reads_file, B, db_heads, db_starts, dbk; ktop, w,
                  batch_size, segs, normalize, progress)
        -> (total, correct, rank_hist, seconds)

Run the full fp16 flow (`rope_topk_stream`: fasta -> rope fp16 -> GEMM +
per-row top-k) over `reads_file` and score every row against its provenance
header: CORRECTLY MAPPED iff the true window [start, start+len-1] in record
src intersects at least one of the ktop returned database windows
(db_heads[c], db_starts[c], length dbk).
"""
function map_and_count(re::RopeEncoder, reads_file::String, B::CuMatrix{Float16},
                       db_heads::Vector{String}, db_starts::Vector{Int}, dbk::Int;
                       ktop::Int = 20, w::Int = 2^15, batch_size::Int = 2^13,
                       segs::Int = 8, normalize::Int = 0,
                       progress::Bool = false)
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(B, 2)
    # per-column record id: intern the handful of unique record headers once
    rec_id = Dict{String,Int32}()
    rec_of_col = Vector{Int32}(undef, N)
    for j in 1:N
        rec_of_col[j] = get!(rec_id, db_heads[j]) do
            Int32(length(rec_id) + 1)
        end
    end
    total = correct = nb_bat = 0
    rank_hist = zeros(Int, ktop)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in rope_topk_stream(re, reads_file, B; k = ktop, w,
                                             batch_size, rows_cap = batch_size,
                                             normalize, segs, progress,
                                             err_out = err, tasks_out = tks)
        nb_bat += 1
        nb = size(bat.vals, 1)
        for r in 1:nb
            h = parse_read_head(bat.heads[r])
            h === nothing &&
                error("unreadable provenance header: $(bat.heads[r])")
            h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rid = get(rec_id, string('>', h.src), Int32(0))
            hit = 0
            @inbounds for j in 1:ktop
                c = Int(bat.locs[r, j])
                # read window [h.start, h.start+h.len-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid &&
                   h.start <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= h.start + h.len - 1
                    hit = j
                    break
                end
            end
            total += 1
            if hit > 0
                correct += 1
                rank_hist[hit] += 1
            end
        end
        @printf("  batch %2d: +%d reads, %d/%d correctly mapped so far\n",
                nb_bat, nb, correct, total)
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    return (total = total, correct = correct, rank_hist = rank_hist, seconds = t)
end

# ------------------------------------------------------------------------------
# Tiny self-test fixture (shared with e2ehuman.jl)
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

function _e2h_sample_reads(ref::String; n::Int, k::Int, err::Real = 0.0,
                           seed::Int = E2H_TEST_SEED,
                           out::String = string(splitext(ref)[1],
                                                ".indexsample_n$(n)_k$(k)_e$(err)_s$(seed).fasta"))
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
        err > 0 && (win = Utils.mutate(win, err; rng = rngs[i]))
        reads[i] = win
    end
    heads = [">read_$(i) start=$(starts[i]) len=$(k) src=$(rheads[recs[i]][2:end])"
             for i in 1:n]
    Utils.save_fasta(reads, out; heads)
    @info "sampled $n reads (err = $err) -> $out"
    return out
end

function _e2h_build_index(ref::String; k::Int = 20_000, kstep::Int = max(k ÷ 10, 1),
                          normalize::Int = 0, fp16::Bool = true,
                          binfile::String = string(splitext(ref)[1], ".indexflowreal.bin"))
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
    Utils.save(nt, binfile)
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
    B = index_to_f16(d.embeds)
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
        error("reads fasta not found: $E2H_READS (generate with indexsample.jl)")
    @info "loading the human index" E2H_INDEX_BIN
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    @info "building the fp16 database on GPU (rows -> columns, verbatim)"
    t_B = @elapsed B = index_to_f16(d.embeds)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
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
if abspath(PROGRAM_FILE) == @__FILE__
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
