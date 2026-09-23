# ==============================================================================
# e2e_complex.jl -- END-TO-END HUMAN-GENOME MAPPING on the COMPLEX flow: the
# rope encodings are treated as what they are -- COMPLEX vectors z in C^D
# (D = m*4^c, bin = (csmer-1)*m + idm) -- the GEMM stage computes the COMPLEX
# inner products <z_query, z_db> with TWO real fp8 GEMMs, and the top-k stage
# ranks by the inner-product MAGNITUDE |<z_query, z_db>|^2 (abs2; ranking by
# |.| is equivalent).  Every read's top-k hits are checked against its
# PROVENANCE (ground truth), exactly as in e2e_fp8.jl.
#
# The REAL flow (e2e_fp8.jl / search/engine_fp8.jl) is UNTOUCHED: this file
# adds the complex variant ALONGSIDE it -- new names, own file.  The complex
# engine is a separate struct (search/engine_complex.jl), not an option
# inside the real one; engine_fp8.jl is NOT included here (the
# one-variant-per-session rule).
#
# THE COMPLEX INNER PRODUCT VIA TWO REAL FP8 GEMMs
#   The real flow's split layout [Re(z); Im(z)] IS the complex vector in
#   split form, so the ENCODING stage is shared verbatim (the rope kernel's
#   norms are complex moduli already).  What changes is the SCORE:
#     Re<z_a, z_b> = a_split . [Yr; Yi]  -- exactly the real flow's database;
#     Im<z_a, z_b> = a_split . [Yi; -Yr] -- the split of -i*z_b (the SAME
#                                           database rotated by -90 degrees).
#   The complex flow runs the SAME fp8 GEMM TWICE per chunk -- once against
#   Bre, once against Bim -- and scores score(r, j) = Cr^2 + Ci^2 =
#   |<z_r, z_j>|^2.  Rationale: a read sampled within kstep/2 of an index
#   window is a SHIFTED copy of it, so its encoding is (approximately) the
#   window's encoding rotated by a GLOBAL phase e^(i*theta*d): the real
#   flow's score decays as cos(theta*d) while abs2 is phase invariant -- the
#   complex magnitude is the right statistic for shifted matching windows.
#
# NORMS: with complex vectors the norm is the MODULUS (the kernel's norms
#   already are); normalize = 0 still means unit energy == the squared norm
#   of the split row.  TopKBatch.vals are now |<z_query, z_db>|^2 in [0, ~1],
#   computed FUSED in the top-k scan.
#
# INDEX: the complex index save is the split-form format
#   "indexflowcomplex.fp8.v1" (<fasta>.indexflowcomplex_fp8_ks<kstep>.bin):
#   embeds8_re / embeds8_im (D, 2*n_frag) e4m3 columns -- a matrix of real
#   values together with a matrix of imaginary values, NEVER Matrix{Complex}.
#   load_index (index/load.jl) ALSO accepts the legacy "indexflowreal.fp8.v1"
#   save; index_database (index/complex.jl) -> (Bre, Bim), both resident.
#
# SELF-TEST (mode `test`): the SAME tiny fixture REFERENCE as e2e_fp8.jl
#   (1M-base two-record reference, built once and shared; the LOCAL _e2h_*
#   variants below build the split-form .bin) -> 1000 err = 0 reads -> ALL
#   must map (at rank 1): abs2 is invariant to the global phase rotation AND
#   quantization noise, so a miss means broken bookkeeping, not noise.
#
# CORRECTNESS (mode `check`): e4m3 quantize/upload == host F8.() bitwise;
#   the complex segmented top-k kernel vs the CPU abs2 multiset over many
#   shapes; the fp8 GEMM unit tests re-run in-process; the complex engine vs
#   the exact same-engine chunked reference for both engines x both couts;
#   the split save -> (Bre, Bim) derivation BITWISE; end-to-end stream vs the
#   float64 CPU reference; early-close teardown; empty file.  ALL PASS (kau).
#
# ENTRY POINT
#   The complex engine + streamed flow (ComplexTopKEngine,
#   upload_fp16_as_f8c!, _gemm_chunk_c!, complex_gemm_topk!,
#   launch_seg_rowtopk_cmerge!, complex_topk_flow, complex_rope_topk_stream)
#   lives in search/engine_complex.jl; the complex loader/derivation and the
#   production index builder (_build_complex_index_matrix) in index/load.jl +
#   index/complex.jl; parse_read_head in reads/provenance.jl; _timed_gpu in
#   common/util.jl.  This file keeps the env consts, the LOCAL _e2h_* fixture
#   variants (they build the split-form .bin; harness.jl's would build the
#   real-flow one -- hence NO harness.jl include here), the correctness
#   suite, and the run entry points -- verbatim from the legacy experiment.
#
# SOURCE: legacy/test/e2ecomplex.jl.  Full measured RESULTS (the err = 0.15
#         accuracy comparison vs the real flow, stage speeds, encoder sweep)
#         live in that file's header.
#
# INCLUDES (canonical order)
#   common/{treearrays,dna,util,testref}.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl,
#   search/engine_complex.jl, index/{build,load,complex}.jl,
#   reads/provenance.jl   (NO harness.jl -- local _e2h_* variants; NO
#   engine_fp8.jl)
#
# Run modes (first non-flag ARGV[1]):
#   save  build the complex human index (kstep = k/16 = 1250 default)
#   check correctness: quantize path, complex top-k kernel unit tests,
#         engine checks, loader/derivation checks, end-to-end stream checks
#   test  tiny end-to-end (1M reference -> complex index -> err=0 reads ->
#         all mapped)
#   bench stage timings on the human index + reads
#   run   production: human complex index + n131072_k20000_e0.15 reads ->
#         mapping score
#   all   test + bench + run (default)
#   flags: --k=20, --w=65536, --batch=8192, --segs=8, --engine=lt, --reps=1
#   env:   E2COMPLEX_FASTA / E2COMPLEX_INDEX / E2COMPLEX_READS /
#          E2COMPLEX_K / E2COMPLEX_KSTEP
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/e2e_complex.jl save
#   julia --project=. -t 16 RotorMap/experiments/e2e_complex.jl check
#   julia --project=. -t 16 RotorMap/experiments/e2e_complex.jl test
#   julia --project=. -t 16 RotorMap/experiments/e2e_complex.jl run --engine=lt
#   julia --project=. -t 16 RotorMap/experiments/e2e_complex.jl  # test+bench+run
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

# --- layers (canonical order) -------------------------------------------------
inc("common", "treearrays.jl")
inc("common", "dna.jl")
inc("common", "util.jl") # provides Utils (+ _timed_gpu)
inc("common", "testref.jl") # provides ensure_data3, _V3_SMALL_READS, _v3_data_dir
inc("fasta", "pack.jl")
inc("fasta", "loader.jl")
inc("fasta", "reader.jl")
inc("encode", "ropeencoder.jl") # RopeEncoder, _frag_codes
inc("encode", "encoder_v3.jl")
inc("encode", "reference.jl") # _ref_rope_frag_real
inc("encode", "kernel.jl") # encode_frag_real_batch!, _ref_forward_pack
inc("encode", "stream.jl") # RopeRealBatch, rope_encode_real_stream,
#                            ref_rope_real_stream, _count_ref_windows
inc("gemm", "fp8_convert.jl")
inc("gemm", "fp8_ptx.jl")
inc("gemm", "fp8_lt.jl")
inc("gemm", "topk_kernels.jl")
inc("search", "engine_complex.jl")
inc("index", "build.jl")
inc("index", "load.jl") # load_index (both formats), E2C_LEGACY_FP8_FORMAT
inc("index", "complex.jl") # index_database -> (Bre, Bim), _complex_databases,
#                            E2C_FORMAT, _build_complex_index_matrix
inc("reads", "provenance.jl") # parse_read_head

using Random
using CUDA
using Printf
using Mmap
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides)
# ------------------------------------------------------------------------------
const E2C_FASTA = get(ENV, "E2COMPLEX_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna")
const E2C_K = parse(Int, get(ENV, "E2COMPLEX_K", "20000"))        # fragment length
const E2C_KSTEP = parse(Int, get(ENV, "E2COMPLEX_KSTEP", string(E2C_K ÷ 16))) # k/16 = 1250
# the complex index: the split-form save of THIS script (the legacy real fp8
# layout loads too -- see load_index / index_database)
const E2C_INDEX_BIN = get(ENV, "E2COMPLEX_INDEX",
                          string(splitext(E2C_FASTA)[1],
                                 ".indexflowcomplex_fp8_ks", E2C_KSTEP, ".bin"))
# the mutated reads (sample `gen` output; the existing k20000 batch on /share)
const E2C_READS = get(ENV, "E2COMPLEX_READS",
                      string(splitext(E2C_FASTA)[1],
                             ".sample_n1048576_k20000_e0.1_s42.fasta"))

# E2C_FORMAT ("indexflowcomplex.fp8.v1", index/complex.jl) and
# E2C_LEGACY_FP8_FORMAT ("indexflowreal.fp8.v1", index/load.jl) are
# layer-provided; load_index dispatches on them.
# tiny self-test fixture (the SAME cached 1M reference as e2ehuman.jl)
const E2C_TEST_BASES = 1_000_000
const E2C_TEST_N = 1000
const E2C_TEST_SEED = 42

# ------------------------------------------------------------------------------
# Tiny self-test fixture: the SHARED 1M-base two-record reference
# (e2ehuman.jl's fixture helpers, verbatim), a COMPLEX split-form index built
# from it, and err = 0 sampled reads
# ------------------------------------------------------------------------------

const _E2H_TEST_REF = joinpath(_v3_data_dir(), "e2ehuman_ref1M.fasta")

function _ensure_e2h_test_ref(; total::Int = E2C_TEST_BASES)
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
                           seed::Int = E2C_TEST_SEED,
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

"""
    _e2h_build_index(ref; k, kstep, normalize, binfile) -> binfile::String

A miniature of the COMPLEX production save: every k-window (step kstep, last
window fully inside the record) of every record of the tiny reference,
forward AND reverse complement, encoded with the PRODUCTION kernel
(`encode_frag_real_batch!`), split into its Re/Im halves, quantized to e4m3
columns and serialized under the split-form complex naming
`<ref>.indexflowcomplex_fp8_ks<kstep>.bin` (format
"indexflowcomplex.fp8.v1": embeds8_re / embeds8_im (D, 2*nwin)) -- so the
test exercises the same load path as the human run.
"""
function _e2h_build_index(ref::String; k::Int = 20_000, kstep::Int = max(k ÷ 16, 1),
                          normalize::Int = 0,
                          binfile::String = string(splitext(ref)[1],
                                                   ".indexflowcomplex_fp8_ks$(kstep).bin"))
    isfile(binfile) && return binfile # shared fixture: a previous run built it
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    D = re.m * 4^re.c
    rdim = 2 * D
    W = cld(k, 16)
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    gheads = String[] # per-window location meta (file order)
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
        codes = _e2h_span_codes(raw, rlo[j], rhi[j]) # full record, once
        @assert length(codes) == rlen[j]
        for wi in 0:((rlen[j] - k) ÷ kstep)
            g += 1
            s = wi * kstep
            copyto!(fw, (g - 1) * W + 1, _ref_forward_pack(codes[s+1:s+k], k), 1, W)
            rcc = [UInt8(0x03 - codes[s+k+1-t]) for t in 1:k] # revcomp codes
            copyto!(rcw, (g - 1) * W + 1, _ref_forward_pack(rcc, k), 1, W)
        end
    end
    @assert g == nwin
    dest_f = CUDA.zeros(Float16, nwin, rdim)
    dest_r = CUDA.zeros(Float16, nwin, rdim)
    dn_f = CUDA.zeros(Float32, re.m, nwin)
    dn_r = CUDA.zeros(Float32, re.m, nwin)
    encode_frag_real_batch!(dest_f, dn_f, re, cu(fw); normalize)
    encode_frag_real_batch!(dest_r, dn_r, re, cu(rcw); normalize)
    ef = Matrix{Float16}(undef, nwin, rdim)
    er = Matrix{Float16}(undef, nwin, rdim)
    nf = Matrix{Float32}(undef, re.m, nwin)
    nr = Matrix{Float32}(undef, re.m, nwin)
    copyto!(ef, dest_f)
    copyto!(er, dest_r)
    copyto!(nf, dn_f)
    copyto!(nr, dn_r)
    # SPLIT complex save: embeds8_re / embeds8_im (D, 2*nwin) e4m3 columns --
    # fwd cols 1:nwin, rc cols nwin+1:2nwin; the exact F8.(Float32.(.))
    # quantization of the fp16 kernel output (rows 1:nwin fwd, rest rc).
    # NB: the Re/Im halves are view-bound FIRST (a nested @view inside @views
    # gets its argument rewritten and fails to parse)
    f_re = @view ef[:, 1:D]
    f_im = @view ef[:, D+1:rdim]
    r_re = @view er[:, 1:D]
    r_im = @view er[:, D+1:rdim]
    embeds8_re = Matrix{F8}(undef, D, 2 * nwin)
    embeds8_im = Matrix{F8}(undef, D, 2 * nwin)
    @views embeds8_re[:, 1:nwin] .= F8.(Float32.(permutedims(f_re)))
    @views embeds8_im[:, 1:nwin] .= F8.(Float32.(permutedims(f_im)))
    @views embeds8_re[:, nwin+1:2*nwin] .= F8.(Float32.(permutedims(r_re)))
    @views embeds8_im[:, nwin+1:2*nwin] .= F8.(Float32.(permutedims(r_im)))
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    copyto!(view(norms, :, 1:nwin), nf)
    copyto!(view(norms, :, nwin+1:2*nwin), nr)
    nt = (format = E2C_FORMAT, source = abspath(ref), k, kstep, s = re.s,
          m = re.m, c = re.c, normalize, complex = true, fp8 = true, n_frag = nwin,
          embeds8_re, embeds8_im, norms, heads = vcat(gheads, gheads),
          starts = vcat(gstarts, gstarts),
          strand = vcat(fill(UInt8(0), nwin), fill(UInt8(1), nwin)))
    save(nt, binfile)
    @info "tiny complex index built" binfile k kstep nwin cols = 2 * nwin D rdim
    return binfile
end

# ==============================================================================
# Correctness.
# ==============================================================================

# The e4m3 quantize/upload path (shared with the real flow): device bits must
# equal the host F8.() conversion, pad rows zeroed.
function _check_quantize_path_c()
    Random.seed!(7)
    nb, rdim, cap = 96, 64, 128
    emb = randn(Float16, nb, rdim)
    a8 = CuMatrix{F8}(undef, cap, rdim)
    a32 = CUDA.zeros(Float32, cap, rdim)
    fill!(a8, F8(1.0)) # poison everything not covered by the converter
    upload_fp16_as_f8c!(a8, a32, emb)
    got = Array(a8)
    ref = F8.(Float32.(emb))
    @assert all(got[1:nb, :] .== ref) "device e4m3 quantization != host F8.() conversion"
    @assert all(iszero, @view(got[nb+1:cap, :])) "pad rows not zeroed"
    @info "  e4m3 quantize/upload path OK (device == host F8.(), pad rows zeroed)"
    return nothing
end

# random SPLIT complex database with unit-MODULUS columns (the production
# normalize = 0 shape): returns the resident (Bre, Bim) pair
function _rand_csplit_db(::Type{T}, D::Int, N::Int) where {T<:Union{Float16,Float32}}
    yr = randn(T, D, N)
    yi = randn(T, D, N)
    nrm = sqrt.(sum(abs2, Float32.(yr); dims = 1) .+
                sum(abs2, Float32.(yi); dims = 1))
    yr ./= nrm
    yi ./= nrm
    re8 = F8.(Float32.(yr))
    im8 = F8.(Float32.(yi))
    return (CuArray(vcat(re8, im8)), CuArray(vcat(im8, F8.(Float32.(.-yr)))))
end

# exact multiset check against a CPU abs2 sort for the COMPLEX segmented
# kernel, for both C element types (fp16, fp32), over several shapes including
# ragged chunk sizes (w ∤ N), a single row, the exact full-matrix chunk size,
# engine-style strided parent views, w == k and an empty-segment case
function test_seg_rowtopk_cmerge_kernel(; k::Integer = 20, segs::Integer = 8)
    CUDA.seed!(321)
    rtol = 2e-5
    atol = 1e-6
    for Tc in (Float16, Float32)
        for (M, N, w) in ((64, 100, 37), (257, 128, 128), (1000, 333, 111), (1, 4096, 512),
                          (511, 2048, 2048), (2048, 2^14, 2^13), (33, 37, 37), (64, 25, 25),
                          (64, 20, 20))
            Crf = randn(Float32, M, N)
            Cif = randn(Float32, M, N)
            Cr = CuMatrix{Tc}(Crf)
            Ci = CuMatrix{Tc}(Cif)
            Crh = Float32.(Array(Cr)) # the chunk values AS STORED
            Cih = Float32.(Array(Ci))
            D_val = CUDA.fill(typemin(Float32), M, k)
            D_loc = CUDA.zeros(Int32, M, k)
            for i in 1:cld(N, w)
                lo = (i - 1) * w + 1
                wi = min(N, i * w) - lo + 1
                parent_r = CuMatrix{Tc}(undef, M, w)  # engine-style buffer, ld = w
                parent_i = CuMatrix{Tc}(undef, M, w)
                prv = @view parent_r[:, 1:wi]         # strided when wi < w
                piv = @view parent_i[:, 1:wi]
                copyto!(prv, @view Cr[:, lo:lo+wi-1])
                copyto!(piv, @view Ci[:, lo:lo+wi-1])
                launch_seg_rowtopk_cmerge!(D_val, D_loc, prv, piv, lo - 1, k; segs)
            end
            Dv = Array(D_val)
            Dl = Array(D_loc)
            for r in 1:M
                sc = [Crh[r, j] * Crh[r, j] + Cih[r, j] * Cih[r, j] for j in 1:N]
                ref = sort(sc; rev = true)[1:k]
                got = Dv[r, :]
                @assert issorted(got; rev = true) "row $r of ($M,$N,$w,$Tc): output not sorted"
                @assert isapprox(sort(got; rev = true), ref; rtol, atol) "row $r of ($M,$N,$w,$Tc): wrong top-$k abs2 values"
                @assert all(j -> 1 <= Dl[r, j] <= N, 1:k) "row $r of ($M,$N,$w,$Tc): locations out of range"
                @assert isapprox(sc[Dl[r, :]], got; rtol, atol) "row $r of ($M,$N,$w,$Tc): locations inconsistent with values"
            end
        end
    end
    @info "unit tests passed: seg_rowtopk_cmerge_kernel! matches the CPU abs2 reference " *
          "(Tc=Float16,Float32; ragged/strided/w==k/empty-segment shapes)"
    return nothing
end

# The engine's chunk loop, exactly: the same GEMM engine/cout per chunk pair
# (same shapes AND leading dimensions -> bitwise-identical C values on a
# deterministic engine), materialized on the host for an exact CPU-reference
# comparison of the merged abs2 top-k.
function _ref_ctopk_chunked(eng::ComplexTopKEngine)
    rdim, N = size(eng.Bre)
    cr = CuMatrix{eltype(eng.cr1)}(undef, eng.rows_cap, eng.w)
    ci = CuMatrix{eltype(eng.cr1)}(undef, eng.rows_cap, eng.w)
    Crh = Matrix{Float32}(undef, eng.rows_cap, N)
    Cih = Matrix{Float32}(undef, eng.rows_cap, N)
    for i in 1:cld(N, eng.w)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        _gemm_chunk_c!(eng, cr, byteptr(pointer(eng.Bre), (lo - 1) * rdim), wi)
        _gemm_chunk_c!(eng, ci, byteptr(pointer(eng.Bim), (lo - 1) * rdim), wi)
        Crh[:, lo:lo+wi-1] .= _gather_rows(@view(cr[:, 1:wi]))
        Cih[:, lo:lo+wi-1] .= _gather_rows(@view(ci[:, 1:wi]))
    end
    return (Crh, Cih)
end

function _assert_ctopk_rows(Crh::Matrix{Float32}, Cih::Matrix{Float32},
                            Dv::Matrix{Float32}, Dl::Matrix{Int32}, k::Int)
    M, N = size(Crh)
    @assert size(Dv) == (M, k) && size(Dl) == (M, k) "output shape mismatch"
    rtol = 2e-5
    atol = 1e-6
    for r in 1:M
        sc = [Crh[r, j] * Crh[r, j] + Cih[r, j] * Cih[r, j] for j in 1:N]
        ref = partialsort!(copy(sc), 1:k; rev = true)
        got = Dv[r, :]
        @assert issorted(got; rev = true) "row $r: output not sorted"
        @assert isapprox(sort(got; rev = true), ref; rtol, atol) "row $r: wrong top-$k abs2 values"
        @assert all(j -> 1 <= Dl[r, j] <= N, 1:k) "row $r: locations out of range"
        @assert isapprox(sc[Dl[r, :]], got; rtol, atol) "row $r: locations inconsistent with values"
    end
    return nothing
end

# dequantized-fp32 gemv spot check of BOTH components (gemmtopkfp8's
# spot_check, complexified): recompute a few random rows' scores exactly in
# fp32 from the dequantized operands and compare top-k values + locations
function _spot_check_complex(eng::ComplexTopKEngine, D_val::CuMatrix{Float32},
                             D_loc::CuMatrix{Int32}, k::Integer, nb::Integer;
                             nrows::Integer = 4, seed::Integer = 42, tol::Real = 0.05)
    Random.seed!(seed)
    rdim, N = size(eng.Bre)
    a32 = CuVector{Float32}(undef, rdim)
    wre = 2^14
    Bre32 = CuMatrix{Float32}(undef, rdim, wre)
    Bim32 = CuMatrix{Float32}(undef, rdim, wre)
    ref_re = CuVector{Float32}(undef, N)
    ref_im = CuVector{Float32}(undef, N)
    maxerr = 0.0
    for _ in 1:nrows
        r = rand(1:nb) # a real (non-pad) batch row
        copyto!(a32, Float32.(Array(@view eng.a8[r, :]))) # dequantize the row
        fill!(ref_re, 0f0)
        fill!(ref_im, 0f0)
        for lo in 1:wre:N
            wi = min(N, lo + wre - 1) - lo + 1
            f8_to_f32!(byteptr(pointer(Bre32)), byteptr(pointer(eng.Bre), (lo - 1) * rdim), rdim * wi)
            f8_to_f32!(byteptr(pointer(Bim32)), byteptr(pointer(eng.Bim), (lo - 1) * rdim), rdim * wi)
            mul!(@view(ref_re[lo:lo+wi-1]), transpose(@view Bre32[:, 1:wi]), a32)
            mul!(@view(ref_im[lo:lo+wi-1]), transpose(@view Bim32[:, 1:wi]), a32)
        end
        reh = Array(ref_re)
        imh = Array(ref_im)
        sc = reh .^ 2 .+ imh .^ 2
        refv = sort(sc; rev = true)[1:k]
        gotv = Array(@view D_val[r, :])
        goti = Array(@view D_loc[r, :])
        @assert all(1 .<= goti .<= N) "row $r: locations out of range"
        @assert issorted(gotv; rev = true) "row $r: values not sorted"
        maxerr = max(maxerr, maximum(abs.(refv .- gotv)))
        @assert count(abs.(sc[goti] .- gotv) .> tol) == 0 "row $r: locations inconsistent with values"
    end
    return maxerr
end

# Engine vs an exact same-engine chunked reference over several shapes per
# engine (lt: ragged chunks + any padded batch height; mma: divisible w | N),
# both couts; each shape includes a single-row/padded-height case.  Then the
# real geometry (4096 x 2^15) for both engines: values AND locations within a
# few ulps, the dequantized-fp32 gemv spot check, a tail batch on a REUSED
# engine, and pipelined == sequential determinism.
function _check_engine_cgemm_topk(; k::Int = 20, seed = 123,
                                  engines = (:lt, :mma), couts = (:f16, :f32))
    CUDA.seed!(seed)
    lt_geo = ((64, 64, 208, 80), (32, 257, 144, 144), (64, 1, 4096, 512))
    mma_geo = ((64, 64, 256, 64), (128, 1, 2^12, 2^10))
    for engine in engines, cout in couts
        for (rdim, nb, N, w) in (engine === :lt ? lt_geo : mma_geo)
            D = rdim ÷ 2
            (Bre8, Bim8) = _rand_csplit_db(Float32, D, N)
            A = randn(Float16, nb, rdim) # the rope stream's output analogue
            eng = ComplexTopKEngine(Bre8, Bim8; k, w, rows_cap = nb, engine, cout)
            upload_fp16_as_f8c!(eng, A)
            D_val = CuMatrix{Float32}(undef, eng.rows_cap, k)
            D_loc = CuMatrix{Int32}(undef, eng.rows_cap, k)
            complex_gemm_topk!(D_val, D_loc, eng)
            (Crh, Cih) = _ref_ctopk_chunked(eng)
            _assert_ctopk_rows(Crh, Cih, Array(D_val), Array(D_loc), k)
            @info "  engine exact check OK (engine=$engine, cout=$cout, rdim=$rdim, " *
                  "nb=$nb, N=$N, w=$w, rows_cap=$(eng.rows_cap), k=$k)"
            eng = Bre8 = Bim8 = nothing
            GC.gc(); CUDA.reclaim()
        end
    end

    # real geometry: spot check vs dequantized fp32 gemv, tail batch on a
    # reused engine, pipelined == sequential
    rdim, nb, N, w = 128, 4096, 2^15, 2^13
    for engine in engines
        D = rdim ÷ 2
        (Bre8, Bim8) = _rand_csplit_db(Float32, D, N)
        eng = ComplexTopKEngine(Bre8, Bim8; k, w, rows_cap = nb, engine) # cout = :f16 fast path
        A = randn(Float16, nb, rdim)
        upload_fp16_as_f8c!(eng, A)
        D_val = CuMatrix{Float32}(undef, eng.rows_cap, k)
        D_loc = CuMatrix{Int32}(undef, eng.rows_cap, k)
        complex_gemm_topk!(D_val, D_loc, eng)
        (Crh, Cih) = _ref_ctopk_chunked(eng)
        _assert_ctopk_rows(Crh, Cih, Array(D_val), Array(D_loc), k)
        maxerr = _spot_check_complex(eng, D_val, D_loc, k, nb)
        # tail batch (100 < rows_cap) on the REUSED engine
        A2 = randn(Float16, 100, rdim)
        upload_fp16_as_f8c!(eng, A2)
        Dv2 = CuMatrix{Float32}(undef, eng.rows_cap, k)
        Dl2 = CuMatrix{Int32}(undef, eng.rows_cap, k)
        complex_gemm_topk!(Dv2, Dl2, eng)
        (Crh2, Cih2) = _ref_ctopk_chunked(eng)
        _assert_ctopk_rows(Crh2, Cih2, Array(Dv2), Array(Dl2), k)
        # sequential (no overlap) must be bitwise identical
        upload_fp16_as_f8c!(eng, A2)
        Dv3 = similar(Dv2)
        Dl3 = similar(Dl2)
        complex_gemm_topk!(Dv3, Dl3, eng; overlap = false)
        @assert Array(Dv3) == Array(Dv2) && Array(Dl3) == Array(Dl2) "overlap=false diverged"
        @info "  real-geometry check OK (engine=$engine, 4096 x 2^15, spot-check maxerr " *
              "$(round(maxerr; digits = 4)), tail batch, pipelined == sequential)"
        eng = Bre8 = Bim8 = nothing
        GC.gc(); CUDA.reclaim()
    end
    return nothing
end

# End-to-end: stream the fasta file through the full COMPLEX flow and compare
# the top-k against a float64 CPU reference that models the pipeline's
# quantization exactly: embeddings fp16 (the rope output) -> e4m3 (the stage
# input); the databases are compared AS STORED (dequantized from the device).
# Remaining deviation: fp16 storage of the two fp32-accumulated C components
# (~1 ulp each of |Re|,|Im| <= ~1) + accumulation-order noise -> abs2 error
# of a few 1e-3 -> atol 8e-3.
function _check_ctopk_stream(re, path, Bre8::CuMatrix{F8}, Bim8::CuMatrix{F8},
                             Bredq::Matrix{Float64}, Bimdq::Matrix{Float64};
                             k::Int, w::Int, batch_size::Int, rows_cap::Int,
                             normalize::Int, kfrag::Int, engine::Symbol = :lt,
                             refw::Dict{String,Vector{UInt32}},
                             sample::Int, seed = 13)
    rs = MersenneTwister(seed)
    N = size(Bre8, 2)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    nbatches = seen = 0
    heads_all = String[]
    for bat in complex_rope_topk_stream(re, path, Bre8, Bim8; k, w, batch_size,
                                        rows_cap, normalize, engine,
                                        err_out = err, tasks_out = tks)
        nbatches += 1
        nb = length(bat.heads)
        @assert bat isa ComplexTopKBatch && eltype(bat.vals) == Float32 "batch type/eltype mismatch"
        @assert size(bat.vals) == (nb, k) && size(bat.locs) == (nb, k) "batch shape mismatch"
        @assert size(bat.norms) == (re.m, nb) "norms passthrough shape mismatch"
        @assert bat.first == seen + 1 "batch first-index mismatch"
        for r in 1:nb
            @assert issorted(@view(bat.vals[r, :]); rev = true) "row $r: not sorted"
            @assert all(x -> x >= 0, @view(bat.vals[r, :])) "row $r: negative abs2 score"
            @assert all(lo -> lo in 1:N, @view(bat.locs[r, :])) "row $r: locations out of range"
        end
        if normalize == 0 # shared raw norm + partials: positive & finite
            @assert all(x -> isfinite(x) && x > 0, bat.norms) "norm passthrough not positive/finite"
        end
        atol = 8e-3
        for r in rand(rs, 1:nb, min(sample, nb))
            head = bat.heads[r]
            href, nrmref = _ref_rope_frag_real(_frag_codes(refw[head], kfrag), re; normalize)
            aq = Float64.(Float32.(F8.(Float16.(Float32.(href))))) # fp16 -> e4m3, as shipped
            scores = (Bredq' * aq) .^ 2 .+ (Bimdq' * aq) .^ 2 # (N,) fp64 reference
            refv = partialsort!(copy(scores), 1:k; rev = true)
            gotv = bat.vals[r, :]
            gotl = bat.locs[r, :]
            # norms passthrough vs the CPU reference (mode 0: only row 1 is the norm)
            for idm in (normalize == 0 ? (1:1) : (1:re.m))
                @assert isapprox(bat.norms[idm, r], nrmref[idm]; rtol = 1e-3) "norms passthrough mismatch ($(head), copy $idm)"
            end
            @assert isapprox(gotv, refv; atol, rtol = 0) "top-k abs2 values vs CPU reference mismatch ($(head))"
            @assert all(abs.(scores[gotl] .- gotv) .<= atol) "value/location inconsistency ($(head))"
        end
        append!(heads_all, bat.heads)
        seen += nb
    end
    foreach(wait, tks[]) # deterministic: all three stages have unwound
    @assert err[] === nothing "stream error: $(err[])"
    @assert seen == length(refw) "fragment count mismatch: $seen vs $(length(refw))"
    @assert nbatches == cld(seen, batch_size) "batch count mismatch ($nbatches for batch_size=$batch_size)"
    @assert sort(heads_all) == sort(collect(keys(refw))) "streamed heads mismatch"
    return nbatches
end

function run_e2e_check(; k::Int = 20, engines = (:lt, :mma))
    @show CUDA.name(device())
    @show nthreads()

    small, _big = ensure_data3()
    kfrag = 20_000
    dir = mktempdir(prefix = "e2ecomplex_")

    # ==========================================================================
    # A. THE PIECES: quantize/upload path, the COMPLEX segmented top-k kernel,
    #    gemmtopkfp8's GEMM unit tests (re-run in this process), and the complex
    #    engine vs exact references.
    # ==========================================================================
    @info "A. quantize path + complex segmented top-k kernel + fp8 engine checks"
    _check_quantize_path_c()
    test_seg_rowtopk_cmerge_kernel(k = k)
    # TODO(reorg): test_fp8_gemm (the fp8 GEMM + chunked-ldb unit test of the
    # legacy gemmtopkfp8.jl) is expected from the gemm/ fp8 layer (gemm/fp8_ptx.jl
    # or gemm/topk_kernels.jl) -- verify the provider when the layers land.
    test_fp8_gemm(engines = engines)
    _check_engine_cgemm_topk(k = k, engines = engines)

    # ==========================================================================
    # A2. THE LOADER: the split-form complex save reloads into the exact
    #     Bre/Bim pair (Bre = [re; im], Bim = [im; -re], bitwise).
    # ==========================================================================
    @info "A2. complex index loader / database derivation"
    tref = _ensure_e2h_test_ref()
    tbin = _e2h_build_index(tref)
    td = load_index_complex(tbin)
    (tBre, tBim) = index_database_complex(td)
    tD = td.m * 4^td.c
    @assert Array(tBre[1:tD, :]) == td.embeds8_re "Bre rows 1:D != embeds8_re"
    @assert Array(tBre[tD+1:2tD, :]) == td.embeds8_im "Bre rows D+1:2D != embeds8_im"
    @assert Array(tBim[1:tD, :]) == td.embeds8_im "Bim rows 1:D != embeds8_im"
    @assert Array(tBim[tD+1:2tD, :]) == .- td.embeds8_re "Bim rows D+1:2D != -embeds8_re"
    @info "  split save -> (Bre, Bim) derivation bitwise OK ($(size(tBre)))"
    tBre = tBim = td = nothing
    GC.gc(); CUDA.reclaim()

    # ==========================================================================
    # B. FLOW end-to-end on the small generated file (2^13 reads x ~20 kb)
    #    vs the float64 CPU reference: batch layouts (tail, exact fit, single
    #    batch), normalize 0/1, a second encoder config -- all on the :lt
    #    engine (the small test N = 3000 admits no mma-legal w).  The mma
    #    engine runs the same check at real geometry (N = 2^15, w = 2^13).
    #    The test databases are unit-modulus complex columns quantized to
    #    e4m3, like the production index.
    # ==========================================================================
    @info "B. complex_rope_topk_stream on the small file vs the float64 CPU reference"
    re = RopeEncoder(k = kfrag, s = 8, m = 4, c = 4) # rdim = 2^11
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k = kfrag, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == _V3_SMALL_READS "unexpected reference fragment count"

    D = re.m * 4^re.c
    N = 3000
    (Bre8, Bim8) = _rand_csplit_db(Float32, D, N)
    Bredq = Float64.(Float32.(Array(Bre8))) # the databases AS STORED (dequantized)
    Bimdq = Float64.(Float32.(Array(Bim8)))

    for batch_size in (3000, 2^13, 10^6) # tail, exact fit, single batch
        nb = _check_ctopk_stream(re, small, Bre8, Bim8, Bredq, Bimdq; k, w = 512,
                                 batch_size, rows_cap = min(batch_size, _V3_SMALL_READS),
                                 normalize = 0, kfrag, engine = :lt, refw, sample = 4)
        @info "  normalize=0, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_ctopk_stream(re, small, Bre8, Bim8, Bredq, Bimdq; k, w = 512,
                             batch_size = 3000, rows_cap = 3000, normalize = 1,
                             kfrag, engine = :lt, refw, sample = 4)
    @info "  normalize=1, batch_size=3000 OK ($nb batches)"

    # other encoder config through the whole flow (rdim = 2*1*4^4 = 512)
    re2 = RopeEncoder(k = kfrag, s = 5, m = 1, c = 4)
    D2 = re2.m * 4^re2.c
    (Bre82, Bim82) = _rand_csplit_db(Float32, D2, 777)
    nb = _check_ctopk_stream(re2, small, Bre82, Bim82,
                             Float64.(Float32.(Array(Bre82))),
                             Float64.(Float32.(Array(Bim82))); k, w = 512,
                             batch_size = 3000, rows_cap = 3000, normalize = 0,
                             kfrag, engine = :lt, refw, sample = 4)
    @info "  config (s=5, m=1, c=4) end-to-end OK ($nb batches)"
    Bre82 = Bim82 = nothing
    GC.gc(); CUDA.reclaim()

    # the mma engine end-to-end at real geometry (w | N, one exact batch)
    if :mma in engines
        (Bm_re, Bm_im) = _rand_csplit_db(Float32, D, 2^15)
        nb = _check_ctopk_stream(re, small, Bm_re, Bm_im,
                                 Float64.(Float32.(Array(Bm_re))),
                                 Float64.(Float32.(Array(Bm_im))); k, w = 2^13,
                                 batch_size = 2^13, rows_cap = 2^13, normalize = 0,
                                 kfrag, engine = :mma, refw, sample = 4)
        @info "  mma engine, real geometry (N=2^15, w=2^13), end-to-end OK ($nb batches)"
        Bm_re = Bm_im = nothing
        GC.gc(); CUDA.reclaim()
    end

    # ==========================================================================
    # C. teardown: closing the result stream early must unwind the top-k stage
    #    AND the whole rope pipeline quietly (no hang, no spurious err_out)
    #    and leave everything reusable.
    # ==========================================================================
    @info "C. early close"
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    ch = complex_rope_topk_stream(re, small, Bre8, Bim8; k, w = 512,
                                  batch_size = 3000, rows_cap = 3000,
                                  err_out = err, tasks_out = tks)
    got = 0
    for _ in ch
        got += 1
        got == 1 && break
    end
    close(ch)
    foreach(wait, tks[]) # deterministic: topk stage + gather + encode all done
    @assert err[] === nothing "early close raised: $(err[])"
    @assert got == 1
    @assert all(istaskdone, tks[]) && length(tks[]) == 3
    nb = _check_ctopk_stream(re, small, Bre8, Bim8, Bredq, Bimdq; k, w = 512,
                             batch_size = 3000, rows_cap = 3000, normalize = 0,
                             kfrag, engine = :lt, refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    # ==========================================================================
    # D. empty file -> zero batches, no error.
    # ==========================================================================
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    err = Ref{Any}(nothing)
    n = 0
    for _ in complex_rope_topk_stream(re, empty_fasta, Bre8, Bim8; k, w = 512,
                                      err_out = err)
        n += 1
    end
    @assert n == 0 && err[] === nothing
    @info "D. empty file OK (0 batches)"

    @info "ALL E2ECOMPLEX CORRECTNESS TESTS PASSED"
    return nothing
end

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end: build the 1M reference (once, shared with e2ehuman.jl), index
it in the COMPLEX split form (kstep = k/16), sample 1000 err = 0 reads, run
the FULL load -> (Bre, Bim) -> complex stream -> score path, and assert every
read is correctly mapped (its encoding is the true window's up to a global
phase rotation -- abs2 is invariant to the rotation and to quantization
noise; any miss means broken location bookkeeping, not noise).
"""
function run_e2e_test(; kfrag::Int = 20_000, n::Int = E2C_TEST_N, ktop::Int = 20,
                      w::Int = 2^16, batch_size::Int = 2^13, segs::Int = 8,
                      engine::Symbol = :lt, seed::Int = E2C_TEST_SEED)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index_complex(binfile)
    (Bre, Bim) = index_database_complex(d)
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(Bre, 2)
    w_test = min(w, prevpow(2, N)) # the tiny N (~1.5e3) admits no production w
    res = map_and_count(re, readsf, Bre, Bim, d.heads, d.starts, d.k; ktop,
                        w = w_test, batch_size, segs, engine)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped"
    @info "TINY E2E PASSED: $n/$n correctly mapped" first_hit_ranks = res.rank_hist
    Bre = Bim = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

"""
    run_e2e_save(; fasta, binfile, k, kstep, normalize, fp16, batch_size)

Build the COMPLEX index matrices for the reference `fasta` and serialize them
next to it in the SPLIT form (format "indexflowcomplex.fp8.v1"):
embeds8_re / embeds8_im (m*4^c, 2*n_frag) e4m3 columns -- a matrix of real
values together with a matrix of imaginary values -- plus the complex-modulus
norms and the heads/starts/strand location meta.  Defaults: k = 20000,
kstep = k/16 = 1250 (point 6), <fasta>.indexflowcomplex_fp8_ks<kstep>.bin.
The windows are indexreal's windows (same walk, same rc convention); the
encoding uses the production single-threaded reader + kernel via
`ref_rope_real_stream`.
"""
function run_e2e_save(; fasta::String = E2C_FASTA,
                      binfile::Union{String,Nothing} = nothing,
                      k::Int = E2C_K, kstep::Int = E2C_KSTEP, normalize::Int = 0,
                      fp16::Bool = true, batch_size::Int = 2^13)
    binfile === nothing && (binfile = E2C_INDEX_BIN)
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    @info "complex index save" fasta k kstep normalize fp16 batch_size layout = "split e4m3 columns (embeds8_re, embeds8_im)" rowdim = re.m * 4^re.c
    data = _build_complex_index_matrix(re, fasta; k, kstep, batch_size, normalize,
                                       fp16, progress = true)
    nt = (format = E2C_FORMAT, source = abspath(fasta), k, kstep, s = re.s,
          m = re.m, c = re.c, normalize, complex = true, fp8 = true,
          n_frag = data.n_frag, embeds8_re = data.embeds8_re,
          embeds8_im = data.embeds8_im, norms = data.norms, heads = data.heads,
          starts = data.starts, strand = data.strand)
    t = @elapsed save(nt, binfile)
    @info "saved $binfile ($(round(filesize(binfile) / 2^30; digits = 2)) GiB) in $(round(t; digits = 1)) s"
    load_index_complex(binfile) # structural verify: reload + shape asserts
    return binfile
end

# _build_complex_index_matrix (the production collection step of the COMPLEX
# save: count pass + ref_rope_real_stream fill) is layer-provided by
# index/complex.jl; run_e2e_save below calls it by name and keeps the
# NamedTuple assembly + Utils.save + reload-verify inline (the extraction to
# a save_complex_index helper is not clean from this body).

"""
    run_e2e_bench(; ktop, w, batch_size, segs, engine, cout, reps)

Stage timings of the production COMPLEX mapping flow: the complex index ->
resident (Bre, Bim) upload, the rope stage alone, per-chunk microbenches
(quantize+upload, the 2x fp8 GEMM, the complex segmented top-k), the
GEMM+top-k stage alone (compute-only: materialized embedding batches
replayed through the engine), and the full end-to-end pipeline (including
the ground-truth score).  GEMM accounting: 4*nreads*rdim*N FLOP (two GEMMs).
"""
function run_e2e_bench(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, cout::Symbol = :f16,
                       reps::Int = 1)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2C_READS) ||
        error("reads fasta not found: $E2C_READS (generate with reads/sample.jl, err = 0.15)")
    t_load = @elapsed d = load_index_complex(E2C_INDEX_BIN)
    t_B = @elapsed (Bre, Bim) = index_database_complex(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    (ds, dm, dc) = (d.s, d.m, d.c) # bench follows the index's encoder config
    @info "complex fp8 databases" dims = size(Bre) gib = round(2 * length(Bre) / 2^30; digits = 2) seconds = round(t_B; digits = 2) load_seconds = round(t_load; digits = 2)
    d = nothing
    GC.gc(); CUDA.reclaim()

    re = RopeEncoder(k = dk, s = ds, m = dm, c = dc)
    N = size(Bre, 2)
    rdim = size(Bre, 1)
    chunks = cld(N, w)

    @info "warming the reads fasta page cache..."
    read(E2C_READS)

    # ---- rope stage alone (no-op consumer) + materialize for the replay ----
    batches = RopeRealBatch{Float16}[]
    t_jit = @elapsed for nt in rope_encode_real_stream(re, E2C_READS; k = re.k,
                                                       batch_size,
                                                       normalize = 0, fp16 = true)
        push!(batches, nt)
    end
    nreads = sum(nt -> size(nt.embeds, 1), batches)
    bytes = filesize(E2C_READS)
    t_rope = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for nt in rope_encode_real_stream(re, E2C_READS; k = re.k, batch_size,
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
    eng = ComplexTopKEngine(Bre, Bim; k = ktop, w, rows_cap = batch_size, segs,
                            engine, cout)
    emb1 = batches[1].embeds
    @assert size(emb1) == (batch_size, rdim)
    Bre_p = byteptr(pointer(Bre))
    Bim_p = byteptr(pointer(Bim))
    CUDA.stream!(eng.sg) do
        upload_fp16_as_f8c!(eng, emb1)
        _gemm_chunk_c!(eng, eng.cr1, Bre_p, w) # warm (JIT + the lt heuristic)
        _gemm_chunk_c!(eng, eng.ci1, Bim_p, w)
    end
    CUDA.device_synchronize()
    tq = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            upload_fp16_as_f8c!(eng, emb1)
        end
    end
    tg = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            _gemm_chunk_c!(eng, eng.cr1, Bre_p, w)
            _gemm_chunk_c!(eng, eng.ci1, Bim_p, w)
        end
    end
    D_val = CuMatrix{Float32}(undef, batch_size, ktop)
    D_loc = CuMatrix{Int32}(undef, batch_size, ktop)
    launch_seg_rowtopk_cmerge!(D_val, D_loc, eng.cr1, eng.ci1, 0, ktop; segs) # warm (JIT)
    CUDA.device_synchronize()
    tt = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            launch_seg_rowtopk_cmerge!(D_val, D_loc, eng.cr1, eng.ci1, 0, ktop; segs)
        end
    end
    @printf("  quantize+upload: %.2f ms/batch | 2x fp8 gemm: %.2f ms/chunk (%.1f TFLOPS) | complex seg top-k: %.2f ms/chunk (%.0f GiB/s of C read)\n",
            tq * 1e3, tg * 1e3, 4.0 * batch_size * rdim * w / tg / 1e12,
            tt * 1e3, 2 * batch_size * w * sizeof(eltype(eng.cr1)) / tt / 2^30)

    # ---- compute-only: GEMM+top-k stage, rope excluded (replayed batches) --
    t_cmp = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for bat in complex_topk_flow(batches, eng; out_cap = 2)
                n += size(bat.vals, 1)
            end
            @assert n == nreads
        end
        t_cmp = min(t_cmp, t)
    end
    flops = 4.0 * nreads * rdim * N
    @printf("gemm+topk stage (quantize+H2D -> %d chunks/batch, gemm || topk): %.2f s  (%.1f TFLOPS sustained over 2^%.1f FLOP)\n",
            chunks, t_cmp, flops / t_cmp / 1e12, log2(flops))

    # ---- end-to-end: the full pipelined flow (and the mapping score) -------
    t_e2e = Inf
    correct = total = 0
    for _ in 1:reps
        res = map_and_count(re, E2C_READS, Bre, Bim, heads, starts, dk; ktop, w,
                            batch_size, segs, engine)
        t_e2e = min(t_e2e, res.seconds)
        correct = res.correct
        total = res.total
    end
    @printf("END-TO-END (fasta -> rope -> 2x fp8 gemm -> abs2 topk -> score): %.2f s  (%d/%d = %.2f%% correctly mapped)\n",
            t_e2e, correct, total, 100 * correct / total)
    hides = t_e2e < t_rope + t_cmp ? "hides completely behind the compute stage" : "is (partially) exposed"
    @printf("stage budget: rope-only %.2f s + compute-only %.2f s vs e2e %.2f s -> the rope stage %s\n",
            t_rope, t_cmp, t_e2e, hides)
    eng = Bre = Bim = batches = nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

"""
Production run: load the human COMPLEX index (whole matrix, fwd + rc, split
form), build the (Bre, Bim) database pair on the GPU, stream the sample
err = 0.15 reads through the complex flow (2x fp8 GEMM + abs2 top-k,
kstep = k/16 = 1250) and report THE number of correctly mapped records (plus
rate and first-hit rank histogram).
"""
function run_e2e_human(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2C_READS) ||
        error("reads fasta not found: $E2C_READS (generate with reads/sample.jl, err = 0.15)")
    @info "loading the human complex index" E2C_INDEX_BIN
    t_load = @elapsed d = load_index_complex(E2C_INDEX_BIN)
    @info "building the complex fp8 databases on GPU (Bre = [Re; Im], Bim = [Im; -Re])"
    t_B = @elapsed (Bre, Bim) = index_database_complex(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    @info "complex fp8 databases built" dims = size(Bre) gib = round(2 * length(Bre) / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # the ~10 GiB host index matrices can go
    GC.gc(); CUDA.reclaim()
    res = map_and_count(re, E2C_READS, Bre, Bim, heads, starts, dk; ktop, w,
                        batch_size, segs, engine, progress = true)
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
    Bre = Bim = nothing
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
    reps = parse(Int, getflag("reps", "1"))
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "save" && run_e2e_save()
    mode == "check" && run_e2e_check()
    mode == "test" && run_e2e_test(; ktop, w, batch_size = batch, segs, engine)
    mode == "bench" && run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps)
    mode == "run" && run_e2e_human(; ktop, w, batch_size = batch, segs, engine)
    mode == "all" && (run_e2e_test(; ktop, w, batch_size = batch, segs, engine);
                      run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps);
                      run_e2e_human(; ktop, w, batch_size = batch, segs, engine))
    mode in ("save", "check", "test", "bench", "run", "all") ||
        error("unknown mode $mode (use save|check|test|bench|run|all)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
