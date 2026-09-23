# ==============================================================================
# e2ecomplex.jl -- END-TO-END HUMAN-GENOME MAPPING on the COMPLEX flow: the
# rope encodings are treated as what they are -- COMPLEX vectors z in C^D
# (D = m*4^c, bin = (csmer-1)*m + idm) -- the GEMM stage computes the COMPLEX
# inner products <z_query, z_db> with TWO real fp8 GEMMs, and the top-k stage
# ranks by the inner-product MAGNITUDE |<z_query, z_db>|^2 (abs2; ranking by
# |.| is equivalent).  Every read's top-k hits are checked against its
# PROVENANCE (ground truth), exactly as in e2ehuman.jl.
#
# The REAL flow (e2ehuman.jl / flowtopkfp8.jl) is UNTOUCHED: this file adds
# the complex variant ALONGSIDE it -- new names, own file (point 5 of the
# spec).  It includes indexflowreal.jl (=> ropeflowreal/fastareads_v3/
# ropeflow_v3 transitively: the encoding stream AND the single-threaded
# reference reader + save machinery) and gemmtopkfp8.jl (the fp8 GEMM
# engines, F8, the converters); flowtopkfp8.jl is NOT included (the
# one-variant-per-session rule -- the complex engine is a separate struct,
# not an option inside the real one).
#
# THE COMPLEX INNER PRODUCT VIA TWO REAL FP8 GEMMs (point 1)
#   The real flow's split layout [Re(z); Im(z)] IS the complex vector in
#   split form, so the ENCODING stage is shared verbatim (the rope kernel's
#   norms are complex moduli already).  What changes is the SCORE.  For
#   z_a = ar + i*ai, z_b = br + i*bi:
#     <z_a, z_b> = Sum conj(z_a_i) z_b_i
#                = (ar*br + ai*bi) + i*(ar*bi - ai*br)
#   Both components are real dot products over the split halves:
#     Re = a_split . [br; bi]  = a_split . Bre[:, j]   Bre = [Yr; Yi]  --
#       exactly the real flow's database (row-stacked [Re; Im] columns);
#     Im = a_split . [bi; -br] = a_split . Bim[:, j]   Bim = [Yi; -Yr] --
#       the split of -i*z_b: the SAME database rotated by -90 degrees.
#   So the complex flow runs the SAME fp8 GEMM as the real flow (M x 2D x w,
#   same engines, same quantize path) TWICE per chunk -- once against Bre,
#   once against Bim -- and scores
#     score(r, j) = Cr[r, j]^2 + Ci[r, j]^2 = |<z_r, z_j>|^2.
#   Rationale: a read sampled within kstep/2 of an index window is a SHIFTED
#   copy of it, so its encoding is (approximately) the window's encoding
#   rotated by a GLOBAL phase e^(i*theta*d), d = start offset difference:
#   the real flow's score decays as cos(theta*d) while abs2 is phase
#   invariant -- the complex magnitude is the right statistic for shifted
#   matching windows.  This test runs kstep = k/16 = 1250 (d up to 625,
#   theta*d <= 2*pi*625/20000 ~ 0.2 rad) and err = 0.15 reads (point 6).
#
# NORMS (point 2)
#   With complex vectors the norm is the MODULUS.  The encoding kernel's
#   norms already are (abs2-based complex moduli, normalize modes 0-3 --
#   ropeflowreal, unchanged), so normalize = 0 still means unit energy
#   |z|^2 = Sum(re^2 + im^2) = 1 == the squared norm of the split row: the
#   "normalize = 0 => normalized database columns" assumption of e2ehuman
#   carries over (asserted at load).  The stage that DOES change is the
#   match statistic: TopKBatch.vals are now |<z_query, z_db>|^2 in [0, ~1]
#   (were raw real dot products), computed FUSED in the top-k scan (the
#   complex kernel reads the Re and Im C chunks per column and inserts the
#   abs2 score into the Float32 reservoir; all sentinels stay
#   typemin(Float32), scores are >= 0).
#
# INDEX (points 3+4)
#   The complex index save is a NEW format "indexflowcomplex.fp8.v1"
#   (<fasta>.indexflowcomplex_fp8_ks<kstep>.bin): the complex embeddings are
#   stored SPLIT -- embeds8_re (D, 2*n_frag) and embeds8_im (D, 2*n_frag)
#   e4m3 columns (fwd block 1:n, rc block n+1:2n, the exact F8.(Float32.(.))
#   quantization of the fp16 kernel output) -- a matrix of real values
#   together with a matrix of imaginary values, NEVER Matrix{Complex}
#   (Utils.save of two plain real matrices).  norms (m, 2n) Float32 and the
#   heads/starts/strand location meta travel along (the windows are the
#   real index's windows).  load_index ALSO accepts the legacy
#   "indexflowreal.fp8.v1" save (the real flow's row-stacked [Re; Im]):
#   Bre = embeds8 verbatim (pure H2D) and Bim derived ON THE GPU per column
#   block (row-half swap + negate; fp8 negation is exact).  Either way
#   index_database(d) -> (Bre, Bim), both (2D, 2*n_frag) resident e4m3 --
#   the GPU footprint is 2x the real flow's database plus 2x its C chunk
#   buffers (four (rows_cap, w) buffers for the double-buffered Re/Im pairs).
#
# FLOW (per batch, the complex stage)
#   H2D(fp16 [Re; Im] rows) + e4m3 quantize (same as the real flow),
#   then per B-column chunk [GEMM(Cr, Bre chunk) ; GEMM(Ci, Bim chunk)] on
#   the GEMM stream || complex seg-rowtop-k on the top-k stream (reads Cr
#   and Ci, inserts abs2 scores) x cld(N, w) chunks, then D2H of the merged
#   (nb x k) abs2/location rows.  locs are GLOBAL 1-based column ids (the
#   complex window id -- the SAME id for both chunks).  Engines: :lt
#   (cuBLASLt fp8, the production default; the human N is not a power of
#   two, so :mma's w | N restriction rules it out) and :mma; cout :f16/:f32.
#
# DESIGN
#   1. Whole-matrix load: fwd + rc columns of BOTH databases (the .bin
#     stores the same head/start for both strands; an rc hit is the same
#     genomic location -- no strand special-casing in the scoring).
#   2. normalize = 0 ASSUMED (asserted): unit-MODULUS complex columns;
#     the streamed reads use the same normalize mode.
#   3. Provenance parsing / intersection scoring: verbatim e2ehuman.jl
#     (`>read_<i> start=<s> len=<k> src=<record header sans '>'`; a read is
#     CORRECTLY MAPPED iff its true window [start, start+len-1] in record
#     src intersects >= 1 of the ktop returned database windows).
#   4. Shared runner: `map_and_count` is the whole scoring loop; the tiny
#     self-test and the production run differ only in the inputs.
#
# SELF-TEST (mode `test`; production files not needed)
#   The SAME tiny fixture REFERENCE as e2ehuman.jl (1M-base two-record
#   reference, built once and shared via fastareads_v3's data dir; each
#   script builds its own format-specific .bin) -> a COMPLEX split-form
#   index (format "indexflowcomplex.fp8.v1", kstep = k/16) -> 1000 err = 0
#   reads -> ALL must map (at rank 1): an err = 0 read shares >= 95% of its
#   bases with an index window AND its encoding is the window's up to a
#   global phase rotation -- abs2 is invariant to both quantization noise
#   and the rotation, so a miss means broken bookkeeping, not noise.
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16; index = the COMPLEX
# split-form save of GRCh38.p14 at kstep = k/16 = 1250: 2,475,311 windows x2
# strands = 4,950,622 columns x rdim 2048, 67 unique records; reads = the
# indexsample n=131,072 k=20,000 err=0.15 s=42 file; k(top) = 20, w = 2^16,
# batch = 2^13 -> 16 batches, 76 chunks/batch, engine :lt):
#
#   complex index save (mode `save`): count pass 0.29 s, stream 34.4 s,
#     the 9.57 GiB split-form .bin (embeds8_re + embeds8_im) written in 1.9 s
#   index load 2.1 s; (Bre, Bim) build 5.5 s -> 18.9 GiB resident e4m3
#     (2 x 2048 x 4,950,622; the legacy real-fp8 load derives Bim on GPU)
#   MAPPING FLOW STAGE SPEEDS (bench mode, --reps=1, first JIT pass excluded;
#   the rope stage is task-pipelined and hides behind the compute stage):
#     rope stage (fasta read + 2-bit pack + rope encode)   0.60 s   4.41 GB/s
#       quantize+upload (fp16 H2D + e4m3 convert)          2.04 ms/batch
#       2x fp8 GEMM per chunk (M = 2^13, K = 2048, w = 2^16)
#                                                          6.12 ms/chunk
#                                                          718.4 TFLOPS over
#                                                          the pair (the real
#          flow's per-GEMM rate: 731.5 -- the complex pair runs at the same
#          per-GEMM speed, same shapes, lt back-to-back)
#       complex seg rowtop-k (abs2 of fp16 Re/Im chunks, segs = 8)
#                                                          2.40 ms/chunk
#                                                          834 GiB/s of C
#                                                          read (2 chunks/row)
#     gemm+topk stage (compute-only, 76 chunks/batch x 16)  11.62 s
#       457.4 TFLOPS sustained over 2^52.2 FLOP (4*nreads*rdim*N; GEMM floor
#       ~7.4 s + exposed top-k ~2.9 s)
#     END-TO-END (fasta -> rope -> 2x fp8 gemm -> abs2 topk -> score)
#       11.47 s   103,352/131,072 = 78.85% correctly mapped
#       (a JIT-inclusive first pass in `run` mode measured 15.8 s; the bench
#       e2e is the warmed number, rope hidden, min over reps)
#
#   ACCURACY at err = 0.15 (the point-6 configuration): 78.85% (rank-1:
#   80,110; unmapped 27,720; the ranks 2..20 tail carries ~17.7 pp of
#   recall).  Comparison -- the REAL flow (e2ehuman.jl) at err = 0.15:
#   77,943/131,072 = 59.47% (rank-1 54,978) at its kstep = 2000 index.  The
#   +19.4 pp come from BOTH changes of this test's configuration: the denser
#   kstep = 1250 index (every read start within 625 of a window start, so
#   ~97% shared bases before mutations, vs within 1000 at kstep 2000) AND
#   the phase-invariant abs2 score (a read shifted by d bases from its best
#   window is a global phase rotation e^(i*theta*d), theta*d <= 0.2 rad here:
#   Re<.,.> decays by up to ~2% while abs2 does not) -- the complex statistic
#   is the one that exploits the denser index fully.
#   Tiny self-test (mode `test`): 1000/1000 err = 0 reads mapped, all at
#   rank 1 -- abs2 is invariant to the shift rotation AND the e4m3 noise, so
#   the true window wins exactly (the real flow's same fixture: also 1000/1000).
#
# CORRECTNESS (mode `check`, all on kau, julia -t 16): e4m3 quantize/upload
# == host F8.() bitwise, pad rows zeroed; seg_rowtopk_cmerge_kernel! matches
# the CPU abs2 multiset within a few ulps (FMA contraction) over Tc =
# Float16/Float32 x ragged/strided/w==k/empty-segment/single-row shapes;
# gemmtopkfp8's GEMM unit tests re-run in-process; the complex engine vs the
# exact same-engine chunked reference (top-k values AND locations) for both
# engines x both couts over ragged/padded/single-row shapes + real geometry
# (4096 x 2^15, spot-check maxerr 0.0154 lt / 0.0158 mma vs the
# dequantized-fp32 gemv reference), tail batch on a reused engine, pipelined
# == sequential bitwise; the split save -> (Bre, Bim) derivation is BITWISE
# (Bre = [re; im], Bim = [im; -re], fp8 negation exact); end-to-end stream on
# the 164 MB file vs the float64 CPU reference modelling fp16->e4m3 exactly:
# batch layouts (tail/exact/single), normalize 0/1, config (5,1,4), mma at
# real geometry, values + locations + complex-modulus norms passthrough,
# early-close teardown (3 tasks), empty file.  ALL PASS.
#
# Run modes (first non-flag ARGV[1]):
#   save  build the complex human index (kstep = k/16 = 1250 default) ->
#         <fasta>.indexflowcomplex_fp8_ks<kstep>.bin (split-form save)
#   check correctness: quantize path, complex segmented top-k kernel unit
#         tests, engine checks vs same-engine chunked references + a
#         dequantized-fp32 gemv spot check (both engines x couts), loader/
#         derivation bitwise checks, end-to-end stream vs the float64 CPU
#         reference (batch layouts, normalize 0/1, second encoder config,
#         mma at real geometry), early-close teardown, empty file
#   test  tiny end-to-end (1M reference -> complex index -> err=0 reads ->
#         all mapped)
#   bench stage timings on the human index + reads (rope-only, quantize+
#         upload, 2x fp8 GEMM, complex seg top-k, compute-only, end-to-end)
#   run   production: human complex index + n131072_k20000_e0.15 reads ->
#         mapping score (kstep = 1250, err = 0.15 -- point 6)
#   all   test + bench + run (default)
#   flags: --k=20 (top-k size), --w=65536 (B-column chunk), --batch=8192,
#          --segs=8, --engine=lt (mma needs w | N; :lt is the production
#          engine), --reps=1
#   env:   E2COMPLEX_FASTA (reference fasta)
#          E2COMPLEX_INDEX (index .bin; default
#          <fasta>.indexflowcomplex_fp8_ks<kstep>.bin; the legacy
#          indexflowreal.fp8.v1 layout loads via the GPU-derived path)
#          E2COMPLEX_READS (reads fasta; default
#          <fasta>.indexsample_n131072_k20000_e0.15_s42.fasta)
#          E2COMPLEX_K / E2COMPLEX_KSTEP (save/run fragment length and step)
#
# USAGE
#   julia --project=. -t 16 test/e2ecomplex.jl save    # once (index build)
#   julia --project=. -t 16 test/e2ecomplex.jl check
#   julia --project=. -t 16 test/e2ecomplex.jl test
#   julia --project=. -t 16 test/e2ecomplex.jl run --engine=lt
#   julia --project=. -t 16 test/e2ecomplex.jl         # test + bench + run
# ==============================================================================

include(joinpath(@__DIR__, "indexflowreal.jl")) # the reference reader + save
# machinery (ref_rope_real_stream, _count_ref_windows, _pack_record!,
# revcomp_frag_words!) AND, transitively, ropeflowreal (RopeRealBatch,
# rope_encode_real_stream, encode_frag_real_batch!, _ref_rope_frag_real, the
# kernel), ropeflow_v3 (_ref_rope_frag, RopeEncoder, fasta_reads,
# _frag_codes, _timed_min/_warm_cache/_reps) and fastareads_v3 (_v3_data_dir,
# _ref_forward_pack, _next_rec3, _LUT3, Utils)

include(joinpath(@__DIR__, "gemmtopkfp8.jl")) # fp8_gemm_mma! / fp8_gemm_lt!,
# build_kernel + k_gemm, f32_to_f8! / f8_to_f32!, randn_fp8!, spot_check,
# _tup_insert!/_tup_replace, TOPK_UNROLL, F8, byteptr, test_fp8_gemm (also
# runs CUDA.allowscalar(false))

using Random
using CUDA
using Printf
using Mmap
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides)
# ------------------------------------------------------------------------------
const E2C_FASTA = get(ENV, "E2COMPLEX_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
const E2C_K = parse(Int, get(ENV, "E2COMPLEX_K", "20000"))        # fragment length
const E2C_KSTEP = parse(Int, get(ENV, "E2COMPLEX_KSTEP", string(E2C_K ÷ 16))) # k/16 = 1250
# the complex index: the split-form save of THIS script (the legacy real fp8
# layout loads too -- see load_index / index_database)
const E2C_INDEX_BIN = get(ENV, "E2COMPLEX_INDEX",
                          string(splitext(E2C_FASTA)[1],
                                 ".indexflowcomplex_fp8_ks", E2C_KSTEP, ".bin"))
# the mutated reads (indexsample `gen` output): err = 0.15 (point 6)
const E2C_READS = get(ENV, "E2COMPLEX_READS",
                      string(splitext(E2C_FASTA)[1],
                             ".indexsample_n131072_k20000_e0.15_s42.fasta"))

const E2C_FORMAT = "indexflowcomplex.fp8.v1"   # the split-form complex save
const E2C_LEGACY_FP8_FORMAT = "indexflowreal.fp8.v1" # the real flow's save:
# Bre = embeds8 verbatim, Bim derived on the GPU (swap + negate)

# tiny self-test fixture (the SAME cached 1M reference as e2ehuman.jl)
const E2C_TEST_BASES = 1_000_000
const E2C_TEST_N = 1000
const E2C_TEST_SEED = 42

# ------------------------------------------------------------------------------
# Ground truth: indexsample's provenance headers (verbatim e2ehuman.jl)
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

# ==============================================================================
# The batched complex fp8 GEMM + top-k engine: the real flow's engine with a
# SECOND database (Bim) and the fused abs2 top-k; every C chunk is an (Re, Im)
# PAIR of same-shape buffers double-buffered together.
# ==============================================================================

"""
One streamed batch of COMPLEX top-k results.  `vals[r, :]` are fragment r's k
best scores against the database, DESCENDING -- the scores are
`|<z_query, z_db>|^2 = Re^2 + Im^2` in Float32 (the kernel's reservoir
currency; `>= 0`, `<= ~1` for normalize = 0 data).  `locs[r, :]` the matching
GLOBAL 1-based column ids (complex window ids into Bre/Bim).  `norms`/`heads`/
`first` are the rope batch's passthrough (per-copy complex-modulus norms
(m, nb) Float32, fasta headers, 1-based global first-fragment index).  Fresh
HOST matrices -- retain freely.
"""
struct ComplexTopKBatch
    vals::Matrix{Float32}      # (nb, k) |<z_query, z_db>|^2, descending
    locs::Matrix{Int32}        # (nb, k), global column ids into Bre/Bim
    norms::Matrix{Float32}     # (m, nb), passthrough
    heads::Vector{String}
    first::Int
end

# batch-height multiple required by the GEMM engines (same as the real flow)
padmul(engine::Symbol) = engine === :mma ? 64 : 16

"""
Reusable per-batch COMPLEX fp8 GEMM+top-k state: the real flow's engine with
the second database and a FOUR-buffer C pool -- `cr1/cr2` hold the Re chunks,
`ci1/ci2` the paired Im chunks (double-buffered together: chunk i uses
`mod1(i, 2)`).  The batch staging `a8`/`a32` and all engine restrictions are
the real flow's verbatim (the A operand is the same M x 2D split matrix --
the complex flow changes the DATABASES and the top-k, not the GEMM).
`rows_cap` is rounded up to `padmul(engine)`; every batch computes the full
height and the pad rows are sliced off after the D2H.  Requires
1 <= k <= w <= N.  Compiles the C++ mma kernel once (nvcc -> PTX -> CuModule).
"""
struct ComplexTopKEngine{Tc<:Union{Float16,Float32}}
    Bre::CuMatrix{F8}          # (2D, N) = [Yr; Yi]: the real flow's database
    Bim::CuMatrix{F8}          # (2D, N) = [Yi; -Yr]: the split of -i*z (the
                               #   -90deg-rotated database)
    cr1::CuMatrix{Tc}          # the two (rows_cap, w) Re chunk buffers,
    cr2::CuMatrix{Tc}          #   double-buffered: chunk i uses mod1(i, 2)
    ci1::CuMatrix{Tc}          # the paired Im chunk buffers
    ci2::CuMatrix{Tc}
    a8::CuMatrix{F8}           # (rows_cap, 2D) quantized batch staging
    a32::CuMatrix{Float32}     # its fp32 scratch (pad rows kept zero -- see
                               # upload_fp16_as_f8c!'s linear-layout note)
    k::Int                     # top-k size
    w::Int                     # B-column chunk width
    engine::Symbol             # :mma (C++ mma.sync kernel) | :lt (cuBLASLt)
    cout::Symbol               # C chunk type: :f16 | :f32
    rows_cap::Int              # padded batch-height capacity (a padmul multiple)
    sg::CuStream               # GEMM stream
    st::CuStream               # top-k stream
    evg::NTuple{2,CuEvent}     # evg[b]: gemms of the chunk using buffer pair b done
    evt::NTuple{2,CuEvent}     # evt[b]: top-k of the chunk using buffer pair b done
    segs::Int                  # column segments per row for the top-k kernel
end

function ComplexTopKEngine(Bre::CuMatrix{F8}, Bim::CuMatrix{F8};
                           k::Int = 20, w::Int = 2^16, rows_cap::Int = 2^13,
                           segs::Int = 8, engine::Symbol = :lt, cout::Symbol = :f16)
    rdim, N = size(Bre)
    size(Bim) == (rdim, N) ||
        throw(ArgumentError("Bre/Bim must have the same shape (got " *
                            "$(size(Bre)) vs $(size(Bim)))"))
    1 <= k <= w <= N ||
        throw(ArgumentError("need 1 <= k <= w <= N (got k = $k, w = $w, N = $N)"))
    engine in (:mma, :lt) || throw(ArgumentError("engine must be :mma or :lt"))
    cout in (:f16, :f32) || throw(ArgumentError("cout must be :f16 or :f32"))
    if engine === :mma
        rdim % 32 == 0 || throw(ArgumentError("mma kernel needs K % 32 == 0 (K = $rdim)"))
        w % 64 == 0 || throw(ArgumentError("mma kernel needs w % 64 == 0 (w = $w)"))
        N % w == 0 ||
            throw(ArgumentError("engine :mma needs w to divide N (no ragged final chunk)"))
    else
        rdim % 16 == 0 ||
            throw(ArgumentError("cuBLASLt fp8 needs K % 16 == 0 (K = $rdim)"))
    end
    rows_cap = cld(rows_cap, padmul(engine)) * padmul(engine)
    isempty(k_gemm) && build_kernel() # nvcc -> PTX -> CuModule (once, lazily)
    Tc = cout === :f16 ? Float16 : Float32
    cr1 = CuMatrix{Tc}(undef, rows_cap, w)
    cr2 = CuMatrix{Tc}(undef, rows_cap, w)
    ci1 = CuMatrix{Tc}(undef, rows_cap, w)
    ci2 = CuMatrix{Tc}(undef, rows_cap, w)
    a8 = CuMatrix{F8}(undef, rows_cap, rdim)
    a32 = CUDA.zeros(Float32, rows_cap, rdim) # zeroed once; pad rows stay zero
    sg = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    st = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    ComplexTopKEngine(Bre, Bim, cr1, cr2, ci1, ci2, a8, a32, Int(k), Int(w),
                      engine, cout, rows_cap, sg, st,
                      (CuEvent(), CuEvent()), (CuEvent(), CuEvent()), Int(segs))
end

"""
H2D-upload a host fp16 embedding batch (the split [Re; Im] rows -- the A
operand is IDENTICAL to the real flow's) and quantize it to e4m3 (satfinite)
into rows 1:nb of `a8`.  LINEAR-LAYOUT CONTRACT (the real flow's, verbatim):
the converter is 1-D over the WHOLE (cap, rdim) staging, so `a32` must have
exactly `a8`'s shape, and rows nb+1:cap of `a32` must be ZERO (they pass
through the converter into `a8`'s pad rows, keeping the pad score rows exact
zeros).  The engine's `a32` is zeroed once at construction and only rows
1:nb are ever overwritten.  Runs on the current stream; `complex_gemm_topk!`
device-synchronizes before its chunk loop, which orders it before the first
GEMM.
"""
function upload_fp16_as_f8c!(a8::CuMatrix{F8}, a32::CuMatrix{Float32},
                             embeds::Matrix{Float16})
    isempty(k_gemm) && build_kernel() # the converter kernels live in the same PTX
    nb, rdim = size(embeds)
    cap = size(a8, 1)
    size(a8, 2) == rdim ||
        throw(ArgumentError("embeds dim $rdim != staging rows $(size(a8, 2))"))
    size(a32) == (cap, rdim) ||
        throw(ArgumentError("fp32 scratch must match the staging shape ($(size(a8, 1)), $rdim)"))
    nb <= cap || throw(ArgumentError("batch height $nb > staging capacity $cap"))
    a16 = CuMatrix{Float16}(undef, nb, rdim)
    copyto!(a16, embeds)                 # contiguous H2D (~32 MiB, pageable src)
    @view(a32[1:nb, :]) .= a16           # fused fp16 -> fp32 broadcast
    f32_to_f8!(byteptr(pointer(a8)), pointer(a32), cap * rdim) # full height, 1-D
    a16 = nothing                        # device pool reuses it next batch
    return a8
end

upload_fp16_as_f8c!(eng::ComplexTopKEngine, embeds::Matrix{Float16}) =
    upload_fp16_as_f8c!(eng.a8, eng.a32, embeds)

# GEMM dispatch for one chunk of ONE database (the caller launches it twice:
# Re chunk from Bre, Im chunk from Bim -- same shapes, same engines).
@inline function _gemm_chunk_c!(eng::ComplexTopKEngine, buf::CuMatrix,
                                Bp::CuPtr{UInt8}, wi::Integer)
    if eng.engine === :mma
        fp8_gemm_mma!(eng.cout, buf, eng.a8, Bp, wi, size(eng.Bre, 1))
    else
        fp8_gemm_lt!(buf, eng.a8, Bp, wi)
    end
    return buf
end

"""
One streamed batch: the real flow's double-buffered chunk loop run against
BOTH databases (two GEMM launches per chunk on the GEMM stream, one fused
abs2 top-k on the top-k stream), with the running top-k RESET first (batches
are independent query sets).  The quantized batch must already be staged in
`eng.a8` (run `upload_fp16_as_f8c!(eng, embeds)` first); the FULL rows_cap
height is computed -- `D_val`/`D_loc` are (rows_cap, k) and rows
nb+1:rows_cap are discarded by the caller.  `overlap = false` runs
sequentially on the default stream (benchmark baseline; bitwise identical
results).  Returns with the device synchronized.
"""
function complex_gemm_topk!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                            eng::ComplexTopKEngine; overlap::Bool = true)
    M, KA = size(eng.a8)
    rdim, N = size(eng.Bre)
    @assert KA == rdim "A's inner dim must match the databases' rows"
    @assert size(eng.Bim) == (rdim, N) "database shape mismatch"
    @assert M == eng.rows_cap "engine staging must be (rows_cap, rdim)"
    @assert size(D_val) == (M, eng.k) && size(D_loc) == (M, eng.k)

    fill!(D_val, typemin(Float32)) # reset the running top-k (scores are >= 0)
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    sg = overlap ? eng.sg : CUDA.default_stream()
    st = overlap ? eng.st : sg
    for i in 1:cld(N, eng.w)
        b = mod1(i, 2)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        cr = b == 1 ? eng.cr1 : eng.cr2
        ci = b == 1 ? eng.ci1 : eng.ci2
        # base pointers of the Bre/Bim column lo; the column stride stays rdim
        Bre_p = byteptr(pointer(eng.Bre), (lo - 1) * rdim)
        Bim_p = byteptr(pointer(eng.Bim), (lo - 1) * rdim)

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(eng.evt[b])   # buffer pair b's previous top-k done
            _gemm_chunk_c!(eng, cr, Bre_p, wi) # Cr = A * Bre chunk (Re parts)
            _gemm_chunk_c!(eng, ci, Bim_p, wi) # Ci = A * Bim chunk (Im parts)
            CUDA.record(eng.evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b])            # both C chunks are ready
            crv = wi == eng.w ? cr : @view(cr[:, 1:wi])
            civ = wi == eng.w ? ci : @view(ci[:, 1:wi])
            launch_seg_rowtopk_cmerge!(D_val, D_loc, crv, civ,
                                       lo - 1, eng.k; segs = eng.segs)
            CUDA.record(eng.evt[b])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

# ==============================================================================
# The COMPLEX segmented per-row top-k kernel: flowtopkfp8's
# seg_rowtopk_merge_kernel! with the scan value changed from the raw C entry
# to the FUSED abs2 score of the (Re, Im) chunk pair:
#     score(j) = Float32(Cr[row, j])^2 + Float32(Ci[row, j])^2
# (the norm/magnitude computation of the complex inner product -- point 2 of
# the spec).  Everything else -- PS threads per 32-row tile, the register
# Float32 reservoir with tracked minimum, the software-pipelined
# threshold-filtered scan, the sort + PS-way merge mechanics, the sentinels
# at typemin(Float32) -- is verbatim (scores are >= 0, so the sentinels stay
# correct).  NB: the GPU may contract r*r + i*i into mul+FMA -- tests compare
# values with a few-ulp tolerance (and cross-check locations via values).
# ==============================================================================

@inline function _cscore(::Type{Tc}, Cr, Ci, row, col) where {Tc}
    @inbounds r = Cr[row, col]
    @inbounds im = Ci[row, col]
    rf = Float32(r)
    imf = Float32(im)
    return rf * rf + imf * imf
end

function seg_rowtopk_cmerge_kernel!(out_vals::CuDeviceMatrix{Float32},
                                    out_inds::CuDeviceMatrix{Int32},
                                    Cr, Ci, offset::Int32, ::Val{K}, ::Val{PS}) where {K, PS}
    tid = threadIdx().x
    mt = ((tid - Int32(1)) & Int32(31)) + Int32(1) # row within the block's 32-row tile
    s0 = (tid - Int32(1)) >> Int32(5)              # this thread's segment (0-based)
    row = (blockIdx().x - Int32(1)) * Int32(32) + mt
    M = size(Cr, 1)
    valid = row <= M
    U = TOPK_UNROLL

    width = size(Cr, 2)
    segw = cld(width, PS)                        # segment width in columns
    seg_lo = s0 * segw + 1                       # my segment's first column (1-based)
    count = max(0, min(width, (s0 + 1) * segw) - seg_lo + 1)  # 0: empty segment

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(Float32), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(Float32)
    min_pos = 1

    if valid && count > 0
        Tc = eltype(Cr)
        # scan my segment: software-pipelined, threshold-filtered; each
        # "value" is the abs2 score of one (Cr, Ci) column pair; the
        # insertion index is segment-local, mapped to a global column below
        c = 1
        @inbounds begin
            n1 = count >= 1 ? _cscore(Tc, Cr, Ci, row, seg_lo) : typemin(Float32)
            n2 = count >= 2 ? _cscore(Tc, Cr, Ci, row, seg_lo + 1) : typemin(Float32)
            n3 = count >= 3 ? _cscore(Tc, Cr, Ci, row, seg_lo + 2) : typemin(Float32)
            n4 = count >= 4 ? _cscore(Tc, Cr, Ci, row, seg_lo + 3) : typemin(Float32)
        end
        @inbounds while c <= count
            # issue the whole next batch of U (paired) loads before processing
            m1 = c + U <= count     ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U - 1) : typemin(Float32)
            m2 = c + U + 1 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U)     : typemin(Float32)
            m3 = c + U + 2 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U + 1) : typemin(Float32)
            m4 = c + U + 3 <= count ? _cscore(Tc, Cr, Ci, row, seg_lo + c + U + 2) : typemin(Float32)
            if n1 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n1, Int32(c), min_pos); end
            if n2 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n2, Int32(c + 1), min_pos); end
            if n3 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n3, Int32(c + 2), min_pos); end
            if n4 > thresh; (lk, li, thresh, min_pos) = _tup_insert!(lk, li, n4, Int32(c + 3), min_pos); end
            n1 = m1; n2 = m2; n3 = m3; n4 = m4
            c += U
        end

        # sort the reservoir descending (once per thread)
        @inbounds for i in 2:K
            v = lk[i]
            d = li[i]
            j = i
            while j > 1 && lk[j-1] < v
                lk = _tup_replace(lk, j, lk[j-1]);  li = _tup_replace(li, j, li[j-1])
                lk = _tup_replace(lk, j - 1, v);    li = _tup_replace(li, j - 1, d)
                j -= 1
            end
        end
    end

    # shared memory: staged segment lists + heads, preloaded running list
    o = 0
    shv = CuDynamicSharedArray(Float32, (PS, 32, K));      o += PS * 32 * K * sizeof(Float32)
    shi = CuDynamicSharedArray(Int32, (PS, 32, K), o);     o += PS * 32 * K * sizeof(Int32)
    shp = CuDynamicSharedArray(Int16, (PS, 32), o);        o += PS * 32 * sizeof(Int16)
    shr_v = CuDynamicSharedArray(Float32, (32, K), o);     o += 32 * K * sizeof(Float32)
    shr_i = CuDynamicSharedArray(Int32, (32, K), o)        # (<= 48 KiB total for K=20)

    # stage my segment's sorted list (sentinels when the row/segment is empty)
    if valid
        @inbounds for j in 1:K
            shv[s0 + 1, mt, j] = lk[j]
            shi[s0 + 1, mt, j] = li[j]
        end
    end
    # preload the running top-K before any output write (merge writes can lap
    # the read cursor)
    if valid && tid <= Int32(32)
        @inbounds for j in 1:K
            shr_v[mt, j] = out_vals[row, j]
            shr_i[mt, j] = out_inds[row, j]
        end
    end
    sync_threads()

    # the block's first warp merges, per row: PS-way max-extraction over the
    # segment heads against the preloaded running list (verbatim)
    if valid && tid <= Int32(32)
        for s in 1:PS
            @inbounds shp[s, mt] = Int16(1)
        end
        pa = 1
        for j in 1:K
            va = pa <= K ? @inbounds(shr_v[mt, pa]) : typemin(Float32)
            bv = typemin(Float32); bs = 0; bli = Int32(0)
            for s in 1:PS
                h = @inbounds shp[s, mt]
                if h <= K
                    v = @inbounds shv[s, mt, h]
                    if bv < v                     # ties keep the lowest segment
                        bv = v; bs = s; bli = @inbounds shi[s, mt, h]
                    end
                end
            end
            if va >= bv                           # running wins (or both done)
                @inbounds out_vals[row, j] = va
                @inbounds out_inds[row, j] = pa <= K ? shr_i[mt, pa] : Int32(0)
                pa += 1
            else
                @inbounds out_vals[row, j] = bv
                @inbounds out_inds[row, j] = bli == Int32(0) ? Int32(0) :
                                             offset + Int32((bs - 1) * segw) + bli
                @inbounds shp[bs, mt] += Int16(1)
            end
        end
    end
    return nothing
end

function launch_seg_rowtopk_cmerge!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                                    Cr, Ci, offset::Integer, k::Integer;
                                    segs::Integer = 8)
    M = size(Cr, 1)
    @assert size(Ci) == size(Cr) "Re/Im chunks must have the same shape"
    @assert size(Cr, 2) >= 1 "chunk must be non-empty"
    PS = Int(segs)
    K = Int(k)
    @assert 1 <= PS && 32 * PS <= 1024 "segs must be in 1:32"
    shmem = PS * 32 * K * (sizeof(Float32) + sizeof(Int32)) + PS * 32 * sizeof(Int16) +
            32 * K * (sizeof(Float32) + sizeof(Int32))
    @assert shmem <= 48 * 1024 "seg_rowtopk_cmerge_kernel! needs $(shmem) B of shared memory " *
                               "(> 48 KiB); lower `segs` (K = 20, Float32 reservoir: segs <= 8 fits)"
    @cuda threads = (32 * PS) blocks = cld(M, 32) shmem = shmem seg_rowtopk_cmerge_kernel!(
        D_val, D_loc, Cr, Ci, Int32(offset), Val(K), Val(PS))
    return nothing
end

# ==============================================================================
# The streamed complex flow
# ==============================================================================

# Task-level error router (the real flow's contract, own log prefix)
function _ctopk_flow_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "e2ecomplex: $(who) failed: ",
            sprint(showerror, err, catch_backtrace()))
    flush(stderr)
    err_out[] === nothing && (err_out[] = err)
    for ch in chs
        try
            close(ch)
        catch
        end
    end
    return false
end

# Contiguous D2H of a (possibly strided) device matrix: a GPU-side gather into
# a dense temp, then one contiguous copyto!.  Fresh host array.
function _gather_rows(v)
    h = Matrix{eltype(v)}(undef, size(v))
    t = CuMatrix{eltype(v)}(undef, size(v))
    t .= v
    copyto!(h, t)
    return h
end

"""
    complex_topk_flow(source, eng::ComplexTopKEngine; out_cap = 2, err_out,
                      tasks_out) -> Channel{ComplexTopKBatch}

Consume `source` -- any iterable of fp16 rope batches (anything with
`.embeds/.norms/.heads/.first`, e.g. a `Channel{RopeRealBatch{Float16}}` or a
pre-materialized `Vector{RopeRealBatch}`) -- and, for every batch, quantize
the embeddings to e4m3 on the GPU and run `complex_gemm_topk!` against the
engine's two resident fp8 databases, emitting `ComplexTopKBatch(vals, locs,
norms, heads, first)` with fresh HOST matrices (rows 1:nb -- the padded
rows_cap height is sliced off here).  Batch eltype must be Float16; batch
height must be <= the engine's rows_cap.  Consume to the end (the channel
closes itself) or `close(ch)` early: the stage closes `source` when it is a
channel, which tears the rope pipeline down quietly.  Real failures land in
`err_out[]` after the stream ends short; `tasks_out` receives this stage's
task.
"""
function complex_topk_flow(source, eng::ComplexTopKEngine;
                           out_cap::Int = 2,
                           err_out::Ref{Any} = Ref{Any}(nothing),
                           tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k = eng.k
    cap = eng.rows_cap
    out = Channel{ComplexTopKBatch}(out_cap)
    t_topk = Threads.@spawn begin
        try
            for bat in source
                emb = bat.embeds
                eltype(emb) == Float16 ||
                    throw(ArgumentError("batch eltype $(eltype(emb)) != Float16 (build the " *
                                        "rope stream with fp16 = true; this stage quantizes " *
                                        "the embeddings to e4m3 on the GPU)"))
                nb = size(emb, 1)
                nb == 0 && continue
                nb <= cap ||
                    throw(ArgumentError("batch height $nb > engine rows_cap $cap"))
                # upload/quantize on the GEMM stream (the real flow's pattern:
                # keep multi-ms kernel queues off the default stream)
                CUDA.stream!(eng.sg) do
                    upload_fp16_as_f8c!(eng, emb)
                end
                D_val = CuMatrix{Float32}(undef, cap, k)
                D_loc = CuMatrix{Int32}(undef, cap, k)
                complex_gemm_topk!(D_val, D_loc, eng)
                vals = _gather_rows(@view(D_val[1:nb, :])) # fresh host matrices,
                locs = _gather_rows(@view(D_loc[1:nb, :])) # pad rows sliced off
                D_val = D_loc = nothing # device pool reuses them next batch
                put!(out, ComplexTopKBatch(vals, locs, bat.norms, bat.heads, bat.first))
            end
        catch err
            if _ctopk_flow_error(err, err_out, out; who = "ctopk")
                source isa AbstractChannel && close(source)
            end
        finally
            close(out)
        end
    end
    append!(tasks_out[], [t_topk])
    return out
end

"""
    complex_rope_topk_stream(re::RopeEncoder, file::String, Bre::CuMatrix{F8},
                             Bim::CuMatrix{F8}; k = 20, w = 2^16,
                             batch_size = 2^13, rows_cap = batch_size,
                             normalize = 0, parts = nthreads(), in_cap = 2,
                             out_cap = 2, topk_out_cap = 2, segs = 8,
                             engine = :lt, cout = :f16, progress = false,
                             err_out, tasks_out) -> Channel{ComplexTopKBatch}

The full COMPLEX fp8 flow on a fasta `file`: stream-rope-encode the fragments
(`rope_encode_real_stream` with `fp16 = true` -- the split [Re; Im] rows ARE
the complex vectors) and process every batch with the engine's double-
buffered fp8 GEMM pair + fused per-row abs2 top-k against the RESIDENT
databases `Bre, Bim ∈ e4m3^(2*m*4^c x N)` (their columns are the reference
windows' [Re; Im] stacks and -90deg rotations, assumed normalized --
normalize = 0).  Emits `ComplexTopKBatch(vals (nb,k) Float32 abs2 scores,
locs (nb,k) Int32 global column ids, norms, heads, first)` with fresh host
matrices.  Consume to the end or `close(ch)` early (quiet teardown of the
whole pipeline); `err_out`/`tasks_out` wire both stages together (three
tasks).
"""
function complex_rope_topk_stream(re::RopeEncoder, file::String, Bre::CuMatrix{F8},
                                  Bim::CuMatrix{F8};
                                  k::Int = 20, w::Int = 2^16,
                                  batch_size::Int = 2^13, rows_cap::Int = batch_size,
                                  normalize::Int = 0, parts::Int = Threads.nthreads(),
                                  in_cap::Int = 2, out_cap::Int = 2, topk_out_cap::Int = 2,
                                  segs::Int = 8, engine::Symbol = :lt, cout::Symbol = :f16,
                                  progress::Bool = false,
                                  err_out::Ref{Any} = Ref{Any}(nothing),
                                  tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(Bre, 1) == rdim && size(Bim, 1) == rdim ||
        throw(ArgumentError("Bre is $(size(Bre, 1)) x $(size(Bre, 2)), Bim is " *
                            "$(size(Bim, 1)) x $(size(Bim, 2)); the encoder's complex " *
                            "embedding dim is 2*m*4^c = $rdim"))
    size(Bim, 2) == size(Bre, 2) ||
        throw(ArgumentError("Bre/Bim must have the same column count"))
    eng = ComplexTopKEngine(Bre, Bim; k, w, rows_cap, segs, engine, cout)
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return complex_topk_flow(rope_ch, eng; out_cap = topk_out_cap, err_out, tasks_out)
end

# ------------------------------------------------------------------------------
# The complex index .bin -> resident fp8 database pair
# ------------------------------------------------------------------------------

"""
    load_index(path = E2C_INDEX_BIN)

Load a complex-flow index .bin (Julia-serialized NamedTuple): EITHER the
split-form complex save (`format = "indexflowcomplex.fp8.v1"`: embeds8_re /
embeds8_im (m*4^c, 2*n_frag) Float8_E4M3FN columns -- a matrix of real values
together with a matrix of imaginary values, quantized at save time) OR the
legacy real fp8 save (`"indexflowreal.fp8.v1"`: embeds8 (2*m*4^c, 2*n_frag),
the row-stacked [Re; Im] columns -- Bre is that matrix verbatim, Bim is
derived on the GPU at load).  This flow assumes (asserts) `normalize = 0`:
unit-MODULUS complex columns (|z|^2 = Re^2 + Im^2 = 1) are what makes both
databases' columns normalized.
"""
function load_index(path::String = E2C_INDEX_BIN)
    isfile(path) || error("complex index .bin not found: $path (generate with " *
                          "`e2ecomplex.jl save`, or point E2COMPLEX_INDEX at an " *
                          "indexflowreal.fp8.v1 .bin for the GPU-derived legacy path)")
    d = Utils.load(path)
    fmt = get(d, :format, nothing)
    fmt in (E2C_FORMAT, E2C_LEGACY_FP8_FORMAT) ||
        error("not an $E2C_FORMAT/$E2C_LEGACY_FP8_FORMAT index: $path")
    d.normalize == 0 ||
        error("e2ecomplex assumes a normalize = 0 index (got $(d.normalize)): " *
              "unit-MODULUS complex columns are what makes Bre/Bim normalized")
    rdim = 2 * d.m * 4^d.c
    D = rdim ÷ 2
    N = 2 * d.n_frag
    if fmt == E2C_FORMAT
        @assert get(d, :complex, false) "complex flag mismatch"
        @assert eltype(d.embeds8_re) == F8 && size(d.embeds8_re) == (D, N) "embeds8_re shape/eltype mismatch"
        @assert eltype(d.embeds8_im) == F8 && size(d.embeds8_im) == (D, N) "embeds8_im shape/eltype mismatch"
    else
        @assert eltype(d.embeds8) == F8 && size(d.embeds8) == (rdim, N) "embeds8 shape/eltype mismatch"
    end
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"
    @info "index loaded" file = basename(path) format = fmt k = d.k kstep = d.kstep n_frag =
        d.n_frag cols = N rdim = rdim encoder = (d.s, d.m, d.c) normalize = d.normalize
    return d
end

"""
    _complex_databases(re_h, im_h) -> (Bre, Bim)

Build both resident GEMM databases from the SPLIT complex columns (host e4m3
matrices (D, N)): Bre = [re; im] and Bim = [im; -re], both (2D, N).  Per
column block: one H2D per half, then the four quarter-writes (the negate is
a fused GPU broadcast; fp8 negation is exact).  Peak device memory: the two
databases + one block of each half.
"""
function _complex_databases(re_h::Matrix{F8}, im_h::Matrix{F8})
    D, N = size(re_h)
    size(im_h) == (D, N) ||
        throw(ArgumentError("split halves must match ($(size(re_h)) vs $(size(im_h)))"))
    Bre = CuMatrix{F8}(undef, 2 * D, N)
    Bim = CuMatrix{F8}(undef, 2 * D, N)
    blk = 2^21 # 2 GiB e4m3 per half per block
    for j0 in 1:blk:N
        j1 = min(j0 + blk - 1, N)
        dre8 = CuArray(@view re_h[:, j0:j1]) # H2D, contiguous column block
        dim8 = CuArray(@view im_h[:, j0:j1])
        @views Bre[1:D, j0:j1] .= dre8
        @views Bre[D+1:2D, j0:j1] .= dim8
        @views Bim[1:D, j0:j1] .= dim8
        @views Bim[D+1:2D, j0:j1] .= .- dre8
    end
    CUDA.synchronize()
    return (Bre, Bim)
end

"""
    index_database(d) -> (Bre, Bim)::Tuple{CuMatrix{F8},CuMatrix{F8}}

The resident fp8 database pair for a loaded index.  For the split-form
complex save this is `_complex_databases` on the stored halves.  For the
legacy real fp8 save ("indexflowreal.fp8.v1") the stored matrix IS Bre --
the real flow's B verbatim (pure H2D) -- and Bim = [Im; -Re] is derived ON
THE GPU per column block (row-half swap + fused negate; bitwise exact).
"""
function index_database(d)
    if get(d, :format, nothing) == E2C_FORMAT
        return _complex_databases(d.embeds8_re, d.embeds8_im)
    end
    Bre = CuArray(d.embeds8) # pure H2D (the real flow's fast path verbatim)
    rdim, N = size(Bre)
    D = rdim ÷ 2
    Bim = CuMatrix{F8}(undef, rdim, N)
    blk = 2^21
    for j0 in 1:blk:N
        j1 = min(j0 + blk - 1, N)
        @views Bim[1:D, j0:j1] .= Bre[D+1:2D, j0:j1]
        @views Bim[D+1:2D, j0:j1] .= .- Bre[1:D, j0:j1]
    end
    CUDA.synchronize()
    return (Bre, Bim)
end

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, score the top-k against the provenance
# (the locs-based intersection test is verbatim e2ehuman.jl -- the complex
# flow changes the RANKING statistic, not the ground truth)
# ------------------------------------------------------------------------------

"""
    map_and_count(re, reads_file, Bre, Bim, db_heads, db_starts, dbk; ktop, w,
                  batch_size, segs, engine, normalize, progress)
        -> (total, correct, rank_hist, seconds)

Run the full COMPLEX fp8 flow (`complex_rope_topk_stream`: fasta -> rope fp16
-> 2x e4m3 GEMM + per-row abs2 top-k) over `reads_file` and score every row
against its provenance header: CORRECTLY MAPPED iff the true window
[start, start+len-1] in record src intersects at least one of the ktop
returned database windows (db_heads[c], db_starts[c], length dbk).  Returns
the processed-read count, the number of correctly mapped records and the
first-hit rank histogram (rank_hist[j] = reads whose first intersecting hit
was rank j).
"""
function map_and_count(re::RopeEncoder, reads_file::String, Bre::CuMatrix{F8},
                       Bim::CuMatrix{F8}, db_heads::Vector{String},
                       db_starts::Vector{Int}, dbk::Int;
                       ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, normalize::Int = 0,
                       progress::Bool = false)
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(Bre, 2)
    size(Bim) == size(Bre) ||
        throw(ArgumentError("Bre/Bim shape mismatch"))
    # per-column record id: the .bin's heads repeat a handful of unique
    # record headers over millions of columns -- intern them once
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
    t = @elapsed for bat in complex_rope_topk_stream(re, reads_file, Bre, Bim;
                                                     k = ktop, w,
                                                     batch_size, rows_cap = batch_size,
                                                     normalize, segs, engine, progress,
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
    Utils.save(nt, binfile)
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
    test_fp8_gemm(engines = engines) # gemmtopkfp8's GEMM + chunked-ldb unit test
    _check_engine_cgemm_topk(k = k, engines = engines)

    # ==========================================================================
    # A2. THE LOADER: the split-form complex save reloads into the exact
    #     Bre/Bim pair (Bre = [re; im], Bim = [im; -re], bitwise).
    # ==========================================================================
    @info "A2. complex index loader / database derivation"
    tref = _ensure_e2h_test_ref()
    tbin = _e2h_build_index(tref)
    td = load_index(tbin)
    (tBre, tBim) = index_database(td)
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
    d = load_index(binfile)
    (Bre, Bim) = index_database(d)
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
The windows are indexflowreal's windows (same walk, same rc convention); the
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
    t = @elapsed Utils.save(nt, binfile)
    @info "saved $binfile ($(round(filesize(binfile) / 2^30; digits = 2)) GiB) in $(round(t; digits = 1)) s"
    load_index(binfile) # structural verify: reload + shape asserts
    return binfile
end

"""
    _build_complex_index_matrix(re, file; k, kstep, batch_size, normalize,
                                fp16, progress, max_records, err_out, tasks_out)

The production collection step of the COMPLEX save: a cheap first pass counts
the windows (so the matrices are allocated ONCE at their exact size), then
`ref_rope_real_stream` fills

    embeds8_re (m*4^c, 2*n_frag) Float8_E4M3FN   the Re columns (fwd 1:n, rc n+1:2n)
    embeds8_im (m*4^c, 2*n_frag) Float8_E4M3FN   the Im columns (same order)
    norms      (m, 2*n_frag) Float32             complex-modulus norms
    heads/starts/strand                            location meta (2*n_frag)

(= indexflowreal's fp8 build with the [Re; Im] row stack SPLIT into two
matrices -- the identical quantization F8.(Float32.(.)) applied per half).
"""
function _build_complex_index_matrix(re::RopeEncoder, file::String;
                                     k::Int = re.k,
                                     kstep::Int = max(k ÷ 16, 1),
                                     batch_size::Int = 2^13,
                                     normalize::Int = 0,
                                     fp16::Bool = true,
                                     progress::Bool = false,
                                     max_records::Int = typemax(Int),
                                     err_out::Ref{Any} = Ref{Any}(nothing),
                                     tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    t_count = @elapsed (nwin, nrec, nkept) = _count_ref_windows(file; k, kstep, max_records)
    @info "count pass: $nrec records, $nkept with >= k bases -> $nwin windows (x2 with revcomp) in $(round(t_count; digits = 2)) s"
    D = re.m * 4^re.c
    embeds8_re = Matrix{F8}(undef, D, 2 * nwin)
    embeds8_im = Matrix{F8}(undef, D, 2 * nwin)
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    heads = Vector{String}(undef, 2 * nwin)
    starts = Vector{Int}(undef, 2 * nwin)
    strand = fill(UInt8(0), 2 * nwin) # rc rows set below
    row = 0
    t_stream = @elapsed for nt in ref_rope_real_stream(re, file; k, kstep, batch_size,
                                                       normalize, fp16, progress,
                                                       max_records, err_out, tasks_out)
        nb = size(nt.embeds, 1) ÷ 2 # batch rows: 1:nb fwd, nb+1:2nb revcomp
        ef = @view nt.embeds[1:nb, :]
        er = @view nt.embeds[nb+1:2nb, :]
        # fused split + transpose + fp16 -> fp32 -> e4m3 quantize into the
        # column blocks: the exact conversion of indexflowreal's fp8 build,
        # applied to each half separately
        @views embeds8_re[:, row+1:row+nb] .= F8.(Float32.(permutedims(ef[:, 1:D])))
        @views embeds8_im[:, row+1:row+nb] .= F8.(Float32.(permutedims(ef[:, D+1:2D])))
        @views embeds8_re[:, row+nb+1:row+2*nb] .= F8.(Float32.(permutedims(er[:, 1:D])))
        @views embeds8_im[:, row+nb+1:row+2*nb] .= F8.(Float32.(permutedims(er[:, D+1:2D])))
        copyto!(view(norms, :, row+1:row+2*nb), nt.norms)
        copyto!(heads, row + 1, nt.heads, 1, 2 * nb)
        copyto!(starts, row + 1, nt.starts, 1, 2 * nb)
        copyto!(strand, row + 1, nt.strand, 1, 2 * nb)
        row += 2 * nb
    end
    err_out[] === nothing || error("stream failed: $(err_out[])")
    row == 2 * nwin || error("stream produced $row rows, expected $(2 * nwin)")
    @info "stream pass: $row rows in $(round(t_stream; digits = 2)) s"
    return (embeds8_re = embeds8_re, embeds8_im = embeds8_im, norms = norms,
            heads = heads, starts = starts, strand = strand, n_frag = nwin)
end

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
        error("reads fasta not found: $E2C_READS (generate with indexsample.jl, err = 0.15)")
    t_load = @elapsed d = load_index(E2C_INDEX_BIN)
    t_B = @elapsed (Bre, Bim) = index_database(d)
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
form), build the (Bre, Bim) database pair on the GPU, stream the indexsample
err = 0.15 reads through the complex flow (2x fp8 GEMM + abs2 top-k,
kstep = k/16 = 1250) and report THE number of correctly mapped records (plus
rate and first-hit rank histogram).
"""
function run_e2e_human(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2C_READS) ||
        error("reads fasta not found: $E2C_READS (generate with indexsample.jl, err = 0.15)")
    @info "loading the human complex index" E2C_INDEX_BIN
    t_load = @elapsed d = load_index(E2C_INDEX_BIN)
    @info "building the complex fp8 databases on GPU (Bre = [Re; Im], Bim = [Im; -Re])"
    t_B = @elapsed (Bre, Bim) = index_database(d)
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

# host-timed, device-synchronized measurement of a (stream-offloaded) GPU
# loop (CUDA.@elapsed cannot time other streams -- flowtopkfp8's finding)
function _timed_gpu(f, iters::Integer)
    CUDA.device_synchronize()
    t0 = time_ns()
    for _ in 1:iters
        f()
    end
    CUDA.device_synchronize()
    return (time_ns() - t0) / 1e9 / iters
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
