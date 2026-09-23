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
# spec).  It includes indexreal.jl (=> ropeflowreal/fastareads_v3/
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
#   "indexreal.fp8.v1" save (the real flow's row-stacked [Re; Im]):
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
# sample n=131,072 k=20,000 err=0.15 s=42 file; k(top) = 20, w = 2^16,
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
#          indexreal.fp8.v1 layout loads via the GPU-derived path)
#          E2COMPLEX_READS (reads fasta; default
#          <fasta>.sample_n131072_k20000_e0.15_s42.fasta)
#          E2COMPLEX_K / E2COMPLEX_KSTEP (save/run fragment length and step)
#
# USAGE
#   julia --project=. -t 16 test/e2ecomplex.jl save    # once (index build)
#   julia --project=. -t 16 test/e2ecomplex.jl check
#   julia --project=. -t 16 test/e2ecomplex.jl test
#   julia --project=. -t 16 test/e2ecomplex.jl run --engine=lt
#   julia --project=. -t 16 test/e2ecomplex.jl         # test + bench + run
# ==============================================================================
# ------------------------------------------------------------------------------
# NEW TREE NOTES (reorganization): this file is now a LIBRARY layer
# (search/engine_complex.jl).  The legacy includes (indexflowreal.jl +
# gemmtopkfp8.jl) are replaced by the entry script's canonical include order;
# requires:
#   common/dna.jl, common/util.jl (_timed_gpu), common/testref.jl,
#   fasta/*, encode/*, gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl
#   (F8, fp8_gemm_mma!, fp8_gemm_lt!, seg_rowtopk_cmerge_kernel!,
#   launch_seg_rowtopk_cmerge!), encode/stream.jl (rope_encode_real_stream),
#   index/{build,load,complex}.jl, reads/provenance.jl (parse_read_head).
# The experiment ENV consts (E2C_FASTA/E2C_K/...), the local harness variants,
# the self-check suite and the run_* entry points moved to
# experiments/e2e_complex.jl; the index loaders moved to index/{load,complex}.jl
# (load_index / load_index_complex / index_database / index_database_complex).
# This engine is NOT co-includable with engine_fp8.jl / engine_fp16.jl
# (name clashes: _gather_rows, map_and_count, ...).
# ------------------------------------------------------------------------------

using Random
using CUDA
using Printf
using Mmap
using LinearAlgebra # norm (came transitively from gemmtopkfp8's usings in the legacy chain)
using Base.Threads

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
    load_index_complex(path = E2C_INDEX_BIN)

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
    # record headers over millions of columns -- intern them once.  The colon
    # eval provenance names records by ACCESSION (first whitespace token of
    # the '>' header, sans '>'), so the same ids are also keyed by accession
    # (unified provenance handling, verbatim engine_fp8.jl's map_and_count).
    rec_id = Dict{String,Int32}()
    rec_id_acc = Dict{String,Int32}()
    rec_of_col = Vector{Int32}(undef, N)
    for j in 1:N
        h = db_heads[j]
        id = get(rec_id, h, Int32(0))
        if id == 0
            id = Int32(length(rec_id) + 1)
            rec_id[h] = id
            rec_id_acc[String(split(h)[1][2:end])] = id # accession sans '>'
        end
        rec_of_col[j] = id
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
            # sample headers carry len == dbk; colon headers carry no
            # length -- the reader trimmed every kept record to dbk bases,
            # so the scored fragment is the dbk-long window at h.start
            h.len === nothing || h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rid = get(rec_id_acc, h.src, Int32(0)) # both formats name by accession
            hit = 0
            @inbounds for j in 1:ktop
                c = Int(bat.locs[r, j])
                # read window [h.start, h.start+dbk-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid &&
                   h.start <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= h.start + dbk - 1
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

