# ==============================================================================
# flowtopkfp8.jl -- flowtopk with the final GEMM+top-k stage in fp8: the
# ropeflowreal streamed REAL rope encodings, quantized to e4m3 at the stage
# boundary, fed into gemmtopkfp8's batched fp8 GEMM + per-row top-k against a
# large fp8 database matrix resident in VRAM.
#
# PROBLEM
#   Same flow as flowtopk.jl (see its header; prompts/flowtopk.md), but the
#   last stage -- the batched GEMM + per-row top-k -- runs on fp8 reals via
#   gemmtopkfp8.jl's machinery:
#     fasta -fasta_reads-> gather -encode-> Channel{RopeRealBatch{Float16}}
#                                               |  out_cap batches in flight
#                 topk stage (this file):  H2D(fp16) + e4m3-quantize on sg,
#                 then per B-chunk [fp8 GEMM on sg || seg-rowtopk on st]
#                 x cld(N, w) chunks, then D2H of the merged (nb x k) rows
#                                               v
#                                    Channel{TopKBatch}
#   A database matrix B ∈ e4m3^(2^11 x 2^21) (4 GiB; 2^21 reference vectors,
#   column-normalized) is GIVEN on the GPU; every streamed batch of fragment
#   embeddings embeds_bat ∈ fp16^(2^13 x 2^11) is quantized to e4m3 and
#   matched against all of B: C_bat = Q(embeds_bat) * B would be
#   (2^13 x 2^21) ~ 32 GiB in fp32 -- as unmaterializable as in flowtopk, so
#   each batch is one gemmtopk(fp8) problem of height 2^13: column chunks of
#   B (w = 2^15), double-buffered C chunks on two non-blocking streams, the
#   fused segmented rowtopk merge into the batch's running top-k.  `locs` are
#   GLOBAL 1-based column ids into B (valid across the whole stream).
#
# DESIGN
#   1. Quantization boundary.  The rope stream stays fp16 (ropeflowreal's
#     fp16 = true; there is no fp8 rope output -- and the fp8 GEMM is where
#     the speedup is: the embedding D2H is unchanged).  The topk stage
#     quantizes each uploaded batch to e4m3 (satfinite) ON THE GPU: one
#     contiguous fp16 H2D, one fused fp16->fp32 broadcast and gemmtopkfp8's
#     f32_to_f8! converter kernel (~180 MiB of device traffic at the default
#     config, i.e. microseconds against ~10 ms of per-batch GEMM).  The fp16
#     -> e4m3 path is exact-modelled by the CPU reference (F8 after fp16).
#   2. GEMM engines.  gemmtopkfp8's two engines, unchanged: the nvcc-compiled
#     mma.sync tensor-core kernel (64x64 tiles, m16n8k32 e4m3, fp32
#     accumulators, PTX .version 9.0 stamping, CuModule + cudacall) and the
#     raw-ccall cuBLASLt fp8 path (needs the 595 driver userland; see
#     gemmfp8cpp.jl's header).  gemmtopkfp8 measured :lt winning at ITS
#     chunk shape (M = 2^17: the mma kernel re-reads its A tile per column
#     block and a 2^17 x 2^11 A no longer fits L2) -- that argument does not
#     apply at the flow's M = 2^13 (A is 16 MiB), so both engines are
#     benchmarked head-to-head AT THE FLOW GEOMETRY (in-process, same
#     quantized batch) and the end-to-end runs report both; the engine is a
#     TopKEngine constructor choice (default :lt, gemmtopkfp8's fast path).
#     C chunks are stored fp16 (cout = :f16, gemmtopkfp8's measured sweet
#     spot: the top-k scan is element-issue-bound, so halving C's bytes is
#     pure win and fp16 storage of fp32-accumulated values is far more
#     accurate than flowtopk's fp16-accumulated GEMM); cout = :f32 is
#     selectable.  cout = :f8 is NOT carried over -- gemmtopkfp8 measured it
#     slower than :f16 (the e4m3 decode costs more than the saved bytes) and
#     location-churning.  mma restrictions (M/w multiples of 64, K multiple
#     of 32, w | N) and the lt restriction (M/K multiples of 16) are
#     constructor asserts.
#   3. Fixed-height batch padding.  The flow contract keeps flowtopk's
#     (arbitrary nb <= batch_size, ragged tail batches), but both fp8 GEMM
#     engines need a fixed, aligned matrix height (M % 64 / M % 16) AND a
#     leading dimension common to A and C.  So the engine allocates its
#     quantized-A staging (rows_cap x rdim e4m3, plus a matching zeroed fp32
#     scratch -- the converter is 1-D over the WHOLE staging, see
#     upload_fp16_as_f8!'s linear-layout contract) and its two C buffers
#     (rows_cap x w) at rows_cap ROUNDED UP to the engine's multiple, and
#     every batch -- including tail batches -- is computed at the FULL
#     rows_cap height: the engine's fp32 scratch keeps rows nb+1:rows_cap
#     zero (only rows 1:nb are overwritten per batch), so the pad rows'
#     scores are exact zeros and are SLICED OFF after the D2H
#     (TopKBatch carries only rows 1:nb).  Steady-state waste is
#     <= (padmul-1)/batch_size < 1% of the GEMM; a ragged tail batch wastes
#     more (once per file); the buffers are never reallocated across the
#     stream and the cuBLASLt layout/algo cache sees ONE key per chunk width.
#   4. Top-k kernel.  gemmtopk's seg_rowtopk_merge_kernel! (the PS-threads-
#     per-row variant; at M = 2^13 the 1-thread kernel launches 64 blocks
#     and reads C at ~160 GiB/s vs 641 for segs = 8 -- see
#     prompts/gemmtopk.md), ported from single-type T to the gemmtopkfp8
#     split: C chunks are read as Tc ∈ {Float16, Float32, UInt8 = raw e4m3
#     bytes} via _scanval, the register reservoir / shared-memory staging /
#     D_val are Tv = Float32.  All sentinels (out-of-range loads, reservoir
#     init, thresholds) are typemin(Tv) DIRECTLY -- converting typemin(Tc)
#     instead would let e.g. Float16's -65504 into a Float32 reservoir whose
#     threshold is -3.4e38.  Shared memory at K = 20: PS*32*K*(4+4) +
#     PS*32*2 + 32*K*(4+4) = 46.6 KiB at segs = 8 < the 48 KiB cap (segs > 8
#     needs an opt-in; the Float32 reservoir costs 2 KiB/segment-row more
#     than flowtopk's fp16 one, dropping the cap from segs <= 10 to <= 8).
#     The running top-k is reset per batch with fill!(typemin(Float32)) --
#     batches are independent query sets.  overlap = false runs the same
#     chunk loop sequentially on the default stream (bitwise identical).
#   5. Results contract.  TopKBatch(vals (nb,k) Float32 descending, locs
#     (nb,k) Int32 global column ids into B, norms (m,nb) Float32, heads,
#     first) -- fresh HOST matrices, retain freely.  D_val is Float32 (not
#     flowtopk's fp16): gemmtopkfp8's rule -- an e4m3 product can reach
#     448^2 ~ 2*10^5 > fp16 max, and the fp32 reservoir is already the
#     kernel's currency.  Numerics: fp8 GEMM accumulates in fp32 and C is
#     stored fp16, so a pipeline score deviates from the exactly-modelled
#     reference (dequantized e4m3 operands, fp64 dot) by ~1 fp16 storage ulp
#     of |C| <= ~1 -- about 1e-3, tighter than flowtopk's 3e-3 (its cuBLAS
#     path accumulated in fp16).  Locations are robust the same way.
#   6. Naming.  The stage keeps flowtopk.jl's names (TopKEngine, TopKBatch,
#     topk_flow, batch_gemm_topk!, rope_topk_stream) -- include ONE of the
#     two scripts per session, never both (struct redefinition), exactly
#     like gemmtopk.jl / gemmtopkfp8.jl share names.
#
#   File map: TopKEngine / padmul                 the reusable fp8 engine
#             upload_fp16_as_f8!                  H2D + e4m3 quantize
#             _gemm_chunk!                        engine dispatch (mma | lt)
#             seg_rowtopk_merge_kernel! /         the (Tc, Tv) segmented
#               launch_seg_rowtopk_merge!           top-k kernel (+ launch)
#             batch_gemm_topk!                    the double-buffered loop
#             topk_flow / rope_topk_stream        the streamed flow
#             _check_quantize_path /              correctness (see below)
#               test_seg_rowtopk_merge_kernel /
#               _ref_topk_chunked / _assert_topk_rows /
#               _check_engine_gemm_topk / _check_topk_stream
#             bench_flow_topk                     micro-bench + driver/sweeps
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16, min of 3, big file =
# 2.6 GB / 2^17 fragments, B = 2^11 x 2^21 e4m3 column-normalized N(0,1),
# k = 20, w = 2^16, batch = 2^13 x 2^11 -> 16 batches of 32 chunks):
#
#   rope stream only, no-op consumer fp16 (H2D+kernel+D2H)   0.550 s    4.77 GB/s
#   flow END-TO-END: fasta -> rope fp16 -> fp8 gemm+topk     3.022 s  372.6 TFLOPS*
#   flow COMPUTE-ONLY (rope excluded, replayed)              2.772 s  406.2 TFLOPS*
#   compute-only, sequential (overlap=false)                 2.77 s    1.00x
#   quantize+upload (H2D + fp16->fp32 + e4m3 convert)        1.52 ms/batch  (~0.9%)
#   per-chunk gemm, engine x cout (w=2^16, M=2^13):
#     lt  :f16   3.148 ms  (698.5 TFLOPS)   lt  :f32   3.292 ms  (667.9)
#     mma :f16  29.943 ms  ( 73.4 TFLOPS)   mma :f32  31.258 ms  ( 70.4)
#   fp16 cuBLAS mul! (same shape, same process)              6.600 ms  (333.2)
#     -> same-process fp8/fp16 GEMM speedup 2.10x (lt vs cuBLAS)
#   per-chunk seg-rowtopk (w=2^16 fp16 chunks, Float32 reservoir):
#     segs 2/4/8 = 3.845/2.360/1.900 ms (526 GiB/s at segs=8); on :f32 chunks
#     2.003 ms (998 GiB/s) -- SAME ELEMENT rate as :f16, i.e. the scan is
#     element-issue-bound (gemmtopkfp8's finding carries over)
#   batch sweep (w=2^15): 2^11: 5.07 s (222.1) · 2^12: 3.67 (306.6) ·
#     2^13: 3.11 (361.9) · 2^14: 3.07 (366.9) -- 2^13 is the knee, 2^14 flat
#   w sweep: 2^13: 5.46 s (206.3) · 2^14: 3.92 (287.2) · 2^15: 3.11 (361.9) ·
#     2^16 (default): 2.77 (406.2) -- faster GEMM => fatter chunks win until
#     the C buffers hit the memory cap (2 x 8 GiB at 2^16; 2^17 = 2 x 16 OOMs)
#
#   * TFLOPS = 2^50 FLOP / wall time over the whole big file.
#   Doubled database (--Nbits=22, B = 2^11 x 2^22 e4m3 = 8 GiB, 2^51 FLOP):
#   e2e 5.68 s (396.2 TFLOPS), compute-only 5.47 s (411.7 TFLOPS) -- exactly
#   2x the 2^21 wall time with slightly BETTER sustained TFLOPS (the 1.5 ms
#   per-batch quantize and the batch boundaries amortize over 64 chunks);
#   per-chunk rates unchanged (lt :f16 3.120 ms / 704.8 TFLOPS, fp16 cuBLAS
#   6.532 / 336.7 -> in-process speedup 2.09x); pool peaked ~14.7 GiB.
#   Engine verdict: lt everywhere at the flow's geometry -- gemmtopkfp8's
#   L2-vs-A-size explanation for mma's losses at M = 2^17 does not apply here
#   (A is 16 MiB), yet mma still loses 4.5x and degrades superlinearly with w
#   (219 TFLOPS at w = 2^15 -> 73 at 2^16); lt it is.
#   Budget at the default config: GEMM floor 1.61 s (32 x 16 x 3.148 ms),
#   exposed seg top-k 0.97 s, residual 0.19 s.  The rope stage (0.55 s) hides
#   completely (e2e 3.02 vs compute-only 2.77).  vs flowtopk's fp16 flow:
#   1.75x end-to-end (5.29 -> 3.02 s), 1.84x compute-only (5.10 -> 2.77 s;
#   cross-process caveat), vs gemmtopkfp8's monolithic M = 2^17 run ~1.1x
#   more time for the same 2^50 FLOP (thinner GEMM batches: 698 vs 657
#   TFLOPS chunk rate, plus a per-batch quantize and the exposed top-k).
#
# CORRECTNESS (all on kau, julia -t 16): e4m3 quantize/upload path == host
# F8.() conversion bitwise with zeroed pad rows; seg_rowtopk_merge_kernel!
# exact CPU-multiset unit test over Tc = Float16/Float32/e4m3-bytes x ragged /
# strided-parent / w == k / empty-segment / real-geometry shapes; engine vs an
# exact same-engine chunk-wise reference (top-k values AND locations bitwise)
# over ragged (w ∤ N), w == N, single-row (M = 1 -> padded), divisible and
# real-geometry (2^13 x 2^15) shapes for both engines and both couts, tail
# batch on a reused engine, pipelined == sequential, dequantized-fp32 gemv
# spot check (gemmtopkfp8's spot_check); end-to-end flow on the 164 MB file
# vs a float64 CPU reference that models the fp16->e4m3 quantization exactly
# (independent rope encoder + fp64 GEMM over the dequantized database):
# batch layouts (tail / exact fit / single), normalize 0/1, a second encoder
# config, the mma engine at real geometry (N = 2^15, w = 2^13), values +
# locations + norms passthrough, heads/first bookkeeping, early-close
# teardown (all 3 tasks joined, err_out clean, pipeline reusable), empty file
# -> 0 batches.  ALL PASS (re-validated after every change below).
#
# KNOWN RISKS / FOLLOW-UPS
#   * CUDACore (6.3.x) fatal: synchronize() of a BUSY default/legacy stream
#     takes the nonblocking slow path, whose worker thread calls
#     context!(stream.ctx) on the legacy stream's ctx = nothing -> MethodError
#     with no handler on a foreign thread (process dies or hangs).  Everything
#     in this script therefore keeps multi-ms kernel queues on NAMED
#     non-blocking streams (eng.sg / sg16) and uses blocking copies on the
#     default stream; batch_gemm_topk!'s default-stream fill!s are covered by
#     the following device_synchronize (context sync takes a real context).
#   * both GEMM engines compute the padded rows_cap height for every batch
#     (see DESIGN 3) -- the pad waste is bounded and sliced off, but a much
#     smaller final batch on a large rows_cap pays for it; a per-height
#     buffer realloc (CUDA pool) would remove it if it ever matters.
#   * the mma kernel's w | N restriction is a constructor assert; ragged
#     final chunks run on :lt only (production N = 2^21 with w = 2^15 is
#     exact for both).
#   * segs > 8 (Float32 reservoir, K = 20) needs a > 48 KiB shared-memory
#     opt-in (gemmtopk measured diminishing returns for segs 4 -> 8).
#   * the fp8 quantization of the embeddings costs ~3% relative rounding on
#     A on top of B's own e4m3 quantization; score-level it is modelled
#     exactly by the reference, but LOCATION agreement with an ideal fp16
#     pipeline degrades to gemmtopkfp8's near-tie statistics (~25% of
#     within-noise swaps) -- inherent to fp8 reals, not to this flow.
#   * in-process fp16 GEMM baseline only (cuBLAS mul! on the pre-quantization
#     fp16 operands, same seg top-k kernel): flowtopk.jl cannot be included
#     in the same session (name clash, DESIGN 6), so end-to-end fp16 numbers
#     are quoted from its header (cross-process caveat applies).
#
# Run modes (first non-flag ARGV[1]):
#   gen   generate the test fasta files if missing (fastareads_v3)
#   test  correctness: quantize path, segmented top-k kernel unit tests
#         (Float16/Float32/e4m3 chunks), engine checks vs same-engine chunked
#         references + dequantized-fp32 spot checks (both engines x couts),
#         end-to-end flow vs the float64 CPU reference (batch layouts,
#         normalize 0/1, second encoder config, mma at real geometry),
#         early-close teardown, empty file
#   bench  per-chunk GEMM engine x cout head-to-head + quantize cost + seg
#         top-k sweep + in-process fp16 GEMM line, end-to-end and
#         compute-only flow per engine, sequential baseline, w and batch
#         sweeps, memory accounting
#   all   everything above (default)
#   flags: --engine=mma|lt|both (bench/test engines, default both),
#          --Nbits=21 (B = 2^11 x 2^Nbits e4m3; 21 = 4 GiB production shape),
#          --quick (reps = 1, no sweeps), --no-sweep
#
# USAGE
#   julia --project=. test/flowtopkfp8.jl            # gen + test + bench
# ------------------------------------------------------------------------------
# NEW TREE NOTES (reorganization): this file is now a LIBRARY layer
# (search/engine_fp8.jl).  The legacy include block is replaced by the entry
# script's canonical include order; requires:
#   common/dna.jl, common/util.jl (_timed_gpu, _reps), common/testref.jl
#   (ensure_data3), fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl (F8, fp8_gemm_mma!,
#   fp8_gemm_lt!, seg_rowtopk_merge_kernel!, launch_seg_rowtopk_merge!,
#   test_seg_rowtopk_merge_kernel, TOPK_UNROLL, _scanval, byteptr),
#   reads/provenance.jl (parse_read_head), index/load.jl (E2H_FP8_FORMAT).
# The kernel copies and _timed_gpu that used to live in this file moved to
# gemm/topk_kernels.jl and common/util.jl (bodies unchanged).  The provenance
# scorer `map_and_count` (legacy e2ehuman.jl) now lives at the bottom of this
# file.  NOT co-includable with engine_fp16.jl / engine_complex.jl (same
# names: TopKEngine, topk_flow, rope_topk_stream, map_and_count, ...).
# The PROGRAM_FILE dispatch below is kept: running this file directly still
# offers gen|test|bench|all.
# ------------------------------------------------------------------------------

using CUDA
using LinearAlgebra
using Random
using Printf
using Base.Threads

# ==============================================================================
# The batched fp8 GEMM + top-k engine (gemmtopk(fp8)'s per-batch loop with
# the flow's fixed-height padding and a hoisted per-batch state)
# ==============================================================================

"""
One streamed batch of fp8 top-k results.  `vals[r, :]` are fragment r's k
best scores against the database B, DESCENDING; `locs[r, :]` the matching
GLOBAL 1-based column ids into B (scores[r, j] = dot(e4m3(embeds_r),
B[:, locs[r, j]])).  `vals` are Float32 (the kernel's reservoir currency; an
e4m3 product can exceed fp16 max -- gemmtopkfp8's rule).  `norms`/`heads`/
`first` are the rope batch's passthrough (per-copy norms (m, nb) Float32,
fasta headers, 1-based global first-fragment index).  Fresh HOST matrices --
retain freely.
"""
struct TopKBatch{T<:Union{Float16,Float32}}
    vals::Matrix{T}            # (nb, k), descending
    locs::Matrix{Int32}        # (nb, k), global column ids into B
    norms::Matrix{Float32}     # (m, nb), passthrough
    heads::Vector{String}
    first::Int
end

# batch-height multiple required by the GEMM engines (mma: the BM = 64 tile;
# cuBLASLt fp8: the 16-byte-multiple leading dimension)
padmul(engine::Symbol) = engine === :mma ? 64 : 16

"""
Reusable per-batch fp8 GEMM+top-k state for one database matrix:
gemmtopk(fp8)'s double-buffered chunk loop with everything hoisted that
flowtopk hoisted -- the two (rows_cap, w) C chunk buffers (Tc = Float16 for
cout = :f16, the fast path), the two non-blocking streams, the two event
pairs -- plus the engine's (rows_cap, rdim) e4m3 batch staging `a8` and the
engine selection (`engine = :mma | :lt`, `cout = :f16 | :f32`).
`rows_cap` is ROUNDED UP to the engine's batch-height multiple (`padmul`);
every batch is computed at that full height and the pad rows sliced off
after the D2H, so both engines always see an aligned height and a common A/C
leading dimension.  Requires 1 <= k <= w <= N (plus the engine's shape
asserts).  Compiles the C++ mma kernel once (nvcc -> PTX -> CuModule).
"""
struct TopKEngine{Tc<:Union{Float16,Float32}}
    B::CuMatrix{F8}            # (rdim, N) resident e4m3 database
    buf1::CuMatrix{Tc}         # the two (rows_cap, w) C chunk buffers,
    buf2::CuMatrix{Tc}         #   double-buffered: chunk i uses mod1(i, 2)
    a8::CuMatrix{F8}           # (rows_cap, rdim) quantized batch staging
    a32::CuMatrix{Float32}     # its fp32 scratch (pad rows kept zero -- see
                               # upload_fp16_as_f8!'s linear-layout note)
    k::Int                     # top-k size
    w::Int                     # B-column chunk width
    engine::Symbol             # :mma (C++ mma.sync kernel) | :lt (cuBLASLt)
    cout::Symbol               # C chunk type: :f16 | :f32
    rows_cap::Int              # padded batch-height capacity (a padmul multiple)
    sg::CuStream               # GEMM stream
    st::CuStream               # top-k stream
    evg::NTuple{2,CuEvent}     # evg[b]: gemm of the chunk using buffer b done
    evt::NTuple{2,CuEvent}     # evt[b]: top-k of the chunk using buffer b done
    segs::Int                  # column segments per row for the top-k kernel
end

function TopKEngine(B::CuMatrix{F8}; k::Int = 20, w::Int = 2^16,
                    rows_cap::Int = 2^13, segs::Int = 8,
                    engine::Symbol = :lt, cout::Symbol = :f16)
    rdim, N = size(B)
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
    buf1 = CuMatrix{Tc}(undef, rows_cap, w)
    buf2 = CuMatrix{Tc}(undef, rows_cap, w)
    a8 = CuMatrix{F8}(undef, rows_cap, rdim)
    a32 = CUDA.zeros(Float32, rows_cap, rdim) # zeroed once; pad rows stay zero
    sg = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    st = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    TopKEngine(B, buf1, buf2, a8, a32, Int(k), Int(w), engine, cout, rows_cap,
               sg, st, (CuEvent(), CuEvent()), (CuEvent(), CuEvent()), Int(segs))
end

"""
H2D-upload a host fp16 embedding batch and quantize it to e4m3 (satfinite)
into rows 1:nb of `a8`: one contiguous fp16 upload, one fused fp16 -> fp32
broadcast and gemmtopkfp8's f32_to_f8! converter kernel (~180 MiB of device	raffic at the default batch -- microseconds).

LINEAR-LAYOUT CONTRACT (the one way this can silently break): the converter
is 1-D over the WHOLE (cap, rdim) staging, so `a32` must have exactly `a8`'s
shape -- a (nb, rdim) scratch would scatter the quantized values across
full columns instead of rows 1:nb (column-major).  Rows nb+1:cap of `a32`
must be ZERO (they pass through the converter into `a8`'s pad rows), so a
batch padded up to rows_cap produces exact-zero score rows that the caller
slices off after the D2H.  The engine's `a32` is zeroed once at construction
and only rows 1:nb are ever overwritten.  Runs on the current stream;
`batch_gemm_topk!` device-synchronizes before its chunk loop, which orders
it before the first GEMM.
"""
function upload_fp16_as_f8!(a8::CuMatrix{F8}, a32::CuMatrix{Float32},
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

upload_fp16_as_f8!(eng::TopKEngine, embeds::Matrix{Float16}) =
    upload_fp16_as_f8!(eng.a8, eng.a32, embeds)

# GEMM dispatch for one chunk: D = A * B chunk (base pointer Bp, wi columns,
# the parent's column stride), into the engine's CONTIGUOUS buffer (so both
# engines see the real leading dimension rows_cap -- no views, ever).
@inline function _gemm_chunk!(eng::TopKEngine, buf::CuMatrix, Bp::CuPtr{UInt8},
                              wi::Integer)
    if eng.engine === :mma
        fp8_gemm_mma!(eng.cout, buf, eng.a8, Bp, wi, size(eng.B, 1))
    else
        fp8_gemm_lt!(buf, eng.a8, Bp, wi)
    end
    return buf
end

"""
One streamed batch: gemmtopk(fp8)'s double-buffered chunk loop against the
engine's resident B, with the running top-k RESET first (batches are
independent query sets).  The quantized batch must already be staged in
`eng.a8` (run `upload_fp16_as_f8!(eng, embeds)` first); the FULL
rows_cap height is computed (DESIGN 3) -- `D_val`/`D_loc` are
(rows_cap, k) and rows nb+1:rows_cap are discarded by the caller.
`overlap = false` runs sequentially on the default stream (benchmark
baseline; bitwise identical results).  Returns with the device synchronized.
"""
function batch_gemm_topk!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                          eng::TopKEngine; overlap::Bool = true)
    M, KA = size(eng.a8)
    rdim, N = size(eng.B)
    @assert KA == rdim "A's inner dim must match B's rows"
    @assert M == eng.rows_cap "engine staging must be (rows_cap, rdim)"
    @assert size(D_val) == (M, eng.k) && size(D_loc) == (M, eng.k)

    fill!(D_val, typemin(Float32)) # reset the running top-k; the kernel merges into it
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    sg = overlap ? eng.sg : CUDA.default_stream()
    st = overlap ? eng.st : sg
    for i in 1:cld(N, eng.w)
        b = mod1(i, 2)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        buf = b == 1 ? eng.buf1 : eng.buf2
        # base pointer of B column lo; the column stride stays the parent rdim
        Bp = byteptr(pointer(eng.B), (lo - 1) * rdim)

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(eng.evt[b])   # buffer b's previous top-k finished
            _gemm_chunk!(eng, buf, Bp, wi)   # launches on the current stream (= sg)
            CUDA.record(eng.evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b])            # C chunk is ready
            launch_seg_rowtopk_merge!(D_val, D_loc,
                                      wi == eng.w ? buf : @view(buf[:, 1:wi]),
                                      lo - 1, eng.k; segs = eng.segs)
            CUDA.record(eng.evt[b])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

# ==============================================================================
# The streamed flow
# ==============================================================================

# Task-level error router: same contract as ropeflowreal's
# `_rope_real_stream_error` (InvalidStateException = benign teardown, never a
# data error; real errors print immediately, land in err_out, first error
# wins, and every channel is closed so all stages unwind).
function _topkfp8_flow_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "flowtopkfp8: $(who) failed: ",
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
# a dense temp (host-dest broadcasts of device arrays scalar-index -- measured,
# not available), then one contiguous copyto!.  Fresh host array.
function _gather_rows(v)
    h = Matrix{eltype(v)}(undef, size(v))
    t = CuMatrix{eltype(v)}(undef, size(v))
    t .= v
    copyto!(h, t)
    return h
end

"""
    topk_flow(source, eng::TopKEngine; out_cap = 2, err_out, tasks_out)
        -> Channel{TopKBatch{Float32}}

Consume `source` -- any iterable of fp16 rope batches (anything with
`.embeds/.norms/.heads/.first`, e.g. a `Channel{RopeRealBatch{Float16}}` or a
pre-materialized `Vector{RopeRealBatch}`) -- and, for every batch, quantize
the embeddings to e4m3 on the GPU and run `batch_gemm_topk!` against the
engine's resident fp8 database, emitting
`TopKBatch(vals, locs, norms, heads, first)` with fresh HOST matrices (rows
1:nb -- the padded rows_cap height is sliced off here).  The stage runs on
its own task; batches flow through a `Channel(out_cap)` so the upstream rope
pipeline keeps encoding ahead (backpressure: a slow consumer of the results
throttles the whole pipeline).  Batch eltype must be Float16 (build the rope
stream with `fp16 = true`); batch height must be <= the engine's rows_cap.

Consume to the end (the channel closes itself) or `close(ch)` early: the
stage closes `source` when it is a channel, which tears the rope pipeline
down quietly (ropeflowreal's contract).  Real failures land in `err_out[]`
after the stream ends short; `tasks_out` receives this stage's task (pass the
same Ref to the rope stream to collect its two tasks as well, so a caller
that stopped early can `wait` all of them).
"""
function topk_flow(source, eng::TopKEngine;
                   out_cap::Int = 2,
                   err_out::Ref{Any} = Ref{Any}(nothing),
                   tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k = eng.k
    cap = eng.rows_cap
    out = Channel{TopKBatch{Float32}}(out_cap)
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
                # upload/quantize on the GEMM stream (flowtopk's pattern): this
                # keeps multi-ms kernel queues OFF the default stream -- a busy
                # default stream takes CUDACore's nonblocking_synchronize slow
                # path, and its worker crashes on the legacy stream's ctx =
                # nothing (context!(nothing) MethodError, fatal; see the header)
                CUDA.stream!(eng.sg) do
                    upload_fp16_as_f8!(eng, emb)
                end
                D_val = CuMatrix{Float32}(undef, cap, k)
                D_loc = CuMatrix{Int32}(undef, cap, k)
                batch_gemm_topk!(D_val, D_loc, eng)
                vals = _gather_rows(@view(D_val[1:nb, :])) # fresh host matrices,
                locs = _gather_rows(@view(D_loc[1:nb, :])) # pad rows sliced off
                D_val = D_loc = nothing # device pool reuses them next batch
                put!(out, TopKBatch(vals, locs, bat.norms, bat.heads, bat.first))
            end
        catch err
            # close the rope pipeline on BOTH paths: on the benign teardown it
            # propagates the unwind upstream; on a REAL error the guarded path
            # below would otherwise leave the rope stage pumping into this dead
            # task forever and the caller's task join would hang (first
            # exercised by e2ecuts's overflow error)
            _topkfp8_flow_error(err, err_out, out; who = "topkfp8")
            source isa AbstractChannel && close(source)
        finally
            close(out) # also the early-close path (close on closed is a no-op)
        end
    end
    append!(tasks_out[], [t_topk])
    return out
end

"""
    rope_topk_stream(re::RopeEncoder, file::String, B::CuMatrix{F8};
                     k = 20, w = 2^15, batch_size = 2^13, rows_cap = batch_size,
                     normalize = 0, parts = nthreads(), in_cap = 2, out_cap = 2,
                     topk_out_cap = 2, segs = 8, engine = :lt, cout = :f16,
                     progress = false, err_out, tasks_out)
                     -> Channel{TopKBatch{Float32}}

The full fp8 flow on a fasta `file`: stream-rope-encode the fragments
(`rope_encode_real_stream` with `fp16 = true` -- the fragment length is the
encoder's `re.k`) and process every batch with the engine's double-buffered
fp8 GEMM + fused per-row top-k against the RESIDENT database
`B ∈ e4m3^(rdim x N)` (rdim must equal the encoder's 2*m*4^c; its columns
are the reference vectors, assumed normalized).  `k` is the TOP-K SIZE, `w`
the B-column chunk width, `batch_size` the fragments per streamed batch,
`rows_cap` the engine's batch-height capacity (rounded up to the engine's
padmul; C buffers: 2 * rows_cap * w * (cout == :f16 ? 2 : 4) bytes); a batch
taller than `rows_cap` is an error, so cap it at the file's read count when
`batch_size` is an over-estimate.  Emits `TopKBatch(vals (nb,k) Float32,
locs (nb,k) Int32 global column ids into B, norms, heads, first)` with fresh
host matrices; `nb <= batch_size` (only the last batch is partial).
Fragment order within the stream is the parsers' arbitrary interleaving
(headers make every row self-describing); per-row results are deterministic.

Consume to the end or `close(ch)` early (quiet teardown of the whole
pipeline); `err_out`/`tasks_out` wire both stages together (three tasks).
"""
function rope_topk_stream(re::RopeEncoder, file::String, B::CuMatrix{F8};
                          k::Int = 20, w::Int = 2^16,
                          batch_size::Int = 2^13, rows_cap::Int = batch_size,
                          normalize::Int = 0, parts::Int = Threads.nthreads(),
                          in_cap::Int = 2, out_cap::Int = 2, topk_out_cap::Int = 2,
                          segs::Int = 8, engine::Symbol = :lt, cout::Symbol = :f16,
                          progress::Bool = false,
                          err_out::Ref{Any} = Ref{Any}(nothing),
                          tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(B, 1) == rdim ||
        throw(ArgumentError("B is $(size(B, 1)) x $(size(B, 2)); the encoder's real " *
                            "embedding dim is 2*m*4^c = $rdim"))
    eng = TopKEngine(B; k, w, rows_cap, segs, engine, cout)
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return topk_flow(rope_ch, eng; out_cap = topk_out_cap, err_out, tasks_out)
end

# ==============================================================================
# Correctness.
# ==============================================================================

# The e4m3 quantize/upload path: device bits must equal the host F8.()
# conversion (the CPU reference models quantization with exactly that), and
# the pad rows must come out zeroed.
function _check_quantize_path()
    Random.seed!(7)
    nb, rdim, cap = 96, 64, 128
    emb = randn(Float16, nb, rdim)
    a8 = CuMatrix{F8}(undef, cap, rdim)
    a32 = CUDA.zeros(Float32, cap, rdim) # the linear-layout contract's scratch
    fill!(a8, F8(1.0)) # poison everything not covered by the converter
    upload_fp16_as_f8!(a8, a32, emb)
    got = Array(a8)
    ref = F8.(Float32.(emb))
    @assert all(got[1:nb, :] .== ref) "device e4m3 quantization != host F8.() conversion"
    @assert all(iszero, @view(got[nb+1:cap, :])) "pad rows not zeroed"
    @info "  e4m3 quantize/upload path OK (device == host F8.(), pad rows zeroed)"
    return nothing
end

# exact multiset check against a CPU sort for the segmented (Tc, Tv) kernel,
# for all C element types (fp16, fp32 and raw e4m3 bytes), over several shapes
# including ragged chunk sizes (w ∤ N), a single row, the exact full-matrix
# chunk size, engine-style strided parent views (ld = w viewed to wi < w),
# w == k and an empty-segment case (w = 25 with segs = 8 -> segment 8 empty)

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, score the top-k against the provenance
# (from legacy e2ehuman.jl -- the fp8 variant; NOT co-includable names apply)
# ------------------------------------------------------------------------------

"""
    map_and_count(re, reads_file, B, db_heads, db_starts, dbk; ktop, w,
                  batch_size, segs, engine, normalize, progress)
        -> (total, correct, rank_hist, seconds)

Run the full fp8 flow (`rope_topk_stream`: fasta -> rope fp16 -> e4m3 GEMM +
per-row top-k) over `reads_file` and score every row against its provenance
header (either format of `parse_read_head`): CORRECTLY MAPPED iff the true
window [start, start+len-1] in record src intersects at least one of the ktop
returned database windows (db_heads[c], db_starts[c], length dbk).  The
reader has already trimmed every kept record to exactly dbk bases (records
shorter than dbk are skipped), so the scored fragment window is dbk-long in
both formats -- for the colon format (no header length) it is the dbk-long
window at the (1-based-normalized) header start.  For every correctly mapped
read the BEST (maximum) intersection LENGTH over ALL its ktop returned hits
is also kept (when > 0) -- not just the first intersecting hit's.  Returns
the processed-read count, the number of correctly mapped records, the
first-hit rank histogram (rank_hist[j] = reads whose first intersecting hit
was rank j), the best-intersection-length total count (inter_sum /
inter_cnt; the average as a share of the fragment length is what the run
mode reports) and the processed-batch count nb.  `stream` selects the
batch producer (default: the resident-database `rope_topk_stream`;
e2ecompact passes its chunk-streaming variant with the same contract).
"""
function map_and_count(re::RopeEncoder, reads_file::String, B::AbstractMatrix{F8},
                       db_heads::Vector{String}, db_starts::Vector{Int}, dbk::Int;
                       ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, normalize::Int = 0,
                       progress::Bool = false,
                       stream::Function = (re, f, B; kw...) ->
                                           rope_topk_stream(re, f, B; kw...))
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(B, 2)
    # per-column record id: the .bin's heads repeat a handful of unique
    # record headers over millions of columns -- intern them once.  The colon
    # eval provenance names records by ACCESSION (first whitespace token of
    # the '>' header, sans '>'), so the same ids are also keyed by accession.
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
    inter_sum = inter_cnt = 0
    rank_hist = zeros(Int, ktop)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in stream(re, reads_file, B; k = ktop, w,
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
            # length -- the reader trimmed every kept record to dbk bases
            h.len === nothing || h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rid = get(rec_id_acc, h.src, Int32(0)) # both formats name by accession
            hit = 0
            ilen = 0
            @inbounds for j in 1:ktop
                c = Int(bat.locs[r, j])
                # read window [h.start, h.start+dbk-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid &&
                   h.start <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= h.start + dbk - 1
                    hit == 0 && (hit = j) # first intersecting hit -> rank hist
                    # how MANY bases the two windows share (> 0 whenever the
                    # overlap test above fired); keep the BEST (max) overlap
                    # over ALL ktop hits, not just the first intersecting one
                    il = min(h.start + dbk - 1, db_starts[c] + dbk - 1) -
                         max(h.start, db_starts[c]) + 1
                    il > ilen && (ilen = il)
                end
            end
            total += 1
            if hit > 0
                correct += 1
                rank_hist[hit] += 1
                ilen > 0 && (inter_sum += ilen; inter_cnt += 1)
            end
        end
        @printf("  batch %2d: +%d reads, %d/%d correctly mapped so far\n",
                nb_bat, nb, correct, total)
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    return (total = total, correct = correct, rank_hist = rank_hist, seconds = t,
            inter_sum = inter_sum, inter_cnt = inter_cnt, nb = nb_bat)
end

# ------------------------------------------------------------------------------
# Tiny self-test fixture: a 1M-base two-record reference (built once), an
# indexreal-format index built from it, and err = 0 sampled reads
# ------------------------------------------------------------------------------


# shapes AND leading dimensions -> bitwise-identical C values on a
# deterministic engine), materialized on the host for an exact CPU-reference
# comparison of the merged top-k.
function _ref_topk_chunked(eng::TopKEngine)
    rdim, N = size(eng.B)
    Cbuf = CuMatrix{eltype(eng.buf1)}(undef, eng.rows_cap, eng.w)
    Ch = Matrix{Float32}(undef, eng.rows_cap, N)
    for i in 1:cld(N, eng.w)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        _gemm_chunk!(eng, Cbuf, byteptr(pointer(eng.B), (lo - 1) * rdim), wi)
        Ch[:, lo:lo+wi-1] .= _gather_rows(@view(Cbuf[:, 1:wi]))
    end
    return Ch
end

function _assert_topk_rows(Ch::Matrix{Float32}, Dv::Matrix{Float32},
                           Dl::Matrix{Int32}, k::Int)
    M, N = size(Ch)
    @assert size(Dv) == (M, k) && size(Dl) == (M, k) "output shape mismatch"
    for r in 1:M
        ref = partialsort!(Vector{Float32}(Ch[r, :]), 1:k; rev = true)
        got = Dv[r, :]
        @assert issorted(got; rev = true) "row $r: output not sorted"
        @assert sort(got; rev = true) == ref "row $r: wrong top-$k values"
        @assert all(>(0), Dl[r, :]) && Ch[r, Dl[r, :]] == got "row $r: wrong locations"
    end
    return nothing
end

# Engine vs an exact same-engine chunked reference over several shapes per
# engine (lt: ragged chunks + any padded batch height; mma: divisible w | N),
# both couts; each shape includes a single-row/padded-height case.  Then the
# real geometry (2^13 x 2^15) for both engines: values AND locations bitwise,
# the dequantized-fp32 gemv spot check (gemmtopkfp8's spot_check), a tail
# batch on a REUSED engine, and pipelined == sequential determinism.
function _check_engine_gemm_topk(; k::Int = 20, seed = 123,
                                 engines = (:lt, :mma), couts = (:f16, :f32))
    CUDA.seed!(seed)
    lt_geo = ((64, 64, 208, 80), (32, 257, 144, 144), (64, 1, 4096, 512))
    mma_geo = ((64, 64, 256, 64), (128, 1, 2^12, 2^10))
    for engine in engines, cout in couts
        for (rdim, nb, N, w) in (engine === :lt ? lt_geo : mma_geo)
            A = randn(Float16, nb, rdim) # the rope stream's output analogue
            B8 = randn_fp8!(CuMatrix{F8}(undef, rdim, N))
            eng = TopKEngine(B8; k, w, rows_cap = nb, engine, cout)
            upload_fp16_as_f8!(eng, A)
            D_val = CuMatrix{Float32}(undef, eng.rows_cap, k)
            D_loc = CuMatrix{Int32}(undef, eng.rows_cap, k)
            batch_gemm_topk!(D_val, D_loc, eng)
            Ch = _ref_topk_chunked(eng)
            _assert_topk_rows(Ch, Array(D_val), Array(D_loc), k)
            @info "  engine exact check OK (engine=$engine, cout=$cout, rdim=$rdim, " *
                  "nb=$nb, N=$N, w=$w, rows_cap=$(eng.rows_cap), k=$k)"
            eng = B8 = nothing
            GC.gc(); CUDA.reclaim()
        end
    end

    # real geometry: spot check vs dequantized fp32 gemv, tail batch on a
    # reused engine, pipelined == sequential
    rdim, nb, N, w = 128, 4096, 2^15, 2^13
    for engine in engines
        A = randn(Float16, nb, rdim)
        B8 = randn_fp8!(CuMatrix{F8}(undef, rdim, N))
        eng = TopKEngine(B8; k, w, rows_cap = nb, engine) # cout = :f16 fast path
        upload_fp16_as_f8!(eng, A)
        D_val = CuMatrix{Float32}(undef, eng.rows_cap, k)
        D_loc = CuMatrix{Int32}(undef, eng.rows_cap, k)
        batch_gemm_topk!(D_val, D_loc, eng)
        Ch = _ref_topk_chunked(eng)
        _assert_topk_rows(Ch, Array(D_val), Array(D_loc), k)
        maxerr = spot_check(eng.a8, B8, D_val, D_loc, k; tol = 1.0)
        # tail batch (100 < rows_cap) on the REUSED engine
        A2 = randn(Float16, 100, rdim)
        upload_fp16_as_f8!(eng, A2)
        Dv2 = CuMatrix{Float32}(undef, eng.rows_cap, k)
        Dl2 = CuMatrix{Int32}(undef, eng.rows_cap, k)
        batch_gemm_topk!(Dv2, Dl2, eng)
        Ch2 = _ref_topk_chunked(eng)
        _assert_topk_rows(Ch2, Array(Dv2), Array(Dl2), k)
        # sequential (no overlap) must be bitwise identical
        upload_fp16_as_f8!(eng, A2)
        Dv3 = similar(Dv2)
        Dl3 = similar(Dl2)
        batch_gemm_topk!(Dv3, Dl3, eng; overlap = false)
        @assert Array(Dv3) == Array(Dv2) && Array(Dl3) == Array(Dl2) "overlap=false diverged"
        @info "  real-geometry check OK (engine=$engine, 4096 x 2^15, spot-check maxerr " *
              "$(round(maxerr; digits = 4)), tail batch, pipelined == sequential)"
        eng = B8 = nothing
        GC.gc(); CUDA.reclaim()
    end
    return nothing
end

# End-to-end: stream the fasta file through the full fp8 flow and compare the
# top-k against a float64 CPU reference that models the pipeline's
# quantization exactly: embeddings fp16 (the rope output) -> e4m3 (the stage
# input); the database is compared AS STORED (dequantized from the device).
# Remaining deviation: fp16 storage of the fp32-accumulated C (~1 ulp of
# |C| <= ~1) + accumulation-order noise -> atol 5e-3 (measured ~1e-3;
# flowtopk's fp16-accumulated pipeline needed 1e-2 for its ~3e-3).
function _check_topk_stream(re, path, B8::CuMatrix{F8}, Bh::Matrix{Float64};
                            k::Int, w::Int, batch_size::Int, rows_cap::Int,
                            normalize::Int, kfrag::Int, engine::Symbol = :lt,
                            refw::Dict{String,Vector{UInt32}},
                            sample::Int, seed = 13)
    rs = MersenneTwister(seed)
    N = size(B8, 2)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    nbatches = seen = 0
    heads_all = String[]
    for bat in rope_topk_stream(re, path, B8; k, w, batch_size, rows_cap,
                                normalize, engine, err_out = err, tasks_out = tks)
        nbatches += 1
        nb = length(bat.heads)
        @assert bat isa TopKBatch && eltype(bat.vals) == Float32 "batch type/eltype mismatch"
        @assert size(bat.vals) == (nb, k) && size(bat.locs) == (nb, k) "batch shape mismatch"
        @assert size(bat.norms) == (re.m, nb) "norms passthrough shape mismatch"
        @assert bat.first == seen + 1 "batch first-index mismatch"
        for r in 1:nb
            @assert issorted(@view(bat.vals[r, :]); rev = true) "row $r: not sorted"
            @assert all(lo -> lo in 1:N, @view(bat.locs[r, :])) "row $r: locations out of range"
        end
        if normalize == 0 # shared raw norm + partials: positive & finite
            @assert all(x -> isfinite(x) && x > 0, bat.norms) "norm passthrough not positive/finite"
        end
        atol = 5e-3
        for r in rand(rs, 1:nb, min(sample, nb))
            head = bat.heads[r]
            href, nrmref = _ref_rope_frag_real(_frag_codes(refw[head], kfrag), re; normalize)
            hrefq = Float64.(Float32.(F8.(Float16.(Float32.(href))))) # fp16 -> e4m3, as shipped
            scores = Bh' * hrefq # (N,) float64 reference scores over the STORED B
            refv = partialsort!(copy(scores), 1:k; rev = true)
            gotv = bat.vals[r, :]
            gotl = bat.locs[r, :]
            # norms passthrough vs the CPU reference (mode 0: only row 1 is the norm)
            for idm in (normalize == 0 ? (1:1) : (1:re.m))
                @assert isapprox(bat.norms[idm, r], nrmref[idm]; rtol = 1e-3) "norms passthrough mismatch ($(head), copy $idm)"
            end
            @assert isapprox(gotv, refv; atol, rtol = 0) "top-k values vs CPU reference mismatch ($(head))"
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
function run_flow_topk_test(; engines = (:lt, :mma))
    @show CUDA.name(device())
    @show nthreads()

    small, _big = ensure_data3()
    kfrag = 20_000
    k = 20
    dir = mktempdir(prefix = "flowtopkfp8_")

    # ==========================================================================
    # A. THE PIECES: quantize/upload path, the (Tc, Tv) segmented top-k kernel,
    #    and both GEMM engines vs exact references (gemmtopkfp8's GEMM unit
    #    tests re-run in this process, then the flow engine's chunk loop).
    # ==========================================================================
    @info "A. quantize path + segmented top-k kernel + fp8 engine checks"
    _check_quantize_path()
    test_seg_rowtopk_merge_kernel(k = k)
    test_fp8_gemm(engines = engines) # gemmtopkfp8's GEMM + chunked-ldb unit test
    _check_engine_gemm_topk(k = k, engines = engines)

    # ==========================================================================
    # B. FLOW end-to-end on the small generated file (2^13 reads x ~20 kb)
    #    vs the float64 CPU reference: batch layouts (tail, exact fit, single
    #    batch), normalize 0/1, a second encoder config -- all on the :lt
    #    engine (the small test N = 3000 admits no mma-legal w).  The mma
    #    engine runs the same check at real geometry (N = 2^15, w = 2^13,
    #    one exact 2^13 batch).  B_test's columns are normalized N(0,1)
    #    vectors quantized to e4m3, like the production database.
    # ==========================================================================
    @info "B. rope_topk_stream on the small file vs the float64 CPU reference"
    re = RopeEncoder(k = kfrag, s = 8, m = 4, c = 4) # rdim = 2^11
    rdim = 2 * re.m * 4^re.c
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k = kfrag, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == _V3_SMALL_READS "unexpected reference fragment count"

    N = 3000
    B_h = randn(Float32, rdim, N)
    B_h ./= sqrt.(sum(abs2, B_h; dims = 1))
    B8 = CuArray(F8.(B_h))
    Bh = Float64.(Float32.(Array(B8))) # the database AS STORED (dequantized)

    for batch_size in (3000, 2^13, 10^6) # tail, exact fit, single batch
        nb = _check_topk_stream(re, small, B8, Bh; k, w = 512, batch_size,
                                rows_cap = min(batch_size, _V3_SMALL_READS),
                                normalize = 0, kfrag, engine = :lt, refw, sample = 4)
        @info "  normalize=0, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_topk_stream(re, small, B8, Bh; k, w = 512, batch_size = 3000,
                            rows_cap = 3000, normalize = 1, kfrag, engine = :lt,
                            refw, sample = 4)
    @info "  normalize=1, batch_size=3000 OK ($nb batches)"

    # other encoder config through the whole flow (rdim = 2*1*4^4 = 512)
    re2 = RopeEncoder(k = kfrag, s = 5, m = 1, c = 4)
    rdim2 = 2 * re2.m * 4^re2.c
    N2 = 777
    B_h2 = randn(Float32, rdim2, N2)
    B_h2 ./= sqrt.(sum(abs2, B_h2; dims = 1))
    B8_2 = CuArray(F8.(B_h2))
    nb = _check_topk_stream(re2, small, B8_2, Float64.(Float32.(Array(B8_2))); k, w = 512,
                            batch_size = 3000, rows_cap = 3000, normalize = 0,
                            kfrag, engine = :lt, refw, sample = 4)
    @info "  config (s=5, m=1, c=4) end-to-end OK ($nb batches)"

    # the mma engine end-to-end at real geometry (w | N, one exact batch)
    if :mma in engines
        Bm = randn_fp8_normcols!(CuMatrix{F8}(undef, rdim, 2^15))
        nb = _check_topk_stream(re, small, Bm, Float64.(Float32.(Array(Bm))); k, w = 2^13,
                                batch_size = 2^13, rows_cap = 2^13, normalize = 0,
                                kfrag, engine = :mma, refw, sample = 4)
        @info "  mma engine, real geometry (N=2^15, w=2^13), end-to-end OK ($nb batches)"
        Bm = nothing
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
    ch = rope_topk_stream(re, small, B8; k, w = 512, batch_size = 3000,
                          rows_cap = 3000, err_out = err, tasks_out = tks)
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
    nb = _check_topk_stream(re, small, B8, Bh; k, w = 512, batch_size = 3000,
                            rows_cap = 3000, normalize = 0, kfrag, engine = :lt,
                            refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    # ==========================================================================
    # D. empty file -> zero batches, no error.
    # ==========================================================================
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    err = Ref{Any}(nothing)
    n = 0
    for _ in rope_topk_stream(re, empty_fasta, B8; k, w = 512, err_out = err)
        n += 1
    end
    @assert n == 0 && err[] === nothing
    @info "D. empty file OK (0 batches)"

    @info "ALL FLOWTOPKFP8 CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks: per-chunk GEMM engine x cout head-to-head at the real batch
# geometry (plus the quantize cost, the segmented top-k sweep and an
# in-process fp16 GEMM reference line), then end-to-end flow (fasta -> rope
# fp16 -> fp8 GEMM+top-k) on the big file, the compute-only stage (rope
# excluded, embeddings replayed), the sequential baseline and the w / batch
# sweeps.  B is the production shape 2^11 x 2^21 e4m3 with column-normalized
# N(0,1) columns.  Run under `julia -t N`.
# ==============================================================================

# host-timed, device-synchronized measurement of a (stream-offloaded) GPU
# loop.  CUDA.@elapsed cannot be used here: its events bracket the CURRENT
# stream, so it measures ~nothing for work launched on eng.sg.
function bench_flow_topk(; reps = _reps(), engines = (:mma, :lt), sweep::Bool = true,
                          Nbits::Int = 21)
    isempty(k_gemm) && build_kernel() # B generation's converters share the PTX module
    small, big = ensure_data3()
    _warm_cache(big)
    kfrag = 20_000
    re = RopeEncoder(k = kfrag, s = 8, m = 4, c = 4)
    rdim = 2 * re.m * 4^re.c # 2^11
    N = 2^Nbits # B = rdim x 2^Nbits e4m3 (21 = 4 GiB production shape, 22 = 8 GiB)
    k = 20
    w = 2^16 # bench-measured optimum (see RESULTS): the fp8 lt GEMM is so
    #         fast (707 TFLOPS at M = 2^13) that per-chunk overhead pushes the
    #         optimum to the memory-capped 2^16 (2 x 8 GiB of C buffers; 2^17
    #         would need 32 GiB); flowtopk's fp16 optimum was 2^15
    rows = 2^13
    iters = 20
    dev = CUDA.device()
    @info "flowtopkfp8 benchmarks" gpu = CUDA.name(dev) vram_gb = round(CUDA.totalmem(dev) / 2^30; digits = 1) julia_threads = nthreads() reps engines
    @info @sprintf("B: %d x %d e4m3 (%.1f GiB), C_bat would be %.0f GiB (fp32), k=%d, w=%d, batch=%d x %d",
                   rdim, N, rdim * N / 2^30, rows * N * 4 / 2^30, k, w, rows, rdim)
    flops_total = 2.0 * _V3_BIG_READS * rdim * N # 2^50 FLOP over the big file
    nb_batches = cld(_V3_BIG_READS, rows)

    # ---- the resident database ----------------------------------------------
    @info "generating B on GPU (column-normalized N(0,1) -> e4m3)..."
    tB = @elapsed B = randn_fp8_normcols!(CuMatrix{F8}(undef, rdim, N))
    @info @sprintf("  B ready in %.1f s (device pool: %.2f GiB used)", tB, CUDA.used_memory() / 2^30)

    # ---- rope stage alone (accounting reference; flowtopk's fp16 line) -------
    _timed_min("rope stream only, no-op consumer fp16 (H2D+kernel+D2H)"; bytes = filesize(big), reps) do
        n = 0
        for nt in rope_encode_real_stream(re, big; k = kfrag, fp16 = true)
            n += length(nt.heads)
        end
        n
    end

    # one real fp16 batch of embeddings, replayed as the source for the
    # compute-only runs (flowtopk's trick: the rope stage hides completely)
    embeds = Matrix{Float16}(undef, 0, rdim)
    for nt in rope_encode_real_stream(re, big; k = kfrag, fp16 = true)
        embeds = nt.embeds
        break
    end
    @assert size(embeds) == (rows, rdim)
    heads1 = [string("frag_", i) for i in 1:rows]
    norms1 = fill(1f0, re.m, rows)
    src = [RopeRealBatch(embeds, norms1, heads1, (i - 1) * rows + 1) for i in 1:nb_batches]

    # ---- per-chunk micro-bench at the real batch geometry --------------------
    Bp1 = byteptr(pointer(B))
    @info "per-chunk micro-bench (quantized real batch, w=$w, M=$rows)..."
    for engine in engines, cout in (:f16, :f32)
        eng = TopKEngine(B; k, w, rows_cap = rows, engine, cout)
        upload_fp16_as_f8!(eng, embeds)
        buf = eng.buf1
        # everything queued here runs on the engine's GEMM stream: a busy
        # DEFAULT stream + sync takes CUDACore's slow path and crashes its
        # worker on the legacy stream's ctx = nothing (see topk_flow)
        CUDA.stream!(eng.sg) do
            _gemm_chunk!(eng, buf, Bp1, w) # warm (JIT + the lt heuristic for this shape)
        end
        CUDA.device_synchronize()
        tq = _timed_gpu(iters) do
            CUDA.stream!(eng.sg) do
                upload_fp16_as_f8!(eng, embeds)
            end
        end
        tg = _timed_gpu(iters) do
            CUDA.stream!(eng.sg) do
                _gemm_chunk!(eng, buf, Bp1, w)
            end
        end
        @printf("  quantize+upload: %.3f ms/batch | %s %s gemm %.3f ms/chunk (%.1f TFLOPS)\n",
                tq * 1e3, engine, cout, tg * 1e3,
                2.0 * rows * rdim * w / tg / 1e12)
        D_val = CuMatrix{Float32}(undef, rows, k)
        D_loc = CuMatrix{Int32}(undef, rows, k)
        seg_list = cout === :f16 ? (2, 4, 8) : (8,)
        for segs in seg_list # 12+ needs > 48 KiB shared memory (K = 20, Float32)
            launch_seg_rowtopk_merge!(D_val, D_loc, buf, 0, k; segs) # warm (JIT)
            CUDA.device_synchronize()
            tt = _timed_gpu(iters) do
                CUDA.stream!(eng.sg) do
                    launch_seg_rowtopk_merge!(D_val, D_loc, buf, 0, k; segs)
                end
            end
            @printf("    seg-rowtopk segs=%d on %s chunks: %.3f ms/chunk (%.0f GiB/s of C read)\n",
                    segs, cout, tt * 1e3, rows * w * sizeof(eltype(buf)) / tt / 2^30)
        end
        eng = nothing
        GC.gc(); CUDA.reclaim()
    end
    # in-process fp16 GEMM reference: cuBLAS mul! on the pre-quantization fp16
    # operands, same chunk shape (the top-k is engine-independent, so this
    # line is the honest same-process GEMM speedup denominator)
    A16 = CuMatrix{Float16}(undef, rows, rdim)
    copyto!(A16, embeds)
    B16 = CuMatrix{Float16}(undef, rdim, N)
    fp16_from_fp8!(B16, B)
    Cv16 = CuMatrix{Float16}(undef, rows, w)
    sg16 = CuStream(; flags = CUDA.STREAM_NON_BLOCKING) # keep queues off the default stream
    CUDA.stream!(sg16) do
        mul!(@view(Cv16[:, 1:w]), A16, @view(B16[:, 1:w]))
    end
    CUDA.device_synchronize()
    t16 = _timed_gpu(iters) do
        CUDA.stream!(sg16) do
            mul!(@view(Cv16[:, 1:w]), A16, @view(B16[:, 1:w]))
        end
    end
    @printf("  fp16 cuBLAS mul! (same shape): %.3f ms/chunk (%.1f TFLOPS)\n",
            t16 * 1e3, 2.0 * rows * rdim * w / t16 / 1e12)
    A16 = B16 = Cv16 = nothing
    GC.gc(); CUDA.reclaim()

    # ---- end-to-end: fasta -> rope fp16 -> fp8 GEMM+top-k -> TopKBatch drain -
    best = (first(engines), Inf)
    for engine in engines
        t_e2e = _timed_min("flow END-TO-END: fasta -> rope fp16 -> fp8 gemm+topk ($engine)"; bytes = filesize(big), reps) do
            n = 0
            for bat in rope_topk_stream(re, big, B; k, w, batch_size = rows, engine)
                n += size(bat.vals, 1)
            end
            n
        end
        @printf("  %s end-to-end: %d fragments -> %d batches in %.2f s  =>  %.1f TFLOPS sustained over 2^%.0f FLOP\n",
                engine, _V3_BIG_READS, nb_batches, t_e2e, flops_total / t_e2e / 1e12,
                log2(flops_total))
        best = t_e2e < best[2] ? (engine, t_e2e) : best
    end
    engine = best[1]
    @printf("  (best engine: %s; TFLOPS = 2^50 FLOP / wall time over the whole big file)\n", engine)

    # ---- compute-only: the rope stage excluded (embeddings replayed) --------
    t_cmp = Inf
    for eng_e in engines
        eng = TopKEngine(B; k, w, rows_cap = rows, engine = eng_e)
        t_cmp_e = _timed_min("flow COMPUTE-ONLY: fp8 gemm+topk, rope excluded ($eng_e)"; bytes = 0, reps) do
            n = 0
            for bat in topk_flow(src, eng)
                n += size(bat.vals, 1)
            end
            n
        end
        @printf("  %s compute-only: %.2f s  =>  %.1f TFLOPS  (flowtopk fp16 compute-only: 5.10 s; gemmtopkfp8 monolithic fp8: ~2.5 s)\n",
                eng_e, t_cmp_e, flops_total / t_cmp_e / 1e12)
        eng_e == engine && (t_cmp = t_cmp_e)
        eng = nothing
        GC.gc(); CUDA.reclaim()
    end

    # sequential baseline on the best engine (manual loop, overlap = false)
    eng = TopKEngine(B; k, w, rows_cap = rows, engine)
    D_val = CuMatrix{Float32}(undef, rows, k)
    D_loc = CuMatrix{Int32}(undef, rows, k)
    CUDA.device_synchronize(); t0 = time()
    for _ in 1:nb_batches
        CUDA.stream!(eng.sg) do
            upload_fp16_as_f8!(eng, embeds)
        end
        batch_gemm_topk!(D_val, D_loc, eng; overlap = false)
    end
    t_seq = time() - t0
    @printf("  compute-only sequential: %.2f s  (overlap speedup %.2fx)\n", t_seq, t_seq / t_cmp)

    # ---- batch-height sweep (compute-only, single runs) ----------------------
    if sweep
        # (at the fixed w = 2^15, NOT the default w = 2^16: batch = 2^14 with
        # w = 2^16 would need 2 x 16 GiB of C buffers and OOM)
        for nb in (2^11, 2^12, 2^14)
            eh = nb <= rows ? embeds[1:nb, :] : vcat(embeds, embeds)[1:nb, :]
            src_nb = [RopeRealBatch(eh, fill(1f0, re.m, nb),
                                    [string("frag_", i) for i in 1:nb], (i - 1) * nb + 1)
                      for i in 1:cld(_V3_BIG_READS, nb)]
            eng_nb = TopKEngine(B; k, w = 2^15, rows_cap = nb, engine)
            CUDA.device_synchronize(); t0 = time()
            n = 0
            for bat in topk_flow(src_nb, eng_nb)
                n += size(bat.vals, 1)
            end
            tw = time() - t0
            @printf("  batch=%5d (%3d batches, %4.0f MiB/buffer): %6.2f s  =>  %.1f TFLOPS\n",
                    nb, cld(_V3_BIG_READS, nb), nb * w * sizeof(eltype(eng_nb.buf1)) / 2^20,
                    tw, flops_total / tw / 1e12)
            eng_nb = nothing
            GC.gc(); CUDA.reclaim()
        end

        # ---- chunk-width sweep (compute-only, single runs) -------------------
        # (w trades C-buffer size against per-chunk launch/event overhead; the
        # default w = 2^16 line is the compute-only run above)
        for w2 in (2^13, 2^14, 2^15)
            eng_w = TopKEngine(B; k, w = w2, rows_cap = rows, engine)
            CUDA.device_synchronize(); t0 = time()
            n = 0
            for bat in topk_flow(src, eng_w)
                n += size(bat.vals, 1)
            end
            tw = time() - t0
            @printf("  w=2^%2d (%3d chunks/batch, %4.0f MiB/buffer): %6.2f s  =>  %.1f TFLOPS\n",
                    log2(w2), cld(N, w2), rows * w2 * sizeof(eltype(eng_w.buf1)) / 2^20,
                    tw, flops_total / tw / 1e12)
            eng_w = nothing
            GC.gc(); CUDA.reclaim()
        end
    end

    println("  (fp16 reference lines: rope-only 0.541 s and compute-only 5.10 s are")
    println("   flowtopk.jl's measurements on the same machine/file -- cross-process")
    println("   caveat applies; the fp16 cuBLAS GEMM line above IS same-process)")
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)

    GC.gc(); CUDA.reclaim()
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    engstr = getflag("engine", "both")
    engines = engstr == "mma" ? (:mma,) : engstr == "lt" ? (:lt,) : (:mma, :lt)
    nbits = parse(Int, getflag("Nbits", "21"))
    quick = "--quick" in ARGS
    sweep = !("--no-sweep" in ARGS) && !quick
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    reps = quick ? 1 : _reps()
    mode == "gen" && ensure_data3()
    mode == "test" && run_flow_topk_test(engines = engines)
    mode == "bench" && bench_flow_topk(reps = reps, engines = engines, sweep = sweep,
                                       Nbits = nbits)
    mode == "all" && (ensure_data3(); run_flow_topk_test(engines = engines);
                      bench_flow_topk(reps = reps, engines = engines, sweep = sweep,
                                      Nbits = nbits))
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
