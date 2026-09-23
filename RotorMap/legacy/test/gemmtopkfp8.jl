# gemmtopkfp8.jl — batched fp8 (e4m3) GEMM + per-row top-k, double-buffered on
# two streams, with the GEMM done by C++ in Julia: an nvcc-compiled mma.sync
# tensor-core kernel (PTX → CuModule → CUDA.cudacall) or a raw-ccall cuBLASLt
# fp8 path — the two engines of gemmfp8cpp.jl, benchmarked head-to-head here.
#
# PROBLEM — same as gemmtopk.jl, in fp8:
#   A ∈ e4m3^(2^17 × 2^11), B ∈ e4m3^(2^11 × 2^21), both resident in VRAM.
#   C = A*B cannot be materialized (2^17 × 2^21 would be 0.5–1 TiB).
#   Wanted: the k=20 largest values of each row of C (over the 2^21 dimension)
#   and their locations → D_val ∈ Float32^(2^17 × k), D_loc ∈ Int32^(2^17 × k).
#   (Orientation note and transposed variant: see gemmtopk.jl's header.)
#
# DESIGN
#   1. Batching axis + running merge: unchanged from gemmtopk.jl. Chunk along
#      columns of B (C_i = A·B[:, chunk], 2^17 × w) and merge each chunk's
#      per-row top-k into the running top-k in D_val/D_loc (one fused kernel
#      per chunk, rowtopk_merge_kernel!).
#   2. GEMM engine. mul! has no fp8 path in CUDA.jl, so the GEMM is the
#      hand-written mma.sync tensor-core kernel from gemmfp8cpp.jl (64×64
#      tile, K-step 32, m16n8k32.row.col.f32.e4m3.e4m3.f32, fp32
#      accumulators, ~2x the fp16 cuBLAS throughput on this RTX 5090),
#      compiled with nvcc into PTX and launched via CUDA.cudacall. Two
#      changes vs gemmfp8cpp.jl:
#        a) `ldb`. B is consumed in column chunks, so the kernel takes the B
#           column stride (the parent K) and the chunk's base pointer. The
#           offset (lo-1)*K stays 16-byte aligned for our shapes, as required
#           by the kernel's uint4 loads. The unit test checks the chunked
#           path bitwise against a whole-matrix GEMM of the same data.
#        b) output type. The fp32 accumulators are stored as fp32
#           (cout=:f32), fp16 (cout=:f16) or e4m3 (cout=:f8; satfinite
#           store). :f16 halves the top-k read traffic and the buffer
#           memory — the top-k scan is element-issue-bound rather than
#           bandwidth-bound (fp16 and fp32 scans run at nearly the same
#           ELEMENT rate), so :f16 is the sweet spot: enough bits for robust
#           top-k locations, half the traffic of fp32. fp16 *storage* of
#           fp32-accumulated sums is still far more accurate than
#           gemmtopk.jl's fp16 accumulation (only the final rounding is
#           lost, ~ulp of |C|). cout=:f8 goes further and quantizes C
#           itself to e4m3 (~6% relative rounding) — measured SLOWER than
#           :f16 (the e4m3 decode costs more than the saved bytes) and it
#           shuffles near-tie top-k locations (11.8% location agreement);
#           kept as an experiment, not a default (see MEASURED RESULTS).
#      The raw-ccall cuBLASLt fp8 engine of gemmfp8cpp.jl (engine=:lt) is
#      included for an in-process comparison; it needs the matching 595
#      driver userland on LD_LIBRARY_PATH (see gemmfp8cpp.jl's header).
#      MEASURED VERDICT: :lt wins at these chunk shapes (657 vs 208 TFLOPS)
#      and is the default fast path; see MEASURED RESULTS for why the
#      hand-written kernel loses its gemmfp8cpp.jl advantage here.
#      Restrictions of the mma kernel: M, w multiples of 64; K multiple of
#      32; and w must divide N (no ragged final chunk) — true for all shapes
#      used here. engine=:lt also handles ragged chunks.
#   3. Top-k kernel. rowtopk_merge_kernel! is generalized from (T) to
#      (Tc, Tv): C chunks are read as Tc ∈ {Float16, Float32, UInt8}; the
#      register reservoir and D_val are Tv = Float32. Tc=UInt8 chunks hold
#      raw e4m3 bytes (cout=:f8), decoded on the fly by _scanval (exact,
#      ~10 integer ops from the PyTorch bit-twiddling decode). Sentinel
#      subtlety: out-of-range loads are set to typemin(Tv) *directly* —
#      converting typemin(Tc) instead would let e.g. Float16's -65504 into a
#      Float32 reservoir whose tracked minimum/threshold is typemin(Float32)
#      = -3.4e38, so the sentinel would wrongly enter the reservoir
#      (`sentinel > thresh` must be false). D_val moved to Float32 because a
#      single e4m3 product can reach 448^2 ≈ 2·10^5 > fp16 max. Reservoir/
#      merge mechanics (unsorted minimum-replace reservoir, TOPK_UNROLL
#      software pipelining, sort once, two-pointer merge) are unchanged from
#      gemmtopk.jl.
#   4. Pipeline. Same event choreography as gemmtopk.jl: two C buffers, two
#      non-blocking streams, topk(i) waits gemm(i), gemm(i+2) waits topk(i);
#      overlap=false runs sequentially on the default stream. cudacall
#      launches on the current (task-local) stream, so inside CUDA.stream!(sg)
#      it lands on the GEMM stream (validated by the determinism check).
#   5. Numerics/validation. e4m3×e4m3 products are exact in fp32 and the mma
#      accumulates in fp32, so the spot-check reference is an fp32 gemv over
#      the dequantized operands; the expected deviation is storage-only
#      (~1 fp16 ulp of |C| for cout=:f16, ~1e-4 for cout=:f32) — orders
#      tighter than the fp16 script's 0.83. Data generation converts fp32
#      randn → e4m3 on-GPU via cuda_fp16.h's __nv_cvt_float_to_fp8
#      (satfinite); DLFP8Types' host conversions are not assumed to compile
#      into GPU broadcast kernels, and an e4m3→fp32 converter kernel serves
#      the dequantized references and the fp16 baseline conversion.
#   6. In-process fp16 baseline (--baseline16). gemmfp8cpp.jl warns absolute
#      numbers drift between runs (thermals, operand bit patterns), so the
#      fp16→fp8 speedup is measured in the same process: A/B are converted
#      chunk-wise to fp16 and the identical pipeline runs with cuBLAS mul!.
#
# --------------------------------------------------------------------------
# ARCHITECTURE AT A GLANCE (per chunk i; w columns of B, w | N)
#
#   gemm_topk_fp8!(D_val, D_loc, A, B; k, w, overlap, threads, engine, cout)
#     buf = mod1(i, 2) alternates the two M×w C buffers; sg = GEMM stream,
#     st = top-k stream; events evg[buf] ("chunk written") / evt[buf]
#     ("buffer scanned") enforce exactly two dependencies per chunk:
#
#       stream sg (GEMM)                          stream st (top-k)
#       --------------------------------         --------------------------------
#       wait evt[buf]                (i > 2)
#       gemmf!(Cbuf, Bp, wi)              evg ──▶ wait evg
#         engine=:mma: cudacall mma.sync          rowtopk_merge_kernel!(C → D)
#         engine=:lt : cublasLtMatmul cc      [record evt[buf]]
#       [record evg[buf]]
#
#     Bp = byteptr(pointer(B), (lo-1)*K) is the chunk base pointer; the
#     column stride stays the parent K for both engines (cuBLASLt's B layout
#     is col-major K×wi with ld=K over the same base pointer).
#
#   File map: CU_SRC / build_kernel              C++ kernels: fp8 GEMM (fp32 &
#                                              fp16 out) + e4m3 converters
#             fp8_gemm_mma! / fp8_gemm_lt!      the two GEMM engines
#             rowtopk_merge_kernel!/launch_rowtopk_merge!  fused top-k + merge
#             gemm_topk_fp8! / gemm_topk_fp16!  pipelines (:mma/:lt / cuBLAS)
#             randn_fp8! / fp16_from_fp8!       chunked test data / conversion
#             test_rowtopk_merge_kernel         CPU-reference unit test
#             test_fp8_gemm                     GEMM + chunked-ldb unit test
#             spot_check                        fp32 gemv reference on rows
#             bench_gemm / main                 micro-bench + driver/sweeps
#
# MEASURED RESULTS (RTX 5090, full problem: 2^17×2^11 × 2^11×2^21 e4m3, k=20)
#   best config .... engine=:lt, cout=:f16, w=2^13: 2.68 s → 420 TFLOPS;
#                    w=2^14: 2.53 s → 445 TFLOPS sustained over the 2^50 FLOP
#   GEMM stage ..... lt :f16 6.69 ms/chunk at w=2^13 (657 TFLOPS; :f32 out is
#                    20% slower); the mma kernel only reaches ~208 TFLOPS at
#                    these shapes: every column block re-reads its A tile, and
#                    A (256 MiB) no longer fits L2 — unlike gemmfp8cpp.jl's
#                    4096³ bench where A was 16 MiB. cuBLASLt's internal tile
#                    scheduling/swizzle avoids the re-reads. (Fixing the mma
#                    kernel would need block-group swizzling; not worth it
#                    while lt exists.)
#   top-k+merge ... 3.2 ms/chunk at w=2^13 (621 GiB/s of fp16 C read —
#                    issue-bound, same as the fp16 script's 747 GiB/s); fp32
#                    chunks read faster per byte (1026 GiB/s, 3.9 ms) but move
#                    2x the bytes, so :f16 stays ahead end-to-end
#   accuracy ...... spot check vs fp32 gemv on dequantized operands:
#                    max err 0.062 for cout=:f16 (≈ fp16 storage ulp of
#                    |C|≈250), 4.6e-5 for cout=:f32; locations robust
#   vs fp16 ....... in-process fp16 baseline (cuBLAS mul!, same top-k kernel,
#                    w=2^13): 4.23 s → 266 TFLOPS. GEMM stage alone ≈ 2.1x
#                    faster in fp8 (657 vs ~315 TFLOPS); end-to-end 1.67x
#                    (2.53 s vs 4.23 s) — the difference is the exposed
#                    bandwidth-bound top-k read, which fp8 cannot shrink
#   vs exact ...... (--exact) full fp32 dequantized reference (sgemm + same
#                    top-k kernel, ~17 s): the best config's top-20 locations
#                    agree on 99.69% of entries; 93.9% of rows perfect, 100%
#                    of rows ≥18/20 (misses are within-ulp near-tie swaps at
#                    the k-boundary, never gross); values within 1 fp16 ulp
#                    (max rank-matched |Δ| = 0.125). For comparison: vs the
#                    fp16-cuBLAS baseline only 75.5% agree — that gap is the
#                    baseline's own fp16 accumulation noise, and with cout=
#                    :f8 only 11.8% (e4m3 quantization)
#   agreement ..... fp8 vs fp16 top-k: max |value diff| 1.5 (fp16
#                    accumulation noise of the baseline), ~25% of near-tie
#                    locations differ (values there agree within noise)
#   w sweep ....... lt :f16 end-to-end: 2^11: 3.9 s · 2^12: 3.1 s · 2^13:
#                    2.7 s · 2^14: 2.5 s · 2^15: 2.5 s (plateau; 2^15 also
#                    OOMs the :f32 variant at 2×16 GiB buffers → w=2^14 is
#                    the sweet spot)
#   normalized .... (--normalize; unit rows of A / unit cols of B, so C is a
#                    cosine-similarity matrix, |C| ≤ ~0.12): end-to-end
#                    unchanged within run variance — lt :f16 2.59 s → 435
#                    TFLOPS (w=2^13), 2.45 s → 459 TFLOPS (w=2^14); fp16
#                    baseline 4.23 s → speedup 1.63x (1.73x at w=2^14);
#                    location agreement vs exact 99.71% (94.1% rows perfect,
#                    100% ≥18/20) — fp has no absolute scale, so e4m3/fp16
#                    rounding and the top-k near-tie statistics are
#                    scale-invariant; even the fp8-out location churn is
#                    identical (11.6% vs 11.8% unnormalized)
#   fp8-out ....... cout=:f8 experiment (mma engine only — cuBLASLt rejects
#                    e4m3 D output, AlgoGetHeuristic status 15): the top-k
#                    scan gets SLOWER, 4.50 ms/chunk (222 GiB/s) vs 3.19 ms
#                    (627 GiB/s) for fp16 chunks, despite moving half the
#                    bytes — the scan is element-issue-bound (≈2.4 vs
#                    3.2 Gelem/ms), so narrower words buy nothing and the
#                    ~10-op e4m3 decode costs more than a fp16→fp32 cvt;
#                    GEMM-side the u8 store is free (21.3 vs 21.1 ms/chunk,
#                    same as fp16 store). Accuracy: top-k values off by 1
#                    e4m3 ulp (max |Δ| 16 at |C|≈250) and only 11.8% of
#                    top-20 locations survive vs f16-out — within-ulp
#                    near-ties shuffle freely. Answer to "does fp8 C speed
#                    up top-k": no — keep C chunks in fp16.
#
#   NB the run picks its own winner: engines are benchmarked in-process and
#   the fastest engine×cout combination is used for the full runs (gemmfp8cpp
#   jl's warning about cross-run comparisons applies — quote ratios, not
#   absolute TFLOPS across processes).
#
# MEMORY (measured on the 31.4 GiB RTX 5090; CUDA.free_memory before/during
#         the best-config run, matching the analytic footprint exactly)
#   best config (lt :f16, w=2^14): 12.3 GiB of arrays, ~12.9 GiB device total
#   incl. ~0.5 GiB CUDA context/JIT overhead — 18.5 GiB still free.
#   Breakdown: 2× C chunk buffers 8 GiB (64% — the dominant term, 2·M·w·2B),
#   B 4 GiB, A 256 MiB, lt workspace 32 MiB, D_val+D_loc 20 MiB.
#   General formula: M·K + K·N + 2·M·w·bytes(cout) + 2·M·k·4 + 32 MiB
#   (+ ~0.5 GiB context). Config table (GiB of arrays):
#     w=2^13 :f16 8.3 · w=2^14 :f16 12.3 · w=2^13 :f32 12.3 ·
#     w=2^14 :f32 20.3 · w=2^15 :f32 OOM (2×16 GiB buffers don't fit)
#   i.e. w is the only knob that matters; the sweep's TFLOPS plateau at
#   w=2^14 costs 12.3 GiB. Transients: data generation holds small fp32
#   chunk temps (1 GiB for --normalize's full-A pass, before the pipeline
#   runs); the engine×cout sweep frees each buffer set before the next, so
#   the peak is one buffer set + data.
#
# USAGE
#   julia --project=. test/gemmtopkfp8.jl                # full run + sweeps
#   julia --project=. test/gemmtopkfp8.jl --quick        # small smoke test
#   julia --project=. test/gemmtopkfp8.jl --engine=mma   # mma | lt | both
#   julia --project=. test/gemmtopkfp8.jl --baseline16   # in-process fp16 base
#   julia --project=. test/gemmtopkfp8.jl --exact        # + exact fp32 reference
#   julia --project=. test/gemmtopkfp8.jl --normalize    # unit rows of A / unit cols of B
#   julia --project=. test/gemmtopkfp8.jl --no-sweep     # skip the w sweep

using CUDA
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Statistics
using Libdl
using Printf
using Random

CUDA.allowscalar(false)

const F8 = Float8_E4M3FN

const CUDA_DIR = "/usr/local/cuda-13.3"   # nvcc that knows sm_120

# nvcc 13.3 emits PTX ISA 9.3 which the pinned 13.2 runtime cannot JIT, and
# older system nvcc lacks fp8 mma; compile with 13.3 and stamp the .version
# header down to 9.0 (codegen-compatible), as in gemmfp8cpp.jl.
const PTX_VERSION = "9.0"

# --------------------------------------------------------------------------
# C++ kernels: fp8 GEMM (m16n8k32 e4m3 tensor-core mma, fp32 accumulate,
# fp32-or-fp16 output) and elementwise e4m3 <-> fp32 converters.
# --------------------------------------------------------------------------
const CU_SRC = raw"""
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BM 64
#define BN 64
#define BK 32

// D = A * B, all Julia (column-major) layout, fp8 (e4m3 bit patterns) inputs,
// Dout (float or __half) output, fp32 accumulation.
// Block tile 64x64, K-step 32; 4 warps, each warp computes a 16x64 strip
// with 8 mma.sync.aligned.m16n8k32.e4m3 tensor-core ops per K step.
// `ldb` is the B column stride: B is consumed in column chunks of a larger
// K x N matrix, so the caller passes the chunk's base pointer and the
// parent's leading dimension. D's column stride is M.
template <typename Dout>
__device__ void fp8_gemm_body(const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ B,
                              Dout* __restrict__ D,
                              int M, int N, int K, int ldb)
{
    const int block_row = blockIdx.y;        // M / BM
    const int block_col = blockIdx.x;        // N / BN
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    __shared__ uint8_t As[BM][BK];           // A tile [m][k]
    __shared__ uint8_t BsT[BN][BK];          // B tile [n][k]

    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[nt][r] = 0.f;

    const int a0 = block_row * BM;
    const int n0 = block_col * BN;
    const int gr = warp << 4;            // warp's row offset in the tile
    const int gid = lane >> 2;           // a-fragment row / b-fragment col
    const int tig = lane & 3;            // k-group index

    for (int kt = 0; kt < K; kt += BK) {
        const int t = threadIdx.x;
        // load A tile: col-major A, m contiguous; read 16 m-bytes per thread,
        // scatter into the m-major shared tile
        {
            const int m = (t & 3) << 4, k = t >> 2;
            const uint4 v = *(const uint4*)&A[(size_t)(kt + k) * M + a0 + m];
            const uint8_t* p = (const uint8_t*)&v;
#pragma unroll
            for (int i = 0; i < 16; i++) As[m + i][k] = p[i];
        }
        // load B tile: col-major B, k contiguous; read 16 k-bytes per thread
        {
            const int k = (t & 1) << 4, n = t >> 1;
            *(uint4*)&BsT[n][k] = *(const uint4*)&B[(size_t)(n0 + n) * ldb + kt + k];
        }
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            uint32_t a[4] = {
                *(const uint32_t*)&As[gr + gid][tig * 4],
                *(const uint32_t*)&As[gr + gid + 8][tig * 4],
                *(const uint32_t*)&As[gr + gid][tig * 4 + 16],
                *(const uint32_t*)&As[gr + gid + 8][tig * 4 + 16]};
            uint32_t b[2] = {
                *(const uint32_t*)&BsT[nt * 8 + gid][tig * 4],
                *(const uint32_t*)&BsT[nt * 8 + gid][tig * 4 + 16]};
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        __syncthreads();
    }

    // store accumulators (m16n8 f32 fragment layout), D column-major.
    // fp16 stores stay scalar (2-byte): a packed __half2 store would need
    // 4-byte alignment, and odd fragment rows are only 2-byte aligned.
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int row = a0 + gr + gid;
        const int col = n0 + nt * 8 + tig * 2;
        Dout* d = &D[(size_t)col * M + row];
        if constexpr (sizeof(Dout) == 4) {
            d[0] = acc[nt][0]; d[M] = acc[nt][1];      // (row, col), (row, col+1)
            d[8] = acc[nt][2]; d[M + 8] = acc[nt][3];  // (row+8, col), (row+8, col+1)
        } else if constexpr (sizeof(Dout) == 2) {
            d[0] = __float2half(acc[nt][0]); d[M] = __float2half(acc[nt][1]);
            d[8] = __float2half(acc[nt][2]); d[M + 8] = __float2half(acc[nt][3]);
        } else {
            // e4m3 (satfinite): ~6% relative quantization of C, values beyond
            // ±448 clamp — enough to rank the tail, not to read values off
            d[0] = __nv_cvt_float_to_fp8(acc[nt][0], __NV_SATFINITE, __NV_E4M3);
            d[M] = __nv_cvt_float_to_fp8(acc[nt][1], __NV_SATFINITE, __NV_E4M3);
            d[8] = __nv_cvt_float_to_fp8(acc[nt][2], __NV_SATFINITE, __NV_E4M3);
            d[M + 8] = __nv_cvt_float_to_fp8(acc[nt][3], __NV_SATFINITE, __NV_E4M3);
        }
    }
}

extern "C" __global__ void fp8_gemm_f32(const uint8_t* A, const uint8_t* B,
                                        float* D, int M, int N, int K, int ldb)
{ fp8_gemm_body(A, B, D, M, N, K, ldb); }

extern "C" __global__ void fp8_gemm_f16(const uint8_t* A, const uint8_t* B,
                                        __half* D, int M, int N, int K, int ldb)
{ fp8_gemm_body(A, B, D, M, N, K, ldb); }

extern "C" __global__ void fp8_gemm_u8(const uint8_t* A, const uint8_t* B,
                                       uint8_t* D, int M, int N, int K, int ldb)
{ fp8_gemm_body(A, B, D, M, N, K, ldb); }

// elementwise fp32 -> e4m3 (satfinite) and e4m3 -> fp32 over contiguous
// ranges of n elements (test-data generation and dequantized references)
extern "C" __global__ void f32_to_f8_kernel(const float* __restrict__ src,
                                            uint8_t* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (uint8_t)__nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void f8_to_f32_kernel(const uint8_t* __restrict__ src,
                                            float* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const __half_raw h = __nv_cvt_fp8_to_halfraw(src[i], __NV_E4M3);
        dst[i] = __half2float(*reinterpret_cast<const __half*>(&h));
    }
}
"""

const k_gemm = Dict{Symbol, CUDA.CuFunction}()   # :f32 / :f16 (GEMM by cout)
const k_conv = Dict{Symbol, CUDA.CuFunction}()   # :f32_f8 / :f8_f32

function build_kernel()
    dir = joinpath(tempdir(), "gemmtopkfp8")
    mkpath(dir)
    cu = joinpath(dir, "fp8_gemm_topk.cu")
    ptxfile = joinpath(dir, "fp8_gemm_topk.ptx")
    write(cu, CU_SRC)
    nvcc = joinpath(CUDA_DIR, "bin", "nvcc")
    isfile(nvcc) || error("nvcc not found at $nvcc — adjust CUDA_DIR")
    # -ptx (not cubin) so the pinned 13.2 runtime JITs it for sm_120
    run(`$nvcc -O3 -std=c++17 -arch=compute_120 -ptx $cu -o $ptxfile`)
    ptx = read(ptxfile, String)
    ptx = replace(ptx, r"^\.version .*$"m => ".version " * PTX_VERSION)
    write(ptxfile, ptx)
    lib = CUDA.CuModule(read(ptxfile, String))
    k_gemm[:f32] = CUDA.CuFunction(lib, "fp8_gemm_f32")
    k_gemm[:f16] = CUDA.CuFunction(lib, "fp8_gemm_f16")
    k_gemm[:f8] = CUDA.CuFunction(lib, "fp8_gemm_u8")
    k_conv[:f32_f8] = CUDA.CuFunction(lib, "f32_to_f8_kernel")
    k_conv[:f8_f32] = CUDA.CuFunction(lib, "f8_to_f32_kernel")
    return nothing
end

# byte address of a GPU array's base plus an element offset, as CuPtr{UInt8}.
# Used instead of pointer(::SubArray) so every C++ launch takes a raw parent
# pointer + explicit element offset (no view-pointer semantics to trust).
@inline byteptr(p::CuPtr{T}, off::Integer = 0) where {T} =
    reinterpret(CuPtr{UInt8}, reinterpret(UInt64, p) + UInt(off) * sizeof(T))

# --------------------------------------------------------------------------
# GEMM engines. Both take (Cbuf M×wi, A M×K, Bp chunk base pointer) and
# launch on the current stream.
# --------------------------------------------------------------------------

# C++ mma kernel: D (M × wi, Tc) = A (M × K, e4m3) * B chunk (K × wi, e4m3,
# base Bp, column stride ldb). Requires M % 64 == 0, wi % 64 == 0, K % 32 == 0
# and a 16-byte aligned chunk base (uint4 loads).
function fp8_gemm_mma!(cout::Symbol, Cv::CuMatrix{<:Union{Float32, Float16, UInt8}},
                       A::CuMatrix{F8}, Bp::CuPtr{UInt8}, wi::Integer, ldb::Integer)
    M, K = size(A)
    @assert reinterpret(UInt64, Bp) % 16 == 0 "B chunk base must be 16-byte aligned"
    dt = eltype(Cv)
    sig = dt === Float16 ?
          (CuPtr{UInt8}, CuPtr{UInt8}, CuPtr{Float16}, Int32, Int32, Int32, Int32) :
          dt === Float32 ?
          (CuPtr{UInt8}, CuPtr{UInt8}, CuPtr{Float32}, Int32, Int32, Int32, Int32) :
          (CuPtr{UInt8}, CuPtr{UInt8}, CuPtr{UInt8}, Int32, Int32, Int32, Int32)
    CUDA.cudacall(k_gemm[cout], sig, byteptr(pointer(A)), Bp, pointer(Cv),
                  Int32(M), Int32(wi), Int32(K), Int32(ldb);
                  blocks = (cld(wi, 64), cld(M, 64)), threads = 128)
    return Cv
end

# --- cuBLASLt fp8 engine (ported from gemmfp8cpp.jl) --------------------------
# D = A * B via cublasLtMatmul, ccalled directly into the toolkit's
# libcublasLt. fp8 requires op(A) = A^T / op(B) = B: our col-major M x K A
# buffer is exactly a row-major K x M matrix (ORDER_ROW); the B chunk is
# col-major K x wi with ld = K over its base pointer; D col-major M x wi.
# Layouts + heuristic algo are cached per (M, wi, K, Tv). Needs the 595
# driver userland on LD_LIBRARY_PATH (see gemmfp8cpp.jl's header).

const LT_R_32F       = Cint(0)    # cudaDataType CUDA_R_32F
const LT_R_16F       = Cint(2)    # cudaDataType CUDA_R_16F
const LT_R_8F_E4M3   = Cint(28)   # cudaDataType CUDA_R_8F_E4M3 (NOT 14 = CUDA_R_16BF!)
const LT_OP_T        = Cint(1)
const LT_OP_N        = Cint(0)
const LT_COMPUTE_32F = Cint(68)   # CUBLAS_COMPUTE_32F
const LT_DESC_TRANSA = Cint(3)
const LT_DESC_TRANSB = Cint(4)
const LT_PREF_MAX_WS = Cint(1)    # CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
const LT_LAYOUT_ORDER = Cint(1)   # CUBLASLT_MATRIX_LAYOUT_ORDER
const LT_ORDER_ROW    = Cint(1)   # CUBLASLT_ORDER_ROW
const LT_ALGO_SZ     = 64         # sizeof(cublasLtMatmulAlgo_t)
const LT_HEUR_SZ     = 96         # sizeof(cublasLtMatmulHeuristicResult_t)

const lt_lib       = Ref{Ptr{Cvoid}}(C_NULL)
const lt_handle    = Ref{Ptr{Cvoid}}(C_NULL)
const lt_opdesc    = Ref{Ptr{Cvoid}}(C_NULL)
const lt_pref      = Ref{Ptr{Cvoid}}(C_NULL)
const p_ltMatmul   = Ref{Ptr{Cvoid}}(C_NULL)
const lt_workspace = Ref{CuVector{UInt8}}()
const lt_cache = Dict{Tuple{Int,Int,Int,Type}, Tuple{Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Vector{UInt8}}}()

checklt(s::Integer, what::AbstractString) = s == 0 || error("cublasLt: $what failed (status $s)")

function cublaslt_init()
    lib = ""
    for dir in ("/usr/local/cuda-13.4", "/usr/local/cuda-13.3")
        p = joinpath(dir, "lib64", "libcublasLt.so.13")
        if isfile(p); lib = p; break; end
    end
    isempty(lib) && error("libcublasLt.so.13 not found in /usr/local/cuda-13.{4,3}")
    lt_lib[] = Libdl.dlopen(lib)
    sym(s) = Libdl.dlsym(lt_lib[], s)

    h = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtCreate), Cint, (Ptr{Ptr{Cvoid}},), h), "cublasLtCreate")
    lt_handle[] = h[]

    op = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtMatmulDescCreate), Cint, (Ptr{Ptr{Cvoid}}, Cint, Cint),
                  op, LT_COMPUTE_32F, LT_R_32F), "MatmulDescCreate")
    ta, tb = LT_OP_T, LT_OP_N
    checklt(ccall(sym(:cublasLtMatmulDescSetAttribute), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                  op[], LT_DESC_TRANSA, Ref(ta), sizeof(Cint)), "Set TRANSA")
    checklt(ccall(sym(:cublasLtMatmulDescSetAttribute), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                  op[], LT_DESC_TRANSB, Ref(tb), sizeof(Cint)), "Set TRANSB")
    lt_opdesc[] = op[]

    pref = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtMatmulPreferenceCreate), Cint, (Ptr{Ptr{Cvoid}},), pref),
            "PreferenceCreate")
    wsz = Csize_t(32 << 20)
    checklt(ccall(sym(:cublasLtMatmulPreferenceSetAttribute), Cint,
                  (Ptr{Cvoid}, Cint, Ptr{Csize_t}, Csize_t),
                  pref[], LT_PREF_MAX_WS, Ref(wsz), sizeof(Csize_t)), "Pref MAX_WS")
    lt_pref[] = pref[]

    p_ltMatmul[] = sym(:cublasLtMatmul)
    lt_workspace[] = CUDA.zeros(UInt8, 32 << 20)
    return nothing
end

function lt_entries(M::Int, N::Int, K::Int, ::Type{Tv}) where {Tv}
    get!(lt_cache, (M, N, K, Tv)) do
        sym(s) = Libdl.dlsym(lt_lib[], s)
        layout(dt, rows, cols, ld) = begin
            l = Ref{Ptr{Cvoid}}(C_NULL)
            checklt(ccall(sym(:cublasLtMatrixLayoutCreate), Cint,
                          (Ptr{Ptr{Cvoid}}, Cint, Culonglong, Culonglong, Clonglong),
                          l, dt, rows, cols, ld), "LayoutCreate")
            l[]
        end
        d_dt = Tv === Float32 ? LT_R_32F : Tv === Float16 ? LT_R_16F : LT_R_8F_E4M3
        # A: our col-major M x K buffer seen as row-major K x M (op(A) = A^T).
        # (The canonical-looking ORDER_COL instead runs without error but
        # contracts the wrong axis; only randomized checks catch it.)
        la = layout(LT_R_8F_E4M3, K, M, M)
        order = Ref(LT_ORDER_ROW)
        checklt(ccall(sym(:cublasLtMatrixLayoutSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                      la, LT_LAYOUT_ORDER, order, sizeof(Cint)), "Layout ORDER")
        lb = layout(LT_R_8F_E4M3, K, N, K)                    # B chunk, ld = parent K
        ld = layout(d_dt, M, N, M)
        buf = Vector{UInt8}(undef, LT_HEUR_SZ * 4)
        nres = Ref{Cint}(0)
        checklt(ccall(sym(:cublasLtMatmulAlgoGetHeuristic), Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{UInt8}, Ptr{Cint}),
                      lt_handle[], lt_opdesc[], la, lb, ld, ld, lt_pref[], 4, buf, nres),
                "AlgoGetHeuristic")
        nres[] >= 1 || error("cublasLt: no fp8 algorithm for ($M, $N, $K, $Tv)")
        (la, lb, ld, copy(buf[1:LT_ALGO_SZ]))
    end
end

function fp8_gemm_lt!(Cv::CuMatrix{Tv}, A::CuMatrix{F8}, Bp::CuPtr{UInt8}, wi::Integer) where {Tv}
    lt_handle[] == C_NULL && cublaslt_init()
    M, K = size(A)
    @assert M % 16 == 0 && K % 16 == 0 "cuBLASLt fp8 needs 16-byte-multiple leading dims"
    la, lb, ld, algo = lt_entries(M, wi, K, Tv)
    alpha, beta = Ref(1.0f0), Ref(0.0f0)
    # ccall's type slot must be a literal tuple, so branch on Tv here; the
    # tuples differ only in the C/D pointer types (slot 9 and 11)
    if Tv === Float32
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{Float32}, Ptr{Cvoid},
                       CuPtr{Float32}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    elseif Tv === Float16
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{Float16}, Ptr{Cvoid},
                       CuPtr{Float16}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    else
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    end
    return Cv
end

# --------------------------------------------------------------------------
# Kernel: fused per-row top-k scan + running merge (generalized from
# gemmtopk.jl to read Tc chunks into a Tv reservoir).
#
# One thread per row of the C chunk. Scans the row into a register-resident
# unsorted top-K reservoir (threshold-filtered, minimum-replace), sorts it
# once (descending), then two-pointer-merges it into the running top-K in
# D_val/D_loc. Fresh indices are chunk-local (1..width) and are shifted by
# `offset` into global column ids. The scan is unrolled TOPK_UNROLLx with the
# next batch of loads issued before the current batch is processed.
# IMPORTANT: all sentinels (out-of-range loads, reservoir init, threshold)
# are typemin(Tv) — NOT typemin(Tc) converted, which for Tc=Float16/Tv=Float32
# (-65504 vs -3.4e38) would wrongly enter the reservoir.
# --------------------------------------------------------------------------
const TOPK_UNROLL = 4

# scan value conversion: C chunks are read as Tc; fp8-out chunks are raw
# e4m3 bytes (CuMatrix{UInt8}) decoded on the fly (exact, ~10 integer ops).
@inline _scanval(::Type{Float16}, x::Float16) = Float32(x)
@inline _scanval(::Type{Float32}, x::Float32) = x
@inline _scanval(::Type{UInt8}, x::UInt8) = Float32(reinterpret(F8, x))

@inline function _tup_argmin(t::NTuple{K, T}) where {K, T}
    mv = t[1]
    mp = 1
    for j in 2:K                     # static bounds → fully unrolled selects
        vj = t[j]
        less = vj < mv
        mv = less ? vj : mv
        mp = less ? j : mp
    end
    (mv, mp)
end

@inline _tup_replace(t::NTuple{K, T}, pos::Int, v) where {K, T} =
    ntuple(j -> j == pos ? v : t[j], Val(K))

@inline function _tup_insert!(lk, li, v, idx::Int32, min_pos)
    lk = _tup_replace(lk, min_pos, v)
    li = _tup_replace(li, min_pos, idx)
    mv, mp = _tup_argmin(lk)
    return (lk, li, mv, mp)
end

function rowtopk_merge_kernel!(out_vals::CuDeviceMatrix{Tv}, out_inds::CuDeviceMatrix{Int32},
                               C::CuDeviceMatrix{Tc}, offset::Int32, row_offset::Int32,
                               ::Val{K}) where {Tc, Tv, K}
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M = size(C, 1)
    row <= M || return nothing
    drow = row + row_offset          # this row's slot in the running top-k
    U = TOPK_UNROLL

    # register-resident unsorted top-K reservoir + tracked minimum
    lk  = ntuple(j -> typemin(Tv), Val(K))
    li  = ntuple(j -> Int32(0),   Val(K))
    thresh  = typemin(Tv)            # current reservoir minimum (register)
    min_pos = 1                      # argmin(lk)

    width = size(C, 2)
    c = 1
    # rolling window of U in-flight loads; nᵢ holds column c+i-1 (as Tv)
    @inbounds begin
        n1 = 1 <= width ? _scanval(Tc, C[row]) : typemin(Tv)
        n2 = 2 <= width ? _scanval(Tc, C[row + M]) : typemin(Tv)
        n3 = 3 <= width ? _scanval(Tc, C[row + 2 * M]) : typemin(Tv)
        n4 = 4 <= width ? _scanval(Tc, C[row + 3 * M]) : typemin(Tv)
    end
    @inbounds while c <= width
        # issue the whole next batch of U loads before processing (latency hiding)
        m1 = c + U     <= width ? _scanval(Tc, C[row + (c + U - 1) * M]) : typemin(Tv)
        m2 = c + U + 1 <= width ? _scanval(Tc, C[row + (c + U) * M]) : typemin(Tv)
        m3 = c + U + 2 <= width ? _scanval(Tc, C[row + (c + U + 1) * M]) : typemin(Tv)
        m4 = c + U + 3 <= width ? _scanval(Tc, C[row + (c + U + 2) * M]) : typemin(Tv)
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

    # merge into the running top-K. Reads only from registers, writes straight
    # to global — no aliasing.
    rv = ntuple(j -> out_vals[drow, j], Val(K))
    ri = ntuple(j -> out_inds[drow, j], Val(K))
    pa = 1; pb = 1
    @inbounds for j in 1:K
        va = pa <= K ? rv[pa] : typemin(Tv)
        vb = pb <= K ? lk[pb] : typemin(Tv)
        if va >= vb
            out_vals[drow, j] = va
            out_inds[drow, j] = ri[pa]
            pa += 1
        else
            out_vals[drow, j] = vb
            idx = li[pb]
            out_inds[drow, j] = idx == Int32(0) ? Int32(0) : idx + offset
            pb += 1
        end
    end
    return nothing
end

function launch_rowtopk_merge!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                               C::CuMatrix, offset::Integer, k::Integer;
                               threads::Integer = 128, row_offset::Integer = 0)
    M = size(C, 1)
    @assert size(C, 2) >= 1 "chunk must be non-empty"
    @cuda threads = threads blocks = cld(M, threads) rowtopk_merge_kernel!(
        D_val, D_loc, C, Int32(offset), Int32(row_offset), Val(Int(k)))
    return nothing
end

# --------------------------------------------------------------------------
# Double-buffered batched GEMM + top-k pipelines.
#
# `w` is the chunk width (columns of B per chunk); buffer usage is 2·M·w
# elements of the C element type. `overlap=false` runs everything in order on
# the default stream (benchmark baseline; results are identical — the kernels
# are deterministic).
# --------------------------------------------------------------------------
function gemm_topk_fp8!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                        A::CuMatrix{F8}, B::CuMatrix{F8};
                        k::Integer = 20, w::Integer = 2^13, overlap::Bool = true,
                        threads::Integer = 128, engine::Symbol = :mma,
                        cout::Symbol = :f16, max_chunks::Integer = typemax(Int))
    M, KA = size(A)
    KB, N = size(B)
    @assert KA == KB "inner dimensions of A and B must match"
    @assert size(D_val) == (M, k) && size(D_loc) == (M, k)
    @assert 1 <= k <= w <= N
    @assert cout in (:f16, :f32, :f8) && engine in (:mma, :lt)
    Tc = cout === :f16 ? Float16 : cout === :f32 ? Float32 : UInt8
    if engine === :mma
        @assert M % 64 == 0 && w % 64 == 0 && KA % 32 == 0 "mma kernel needs M%64==0, w%64==0, K%32==0"
        @assert N % w == 0 "engine :mma needs w to divide N (no ragged final chunk)"
    else
        @assert M % 16 == 0 && KA % 16 == 0 "cuBLASLt fp8 needs 16-byte-multiple leading dims"
    end

    # reset the running top-k; the kernel merges into it chunk by chunk
    fill!(D_val, typemin(Float32))
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    w = Int(w)
    nb = min(cld(N, w), Int(max_chunks))
    bufs = [CuMatrix{Tc}(undef, M, w), CuMatrix{Tc}(undef, M, w)]

    gemmf! = engine === :mma ?
             (Cbuf, Bp, wi) -> fp8_gemm_mma!(cout, Cbuf, A, Bp, wi, KA) :
             (Cbuf, Bp, wi) -> fp8_gemm_lt!(Cbuf, A, Bp, wi)

    sg = overlap ? CuStream(; flags = CUDA.STREAM_NON_BLOCKING) : CUDA.default_stream()
    st = overlap ? CuStream(; flags = CUDA.STREAM_NON_BLOCKING) : sg
    evg = (CuEvent(), CuEvent())   # gemm of the chunk using buffer b is done
    evt = (CuEvent(), CuEvent())   # top-k of the chunk using buffer b is done

    for i in 1:nb
        b = mod1(i, 2)
        lo = (i - 1) * w + 1
        wi = min(N, i * w) - lo + 1
        Cbuf = bufs[b]
        # base pointer of B column lo; the column stride stays the parent KB
        Bp = byteptr(pointer(B), (lo - 1) * KB)

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(evt[b])       # buffer b's previous top-k finished
            gemmf!(Cbuf, Bp, wi)             # launches on the current stream (= sg)
            CUDA.record(evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(evg[b])                # C chunk is ready
            launch_rowtopk_merge!(D_val, D_loc, wi == w ? Cbuf : @view(Cbuf[:, 1:wi]),
                                  lo - 1, k; threads)
            CUDA.record(evt[b])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

# fp16 baseline (cuBLAS mul! GEMM), same pipeline shape as gemmtopk.jl but
# with a Float32 D_val — used by --baseline16 for the in-process speedup.
function gemm_topk_fp16!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                         A::CuMatrix{Float16}, B::CuMatrix{Float16};
                         k::Integer = 20, w::Integer = 2^13, overlap::Bool = true,
                         threads::Integer = 128, max_chunks::Integer = typemax(Int))
    M, KA = size(A)
    KB, N = size(B)
    @assert KA == KB && size(D_val) == (M, k) && size(D_loc) == (M, k) && 1 <= k <= w <= N
    fill!(D_val, typemin(Float32))
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()
    w = Int(w)
    nb = min(cld(N, w), Int(max_chunks))
    bufs = [CuMatrix{Float16}(undef, M, w), CuMatrix{Float16}(undef, M, w)]
    sg = overlap ? CuStream(; flags = CUDA.STREAM_NON_BLOCKING) : CUDA.default_stream()
    st = overlap ? CuStream(; flags = CUDA.STREAM_NON_BLOCKING) : sg
    evg = (CuEvent(), CuEvent()); evt = (CuEvent(), CuEvent())
    for i in 1:nb
        b = mod1(i, 2)
        lo = (i - 1) * w + 1
        wi = min(N, i * w) - lo + 1
        Bv = @view B[:, lo:lo+wi-1]
        Cv = @view bufs[b][:, 1:wi]
        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(evt[b])
            mul!(Cv, A, Bv)                  # cuBLAS fp16 GEMM (tensor cores)
            CUDA.record(evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(evg[b])
            launch_rowtopk_merge!(D_val, D_loc, Cv, lo - 1, k; threads)
            CUDA.record(evt[b])
        end
    end
    CUDA.device_synchronize()
    return D_val, D_loc
end

# --------------------------------------------------------------------------
# Data generation / conversion (chunked, all on GPU)
# --------------------------------------------------------------------------

# fills an fp8 matrix with standard normals (fp32 randn → e4m3 satfinite,
# converted by the CUDA kernel; DLFP8Types' host conversions are not assumed
# to compile into GPU broadcast kernels)
function randn_fp8!(X::CuMatrix{F8}; chunk::Integer = 2^12)
    M, N = size(X)
    tmp = CuMatrix{Float32}(undef, M, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(tmp)
        f32_to_f8!(byteptr(pointer(X), (lo - 1) * M), pointer(tmp), M * wi)
    end
    CUDA.unsafe_free!(tmp)
    return X
end

# normalized variants: rows of A / columns of B are scaled to unit L2 norm in
# fp32 BEFORE e4m3 quantization, so C = A·B holds cosine similarities
# (|C| ≤ 1) — the production-like regime. Column chunks are self-contained
# for B; row normalization of A needs the full fp32 matrix (1 GiB).
function randn_fp8_normrows!(A::CuMatrix{F8})
    M, K = size(A)
    T = CuMatrix{Float32}(undef, M, K)
    randn!(T)
    T ./= sqrt.(sum(abs2, T; dims = 2))
    f32_to_f8!(byteptr(pointer(A)), pointer(T), M * K)
    CUDA.unsafe_free!(T)
    return A
end

function randn_fp8_normcols!(B::CuMatrix{F8}; chunk::Integer = 2^12)
    K, N = size(B)
    T = CuMatrix{Float32}(undef, K, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(T)
        T ./= sqrt.(sum(abs2, T; dims = 1))   # per-column norms
        f32_to_f8!(byteptr(pointer(B), (lo - 1) * K), pointer(T), K * wi)
    end
    CUDA.unsafe_free!(T)
    return B
end

# chunked e4m3 → fp16 conversion (via exact fp32), for the fp16 baseline
function fp16_from_fp8!(X16::CuMatrix{Float16}, X8::CuMatrix{F8}; chunk::Integer = 2^12)
    M, N = size(X8)
    tmp = CuMatrix{Float32}(undef, M, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        f8_to_f32!(pointer(tmp), byteptr(pointer(X8), (lo - 1) * M), M * wi)
        broadcast!(Float16, @view(X16[:, lo:lo+wi-1]), @view(tmp[:, 1:wi]))
    end
    CUDA.unsafe_free!(tmp)
    return X16
end

# (dst/src pointers arrive via byteptr, so both are CuPtr{UInt8} — only the
# element offset encodes the type; the sig tuple fixes the real pointer types)
function f32_to_f8!(dst::CuPtr, src::CuPtr, n::Integer)
    CUDA.cudacall(k_conv[:f32_f8], (CuPtr{Float32}, CuPtr{UInt8}, Int32),
                  src, dst, Int32(n); blocks = cld(n, 256), threads = 256)
end

function f8_to_f32!(dst::CuPtr, src::CuPtr, n::Integer)
    CUDA.cudacall(k_conv[:f8_f32], (CuPtr{UInt8}, CuPtr{Float32}, Int32),
                  src, dst, Int32(n); blocks = cld(n, 256), threads = 256)
end

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# exact multiset check against a CPU sort, for all C element types (fp16, fp32
# and raw e4m3 bytes), over several shapes including ragged chunk sizes (w ∤ N),
# a single row, and the exact full-matrix chunk size
function test_rowtopk_merge_kernel(; k::Integer = 20)
    CUDA.seed!(123)
    for Tc in (Float16, Float32, UInt8)
        for (M, N, w) in ((64, 100, 37), (257, 128, 128), (1000, 333, 111), (1, 4096, 512), (511, 2048, 2048))
            Cf = randn(Float32, M, N)
            C8 = Tc === UInt8 ? CuMatrix{F8}(Cf) : nothing
            C = Tc === UInt8 ? reinterpret(UInt8, C8) : CuMatrix{Tc}(Cf)
            Ch = Tc === UInt8 ? Float32.(Array(C8)) : Array(C)
            D_val = CUDA.fill(typemin(Float32), M, k)
            D_loc = CUDA.zeros(Int32, M, k)
            for i in 1:cld(N, w)
                lo = (i - 1) * w + 1
                wi = min(N, i * w) - lo + 1
                buf = CuMatrix{Tc}(@view C[:, lo:lo+wi-1])
                launch_rowtopk_merge!(D_val, D_loc, buf, lo - 1, k)
            end
            Dv = Array(D_val)
            Di = Array(D_loc)
            for r in 1:M
                ref = sort(Ch[r, :]; rev = true)[1:k]
                got = Dv[r, :]
                @assert issorted(got; rev = true) "row $r of ($M,$N,$w,$Tc): output not sorted"
                @assert sort(got; rev = true) == ref "row $r of ($M,$N,$w,$Tc): wrong top-$k values"
                @assert all(>(0), Di[r, :]) && Ch[r, Di[r, :]] == got "row $r of ($M,$N,$w,$Tc): wrong locations"
            end
        end
    end
    @info "unit tests passed: rowtopk_merge_kernel! matches CPU reference (Tc=Float16,Float32,e4m3 bytes)"
    return nothing
end

# GEMM engines vs a dequantized fp64 reference, plus the chunked-ldb path
# (B consumed in column chunks through the base-pointer/ldb interface) which
# must reproduce the whole-matrix GEMM bitwise.
function test_fp8_gemm(; engines = (:mma, :lt))
    Random.seed!(42)
    M, N, K = 256, 256, 256
    sa, sb = 0.05f0, 0.02f0
    A = CuArray(F8.(randn(Float32, M, K) .* sa))
    B = CuArray(F8.(randn(Float32, K, N) .* sb))
    ref = Float64.(Float32.(Array(A))) * Float64.(Float32.(Array(B)))

    # ref is the fp64 product of the STORED (scaled) values, so the GEMM
    # outputs are compared directly — no dequantization on either side
    # (gemmfp8cpp.jl instead divides the reference by sa·sb; equivalent).

    # mma fp32-out
    D = CUDA.zeros(Float32, M, N)
    fp8_gemm_mma!(:f32, D, A, byteptr(pointer(B)), N, K)
    rel = norm(Array(D) .- ref) / norm(ref)
    @assert rel < 1e-6 "mma :f32 GEMM wrong (relerr $rel)"
    @printf("  mma :f32 relerr %.3g\n", rel)

    # mma fp16-out (fp16 storage of the fp32-accumulated result)
    D16 = CuMatrix{Float16}(undef, M, N)
    fp8_gemm_mma!(:f16, D16, A, byteptr(pointer(B)), N, K)
    rel = norm(Array(D16) .- ref) / norm(ref)
    @assert rel < 5e-3 "mma :f16 GEMM wrong (relerr $rel)"
    @printf("  mma :f16 relerr %.3g\n", rel)

    # mma fp8-out: C itself quantized to e4m3 — ~6% relative rounding error,
    # satfinite clamp at ±448. relerr ≈ ulp/√12·√2/σ ≈ 3%; max abs err ≤ 1 ulp
    # of the largest |ref| (here |ref| < 0.125 → ulp = 2^-7, so < 0.008)
    D8 = CuMatrix{UInt8}(undef, M, N)
    fp8_gemm_mma!(:f8, D8, A, byteptr(pointer(B)), N, K)
    Q = Float32.(reinterpret(F8, Array(D8)))
    rel = norm(Q .- ref) / norm(ref)
    maxabs = maximum(abs.(Q .- ref))
    @assert rel < 0.06 "mma :f8 GEMM wrong (relerr $rel)"
    @assert maxabs < 0.008 "mma :f8 max abs err $maxabs exceeds 1 e4m3 ulp"
    @printf("  mma :f8  relerr %.3g  maxabs %.3g\n", rel, maxabs)

    # chunked-ldb path: same result, bitwise
    wc = 64
    Dc = CuMatrix{Float32}(undef, M, wc)
    D2 = CUDA.zeros(Float32, M, N)
    for lo in 1:wc:N
        fp8_gemm_mma!(:f32, Dc, A, byteptr(pointer(B), (lo - 1) * K), wc, K)
        copyto!(@view(D2[:, lo:lo+wc-1]), Dc)
    end
    @assert Array(D2) == Array(D) "chunked-ldb GEMM differs from whole-matrix GEMM"
    @info "unit tests passed: fp8_gemm_mma! (incl. chunked-ldb path)"

    if :lt in engines
        for (co, Dlt, tolf) in ((:f32, CUDA.zeros(Float32, M, N), 1e-6),
                                (:f16, CuMatrix{Float16}(undef, M, N), 5e-3),
                                (:f8, CuMatrix{UInt8}(undef, M, N), 0.06))
            try
                fp8_gemm_lt!(Dlt, A, byteptr(pointer(B)), N)
                vals = co === :f8 ? Float32.(reinterpret(F8, Array(Dlt))) : Array(Dlt)
                rel = norm(vals .- ref) / norm(ref)
                @assert rel < tolf "lt $co GEMM wrong (relerr $rel)"
                @printf("  lt  :%s relerr %.3g\n", string(co), rel)
            catch e
                @warn "cuBLASLt fp8 ($co) unavailable or failed — skipping its test" e
            end
        end
    end
    return nothing
end

# exact top-k ground truth: dequantize A and B to fp32 (e4m3→fp32 is exact),
# compute C = A·B in fp32 (sgemm, no TF32) block-wise, and merge each block
# into the running top-k with the same rowtopk_merge_kernel!. Runs the full
# 2^50-FLOP product in fp32 — ~30 s — so it is flag-gated (--exact).
function exact_topk!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                     A::CuMatrix{F8}, B::CuMatrix{F8}; k::Integer = 20,
                     wref::Integer = 2^15, rowblk::Integer = 2^13)
    M, K = size(A)
    N = size(B, 2)
    A32 = CuMatrix{Float32}(undef, M, K)
    f8_to_f32!(pointer(A32), byteptr(pointer(A)), M * K)
    Ab = CuMatrix{Float32}(undef, rowblk, K)      # row block, contiguous
    B32 = CuMatrix{Float32}(undef, K, wref)
    Cb = CuMatrix{Float32}(undef, rowblk, wref)
    fill!(D_val, typemin(Float32))
    fill!(D_loc, Int32(0))
    # row blocks: each launch merges into D rows (rb-1)+1 .. rb-1+rowblk
    # (row_offset), so Cb — always dense — is read via its own row indexing.
    @assert M % rowblk == 0 "rowblk must divide M"
    for lo in 1:wref:N
        wi = min(N, lo + wref - 1) - lo + 1
        f8_to_f32!(pointer(B32), byteptr(pointer(B), (lo - 1) * K), K * wi)
        Bv = @view B32[:, 1:wi]
        for rb in 1:rowblk:M
            broadcast!(identity, Ab, A32[rb:rb+rowblk-1, :])
            mul!(Cb, Ab, Bv)                      # fp32 gemm (no TF32 by default)
            launch_rowtopk_merge!(D_val, D_loc, Cb, lo - 1, k; row_offset = rb - 1)
        end
    end
    CUDA.unsafe_free!.((A32, Ab, B32, Cb))
    return D_val, D_loc
end

# end-to-end accuracy: recompute a few random rows of C = A*B exactly in fp32
# (dequantized operands, chunked gemv) and compare top-k. e4m3×e4m3 products
# are exact in fp32 and the mma accumulates in fp32, so `tol` only needs to
# cover the C storage rounding (~1 ulp of |C|) plus accumulation-order noise.
function spot_check(A::CuMatrix{F8}, B::CuMatrix{F8}, D_val::CuMatrix{Float32},
                    D_loc::CuMatrix{Int32}, k::Integer;
                    nrows::Integer = 8, seed::Integer = 42, tol::Real = 1.0)
    Random.seed!(seed)
    M, K = size(A)
    N = size(B, 2)
    wref = 2^15                          # fp32 reference chunks (fits easily)
    a32 = CuVector{Float32}(undef, K)
    B32 = CuMatrix{Float32}(undef, K, wref)
    ref = CuVector{Float32}(undef, N)

    maxvalerr = 0.0
    locbad = 0
    for _ in 1:nrows
        r = rand(1:M)
        copyto!(a32, Float32.(Array(@view A[r, :])))   # dequantize the row (small)
        fill!(ref, 0f0)
        for lo in 1:wref:N
            wi = min(N, lo + wref - 1) - lo + 1
            # dequantize the B chunk on GPU (e4m3 → fp32 is exact)
            f8_to_f32!(byteptr(pointer(B32)), byteptr(pointer(B), (lo - 1) * K), K * wi)
            # ref chunk = B32ᵀ * a32  (fp32 gemv)
            mul!(view(ref, lo:lo+wi-1), transpose(@view B32[:, 1:wi]), a32)
        end
        refh = Array(ref)
        refv = sort(refh; rev = true)[1:k]
        gotv = Array(@view D_val[r, :])
        goti = Array(@view D_loc[r, :])

        @assert all(1 .<= goti .<= N) "row $r: locations out of range"
        @assert issorted(gotv; rev = true) "row $r: values not sorted"
        maxvalerr = max(maxvalerr, maximum(abs.(refv .- gotv)))
        # a location is "wrong" only if the value stored there disagrees with
        # the claimed top-k value by more than the storage tolerance
        locbad += count(abs.(refh[goti] .- gotv) .> tol)
    end
    @assert maxvalerr < tol "top-$k values deviate too much from fp32 reference (max err $maxvalerr)"
    @assert locbad == 0 "$locbad locations are inconsistent with their values"
    return maxvalerr
end

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------

# per-chunk GEMM time for every engine×cout, and the top-k read cost per C
# element type; returns Dict((engine, cout) => seconds/chunk)
function bench_gemm(A::CuMatrix{F8}, B::CuMatrix{F8}, D_val::CuMatrix{Float32},
                    D_loc::CuMatrix{Int32}, k::Integer, w::Integer;
                    engines::Tuple = (:mma, :lt), iters::Integer = 20)
    M, K = size(A)
    flops = 2.0 * M * K * w
    Bp1 = byteptr(pointer(B))
    times = Dict{Tuple{Symbol, Symbol}, Float64}()
    @printf("%-6s %-5s | %10s %11s\n", "engine", "cout", "ms/chunk", "TFLOPS")
    for cout in (:f16, :f32, :f8)
        Tc = cout === :f16 ? Float16 : cout === :f32 ? Float32 : UInt8
        Cv = CuMatrix{Tc}(undef, M, w)
        for eng in engines
            gemmf! = eng === :mma ?
                     (Cbuf, Bp, wi) -> fp8_gemm_mma!(cout, Cbuf, A, Bp, wi, K) :
                     (Cbuf, Bp, wi) -> fp8_gemm_lt!(Cbuf, A, Bp, wi)
            try
                gemmf!(Cv, Bp1, w)                       # warmup
                CUDA.device_synchronize()
                t = CUDA.@elapsed CUDA.@sync for _ in 1:iters
                    gemmf!(Cv, Bp1, w)
                end
                times[(eng, cout)] = t / iters
                @printf("%-6s %-5s | %10.2f %11.1f\n", string(eng), string(cout),
                        t / iters * 1e3, flops / (t / iters) / 1e12)
            catch e
                @warn "fp8 GEMM engine ($eng, $cout) unavailable" e
            end
        end
        # top-k scan cost for this C element type
        launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k)
        CUDA.device_synchronize()
        tt = CUDA.@elapsed CUDA.@sync for _ in 1:iters
            launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k)
        end
        @printf("%-6s %-5s | %10.2f %11s   <- top-k only (%.0f GiB/s of C read)\n",
                "topk", string(cout), tt / iters * 1e3, "", M * w * sizeof(Tc) / (tt / iters) / 2^30)
        CUDA.unsafe_free!(Cv)
    end
    return times
end

function bench_topk_threads(Cv::CuMatrix, D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                            k::Integer; iters::Integer = 20)
    best = (128, Inf)
    for th in (64, 128, 256, 512)
        launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k; threads = th)
        CUDA.device_synchronize()
        t = CUDA.@elapsed CUDA.@sync for _ in 1:iters
            launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k; threads = th)
        end
        @printf("  top-k threads=%3d: %.3f ms/chunk\n", th, t / iters * 1e3)
        t < best[2] && (best = (th, t / iters))
    end
    return best[1]
end

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

function main(; M::Integer = 2^17, K::Integer = 2^11, N::Integer = 2^21, k::Integer = 20,
              w::Integer = 2^13, sweep::Bool = true, engines::Tuple = (:mma, :lt),
              baseline16::Bool = false, exact::Bool = false, normalize::Bool = false)
    dev = CUDA.device()
    @info @sprintf("GPU: %s, %.1f GiB", CUDA.name(dev), CUDA.totalmem(dev) / 2^30)
    @info @sprintf("A: %d×%d e4m3 (%.0f MiB),  B: %d×%d e4m3 (%.1f GiB),  C would be %.0f GiB (fp32),  k=%d",
                   M, K, M * K / 2^20, K, N, K * N / 2^30, M * N * 4 / 2^30, k)
    flops = 2.0 * M * K * N

    @info "compiling C++ mma kernel (nvcc → PTX → CuModule)..."
    build_kernel()

    test_rowtopk_merge_kernel(k = k)
    test_fp8_gemm(engines = engines)

    @info("generating fp8 data on GPU" * (normalize ? " (rows of A / cols of B normalized → cosine-similarity C)" : ""))
    A = CuMatrix{F8}(undef, M, K)
    B = CuMatrix{F8}(undef, K, N)
    if normalize
        randn_fp8_normrows!(A)
        randn_fp8_normcols!(B)
    else
        randn_fp8!(A)
        randn_fp8!(B)
    end

    D_val = CuMatrix{Float32}(undef, M, k)
    D_loc = CuMatrix{Int32}(undef, M, k)

    @info "warmup (2 chunks per engine×cout)..."
    for (eng, co) in ((:mma, :f16), (:mma, :f32), (:mma, :f8),
                      (:lt, :f16), (:lt, :f32), (:lt, :f8))
        eng in engines || continue
        try
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w, max_chunks = 2, engine = eng, cout = co)
        catch e
            @warn "warmup failed for ($eng, $co)" e
        end
    end

    @info "per-chunk micro-benchmark (w=$w)..."
    tg = bench_gemm(A, B, D_val, D_loc, k, w; engines)
    Cv16 = CuMatrix{Float16}(undef, M, w)
    best_threads = bench_topk_threads(Cv16, D_val, D_loc, k)
    @info @sprintf("top-k thread count: using %d", best_threads)
    CUDA.unsafe_free!(Cv16)

    @info "end-to-end runs (pipelined) per engine×cout..."
    best = ((:mma, :f16), Inf)
    for eng in engines, co in (:f16, :f32, :f8)
        GC.gc(); CUDA.reclaim()
        try
            CUDA.device_synchronize(); t0 = time()
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w, engine = eng, cout = co)
            CUDA.device_synchronize()
            t = time() - t0
            @printf("  %-4s %-4s: %6.2f s  →  %.1f TFLOPS\n", string(eng), string(co), t, flops / t / 1e12)
            t < best[2] && (best = ((eng, co), t))
        catch e
            @warn "full run failed for ($eng, $co)" e
        end
    end
    (engine, cout), t_pipe = best
    @info @sprintf("best config: engine=%s cout=%s (%.2f s, %.1f TFLOPS)",
                   engine, cout, t_pipe, flops / t_pipe / 1e12)
    gemm_floor = get(tg, (engine, cout), NaN) * (N / w)
    @printf("(GEMM-only floor ≈ %.2f s → top-k exposes ≈ %.2f s over the floor)\n",
            gemm_floor, max(0.0, t_pipe - gemm_floor))

    # re-run the best config to have its results in D_val/D_loc
    gemm_topk_fp8!(D_val, D_loc, A, B; k, w, engine = engine, cout = cout)

    @info "full run, sequential (no overlap)..."
    D_val2 = similar(D_val); D_loc2 = similar(D_loc)
    CUDA.device_synchronize(); t0 = time()
    gemm_topk_fp8!(D_val2, D_loc2, A, B; k, w, engine = engine, cout = cout, overlap = false)
    t_seq = time() - t0
    @printf("sequential: %6.2f s  →  %.1f TFLOPS   (overlap speedup %.2fx)\n",
            t_seq, flops / t_seq / 1e12, t_seq / t_pipe)

    @info "checking pipelined == sequential (determinism)..."
    @assert Array(D_val) == Array(D_val2) && Array(D_loc) == Array(D_loc2) "pipelined and sequential results differ"

    @info "spot check vs dequantized fp32 reference (8 random rows)..."
    # tolerance covers C storage rounding: ~fp16 ulp of |C| (:f16), ~1e-4
    # accumulation noise (:f32), or 1 e4m3 ulp of the tail values, up to 16
    # at |C|≈250 (:f8)
    tol = cout === :f16 ? 1.0 : cout === :f32 ? 0.05 : 40.0
    maxerr = spot_check(A, B, D_val, D_loc, k; tol)
    @printf("accuracy: max |top-k value - fp32 reference| = %.4g (tolerance %.2g, cout=%s)\n",
            maxerr, tol, string(cout))

    if cout !== :f8
        # the fp8-out experiment: C quantized to e4m3 in DRAM — halves top-k
        # read traffic; does the scan actually get faster, and what does the
        # ~6% value quantization do to the top-k? cuBLASLt rejects e4m3 D
        # (AlgoGetHeuristic status 15), so both passes run on engine=:mma for
        # a controlled cout comparison.
        @info "fp8-out experiment (engine=:mma, cout=:f8 vs :f16)..."
        D_val8 = CuMatrix{Float32}(undef, M, k)
        D_loc8 = CuMatrix{Int32}(undef, M, k)
        D_valb = CuMatrix{Float32}(undef, M, k)
        D_locb = CuMatrix{Int32}(undef, M, k)
        try
            for (co, Dv_, Dl_) in ((:f16, D_valb, D_locb), (:f8, D_val8, D_loc8))
                CUDA.device_synchronize(); t0 = time()
                gemm_topk_fp8!(Dv_, Dl_, A, B; k, w, engine = :mma, cout = co)
                t_ = time() - t0
                @printf("  %-4s %-4s: %6.2f s  →  %.1f TFLOPS\n", "mma", string(co), t_,
                        flops / t_ / 1e12)
            end
            spot_check(A, B, D_val8, D_loc8, k; tol = 40.0)
            v8 = Array(D_val8); l8 = Array(D_loc8)
            vb = Array(D_valb); lb = Array(D_locb)
            @printf("fp8-out vs f16-out: max |Δvalue| = %.2f, %d/%d locations differ (%.1f%% agree)\n",
                    maximum(abs.(v8 .- vb)), count(l8 .!= lb), length(l8),
                    100 * count(l8 .== lb) / length(lb))
            CUDA.unsafe_free!.((D_val8, D_loc8, D_valb, D_locb))
        catch e
            @warn "fp8-out run failed" e
        end
    end

    if exact
        @info "exact fp32 reference: full dequantized sgemm + same top-k kernel..."
        D_ref_val = CuMatrix{Float32}(undef, M, k)
        D_ref_loc = CuMatrix{Int32}(undef, M, k)
        CUDA.device_synchronize(); t0 = time()
        exact_topk!(D_ref_val, D_ref_loc, A, B; k)
        @printf("  exact reference: %.1f s\n", time() - t0)
        v = Array(D_val); l = Array(D_loc)
        ve = Array(D_ref_val); le = Array(D_ref_loc)
        overlap = [length(intersect(@view(l[r, :]), @view(le[r, :]))) for r in 1:M]
        @printf("location agreement vs exact top-%d: %.2f%% of entries (%.2f%% of rows perfect, %.2f%% with ≥%d/%d)\n",
                k, 100 * sum(overlap) / (M * k), 100 * count(==(k), overlap) / M,
                100 * count(>=(k - 2), overlap) / M, k - 2, k)
        @printf("value agreement: max rank-matched |Δ| = %.3f (order-stat gaps when locations swap; ulp-level otherwise)\n",
                maximum(abs.(v .- ve)))
        @printf("k-th value: max |Δ| = %.3f, mean |Δ| = %.4f\n",
                maximum(abs.(v[:, end] .- ve[:, end])), mean(abs.(v[:, end] .- ve[:, end])))
        CUDA.unsafe_free!.((D_ref_val, D_ref_loc))
    end

    if sweep
        @info "chunk width sweep (pipelined, full runs, best config)..."
        for w2 in (2^11, 2^12, 2^14)
            GC.gc(); CUDA.reclaim()
            CUDA.device_synchronize(); t0 = time()
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w = w2, engine = engine, cout = cout)
            tw = time() - t0
            @printf("  w=2^%2d (%2d chunks, %4.0f MiB/buffer): %6.2f s  →  %.1f TFLOPS\n",
                    Int(log2(w2)), cld(N, w2), M * w2 * (cout === :f16 ? 2 : cout === :f32 ? 4 : 1) / 2^20,
                    tw, flops / tw / 1e12)
        end
    end

    if baseline16
        @info "in-process fp16 baseline (cuBLAS mul! GEMM, same top-k kernel)..."
        A16 = CuMatrix{Float16}(undef, M, K); fp16_from_fp8!(A16, A)
        B16 = CuMatrix{Float16}(undef, K, N); fp16_from_fp8!(B16, B)
        GC.gc(); CUDA.reclaim()
        D_val16 = CuMatrix{Float32}(undef, M, k)
        D_loc16 = CuMatrix{Int32}(undef, M, k)
        CUDA.device_synchronize(); t0 = time()
        gemm_topk_fp16!(D_val16, D_loc16, A16, B16; k, w)
        t16 = time() - t0
        @printf("fp16     : %6.2f s  →  %.1f TFLOPS   (fp8 speedup %.2fx)\n",
                t16, flops / t16 / 1e12, t16 / t_pipe)
        v8 = Array(D_val); l8 = Array(D_loc)
        v16 = Array(D_val16); l16 = Array(D_loc16)
        @printf("agreement: max |D_val8 - D_val16| = %.3f, %d/%d top-k locations differ\n",
                maximum(abs.(v8 .- v16)), count(l8 .!= l16), length(l8))
        CUDA.unsafe_free!.((A16, B16, D_val16, D_loc16))
    end

    GC.gc(); CUDA.reclaim()
    return D_val, D_loc
end

if abspath(PROGRAM_FILE) == @__FILE__
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    quick = "--quick" in ARGS
    sweep = !("--no-sweep" in ARGS)
    base16 = "--baseline16" in ARGS
    exactref = "--exact" in ARGS
    normalize = "--normalize" in ARGS
    engstr = getflag("engine", "both")
    engines = engstr == "mma" ? (:mma,) : engstr == "lt" ? (:lt,) : (:mma, :lt)
    if quick
        main(M = 2^14, K = 2^11, N = 2^16, k = 20, w = 2^11, sweep = false,
             engines = engines, baseline16 = base16, exact = exactref, normalize = normalize)
    else
        main(k = 20, sweep = sweep, engines = engines, baseline16 = base16,
             exact = exactref, normalize = normalize)
    end
end
