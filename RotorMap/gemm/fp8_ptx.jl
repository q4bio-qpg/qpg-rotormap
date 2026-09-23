# ==============================================================================
# fp8_ptx.jl — the nvcc PTX mma.sync fp8 GEMM engine.
#
# PURPOSE
#   D (M x wi) = A (M x K, e4m3) * B chunk (K x wi, e4m3, raw base pointer +
#   column stride): a 64x64-tile m16n8k32.row.col e4m3 tensor-core mma.sync
#   kernel with fp32 accumulators and fp32/fp16/e4m3 (satfinite) output,
#   compiled by nvcc to PTX (its .version header is stamped down so the pinned
#   CUDA runtime can JIT it) and launched via CUDA.cudacall on the current
#   stream.  Also owns the e4m3 <-> fp32 converter kernels that back
#   fp8_convert.jl's f32_to_f8!/f8_to_f32!.
#
# SOURCES
#   legacy/test/gemmtopkfp8.jl: CUDA_DIR, PTX_VERSION, CU_SRC, k_gemm, k_conv,
#   build_kernel, fp8_gemm_mma! (bodies verbatim).
#
# DEPS
#   include fp8_convert.jl (F8, byteptr); using CUDA.
#
# NOTES
#   Restrictions: M % 64 == 0, wi % 64 == 0, K % 32 == 0, and a 16-byte
#   aligned chunk base (uint4 loads).  CUDA_DIR's default is preserved
#   exactly; the legacy script had no ENV override for it — adjust the const
#   to point at a different nvcc.
# ==============================================================================

include(joinpath(@__DIR__, "fp8_convert.jl"))

using CUDA

const CUDA_DIR = "/usr/local/cuda-13.3"   # nvcc that knows sm_120

# nvcc 13.3 emits PTX ISA 9.3 which the pinned 13.2 runtime cannot JIT, and
# older system nvcc lacks fp8 mma; compile with 13.3 and stamp the .version
# header down to 9.0 (codegen-compatible).
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

# --------------------------------------------------------------------------
# GEMM engine (mma). Takes (Cbuf M×wi, A M×K, Bp chunk base pointer) and
# launches on the current stream.
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

# -----------------------------------------------------------------------------
# test_fp8_gemm -- correctness self-check for BOTH fp8 engines (moved here from
# legacy gemmtopkfp8.jl; the complex experiment's check mode calls it).  For
# :lt it requires gemm/fp8_lt.jl in the session (canonical order provides it).
# -----------------------------------------------------------------------------
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
