# (verbatim copy of legacy/test/gemmfp8cpp.jl -- self-contained C++/PTX mma + cuBLASLt fp8 GEMM study; its header documents the kau driver-userland situation: the container's stale ld.so.cache resolves libcuda.so.1 to a 580 compat build which the 595 open kernel module rejects (CUDA error 803), so ~/.bashrc must prepend the Nix 595.71.05 libs to LD_LIBRARY_PATH before Julia starts)
# FP8 (e4m3) matrix multiplication, two ways: a C++/CUDA PTX mma.sync
# tensor-core kernel compiled with nvcc into a cubin and launched from Julia
# with CUDA.cudacall, and a pure-ccall cuBLASLt path for comparison.
#
# Note: cuBLASLt's fp8 path works — and is ~2x faster than fp16 cuBLAS (see the
# "lt8" benchmark column) — but only when the driver userland matches the 595
# open-kernel-module driver: the container's stale ld.so.cache resolves
# libcuda.so.1 to a 580 compat build, which the driver rejects (CUDA error 803)
# and under which cublasLtMatmul used to segfault. ~/.bashrc must prepend the
# Nix 595.71.05 libs to LD_LIBRARY_PATH before Julia starts. Unlike CUDA.jl's
# 13.2 JLL artifact, the toolkit's own libcublasLt (cuda-13.4 / cuda-13.3) is
# verified exact on sm_120, so we ccall into it directly.
#
# Restrictions: M, N multiples of 64; K multiple of 32.
#   A: M x K row-major Float8_E4M3FN, B: K x N row-major, D: M x N Float32.

using CUDA
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Libdl
using Printf
using Random
using Test

const F8 = Float8_E4M3FN

const CUDA_DIR = "/usr/local/cuda-13.3"   # nvcc that knows sm_120

# nvcc 13.3 emits PTX ISA 9.3 which the pinned 13.2 runtime cannot JIT, and the
# system nvcc 12.0 emits ISA 8.0 which lacks fp8 mma; we compile with 13.3 and
# stamp the .version header down to 9.0 (the codegen is compatible).
const PTX_VERSION = "9.0"

const CU_SRC = raw"""
#include <cuda_runtime.h>
#include <cstdint>

#define BM 64
#define BN 64
#define BK 32

// C = A * B, all Julia (column-major) layout, fp8 inputs / fp32 output.
// Block tile 64x64, K-step 32; 4 warps, each warp computes a 16x64 strip
// with 8 mma.sync.aligned.m16n8k32.e4m3 tensor-core ops per K step.
extern "C" __global__ void fp8_gemm_kernel(const uint8_t* __restrict__ A,
                                           const uint8_t* __restrict__ B,
                                           float* __restrict__ D,
                                           int M, int N, int K)
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
            *(uint4*)&BsT[n][k] = *(const uint4*)&B[(size_t)(n0 + n) * K + kt + k];
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

    // store accumulators (m16n8 f32 fragment layout), D column-major
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int row = a0 + gr + gid;
        const int col = n0 + nt * 8 + tig * 2;
        float* d = &D[(size_t)col * M + row];
        d[0] = acc[nt][0]; d[M] = acc[nt][1];      // (row, col), (row, col+1)
        d[8] = acc[nt][2]; d[M + 8] = acc[nt][3];  // (row+8, col), (row+8, col+1)
    }
}
"""

const k_fp8_gemm = Ref{CUDA.CuFunction}()

function build_kernel()
    dir = joinpath(tempdir(), "gemmfp8cpp")
    mkpath(dir)
    cu = joinpath(dir, "fp8_gemm.cu")
    cubin = joinpath(dir, "fp8_gemm.ptx")
    write(cu, CU_SRC)
    nvcc = joinpath(CUDA_DIR, "bin", "nvcc")
    run(`$nvcc -O3 -arch=compute_120 -ptx $cu -o $cubin`)
    # stamp the PTX ISA version down to what the 13.2 runtime accepts
    ptx = read(cubin, String)
    ptx = replace(ptx, r"^\.version .*$"m => ".version " * PTX_VERSION)
    write(cubin, ptx)
    lib = CUDA.CuModule(read(cubin, String))
    k_fp8_gemm[] = CUDA.CuFunction(lib, "fp8_gemm_kernel")
    return nothing
end

fp8_gemm_cpp!(A::CuMatrix{F8}, B::CuMatrix{F8}, D::CuMatrix{Float32}) =
    CUDA.cudacall(k_fp8_gemm[], (CuPtr{UInt8}, CuPtr{UInt8}, CuPtr{Float32}, Int32, Int32, Int32),
             A, B, D, size(A, 1), size(B, 2), size(A, 2);
             blocks = (cld(size(B, 2), 64), cld(size(A, 1), 64)), threads = 128)

# --- cuBLASLt fp8 path --------------------------------------------------------
# D = A * B via cublasLtMatmul, ccalled directly into the toolkit's libcublasLt.
# fp8 requires op(A) = A^T / op(B) = B: our col-major M x K A buffer is exactly
# a row-major K x M matrix (ORDER_ROW), B is col-major K x N, D col-major M x N.
# Layouts and the heuristic algo are cached per (M, N, K); fp8 leading dims must
# be 16-byte multiples (satisfied by the shapes used below). This coexists with
# CUDA.jl's own libcublasLt (the 13.2 JLL artifact CUBLAS.jl loads for fp16
# mul!): two copies live side by side with independent handles -- verified to
# work correctly whichever loads first.

# Enum values below were probed from the cuda-13.4 headers, not trusted to
# memory -- several are unintuitive. E.g. CUDA_R_8F_E4M3 is 28: the "obvious"
# 14 is CUDA_R_16BF, which would silently type-pun the operands. The heuristic
# result stride matters too (96 B, not 104): the algo bytes must be sliced at
# that stride.
const LT_R_32F       = Cint(0)    # cudaDataType CUDA_R_32F
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

const lt_lib       = Ref{Ptr{Cvoid}}(C_NULL)   # dlopen handle
const lt_handle    = Ref{Ptr{Cvoid}}(C_NULL)   # cublasLtHandle_t
const lt_opdesc    = Ref{Ptr{Cvoid}}(C_NULL)   # matmul desc: op(A)=A^T, op(B)=B, 32F
const lt_pref      = Ref{Ptr{Cvoid}}(C_NULL)   # matmul preference: 32 MiB workspace
const p_ltMatmul   = Ref{Ptr{Cvoid}}(C_NULL)
const lt_workspace = Ref{CuVector{UInt8}}()    # 32 MiB
const lt_cache = Dict{NTuple{3,Int}, Tuple{Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Vector{UInt8}}}()
# lt_cache: (M, N, K) -> (layoutA, layoutB, layoutD, algo bytes)

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

function lt_entries(M::Int, N::Int, K::Int)
    get!(lt_cache, (M, N, K)) do
        sym(s) = Libdl.dlsym(lt_lib[], s)
        layout(dt, rows, cols, ld) = begin
            l = Ref{Ptr{Cvoid}}(C_NULL)
            checklt(ccall(sym(:cublasLtMatrixLayoutCreate), Cint,
                          (Ptr{Ptr{Cvoid}}, Cint, Culonglong, Culonglong, Clonglong),
                          l, dt, rows, cols, ld), "LayoutCreate")
            l[]
        end
        # A: our col-major M x K buffer seen as row-major K x M (op(A) = A^T).
        # Trap: the canonical-looking ORDER_COL (rows=K, cols=M, ld=K) instead
        # also runs without any error but contracts the wrong axis -- garbage
        # results that an all-ones fill cannot detect (1.0 products sum to K
        # under any consistent transposition); only a randomized check catches it.
        la = layout(LT_R_8F_E4M3, K, M, M)
        order = Ref(LT_ORDER_ROW)
        checklt(ccall(sym(:cublasLtMatrixLayoutSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                      la, LT_LAYOUT_ORDER, order, sizeof(Cint)), "Layout ORDER")
        lb = layout(LT_R_8F_E4M3, K, N, K)
        ld = layout(LT_R_32F, M, N, M)
        buf = Vector{UInt8}(undef, LT_HEUR_SZ * 4)
        nres = Ref{Cint}(0)
        checklt(ccall(sym(:cublasLtMatmulAlgoGetHeuristic), Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{UInt8}, Ptr{Cint}),
                      lt_handle[], lt_opdesc[], la, lb, ld, ld, lt_pref[], 4, buf, nres),
                "AlgoGetHeuristic")
        nres[] >= 1 || error("cublasLt: no fp8 algorithm for ($M, $N, $K)")
        (la, lb, ld, copy(buf[1:LT_ALGO_SZ]))
    end
end

function fp8_gemm_cublaslt!(A::CuMatrix{F8}, B::CuMatrix{F8}, D::CuMatrix{Float32})
    lt_handle[] == C_NULL && cublaslt_init()
    la, lb, ld, algo = lt_entries(size(A, 1), size(B, 2), size(A, 2))
    alpha, beta = Ref(1.0f0), Ref(0.0f0)
    checklt(ccall(p_ltMatmul[], Cint,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{F8}, Ptr{Cvoid},
                   CuPtr{F8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{Float32}, Ptr{Cvoid},
                   CuPtr{Float32}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                  lt_handle[], lt_opdesc[], alpha, A, la, B, lb, beta, D, ld, D, ld,
                  pointer(algo), lt_workspace[], sizeof(lt_workspace[]), CUDA.stream().handle),
            "cublasLtMatmul")
    return D
end

# --- correctness -------------------------------------------------------------

@testset "fp8 gemm via C++ mma kernel" begin
    build_kernel()

    Random.seed!(42)
    M, N, K = 256, 256, 256
    scale_a, scale_b = 0.05f0, 0.02f0

    A_gpu = CuArray(F8.(randn(Float32, M, K) .* scale_a))
    B_gpu = CuArray(F8.(randn(Float32, K, N) .* scale_b))
    C_gpu = CUDA.zeros(Float32, M, N)

    fp8_gemm_cpp!(A_gpu, B_gpu, C_gpu)
    CUDA.synchronize()

    A_ref = Float64.(Float32.(Array(A_gpu))) ./ scale_a
    B_ref = Float64.(Float32.(Array(B_gpu))) ./ scale_b
    C_ref = A_ref * B_ref
    C_h = Array(C_gpu) ./ (scale_a * scale_b)

    relerr = norm(C_h .- C_ref) / norm(C_ref)
    @test relerr < 1e-6
    @show relerr

    C_lt = CUDA.zeros(Float32, M, N)
    fp8_gemm_cublaslt!(A_gpu, B_gpu, C_lt)
    CUDA.synchronize()
    relerr_lt = norm(Array(C_lt) ./ (scale_a * scale_b) .- C_ref) / norm(C_ref)
    @test relerr_lt < 1e-6
    @show relerr_lt
end

# --- benchmark vs FP16 cuBLAS -------------------------------------------------
# NB: in-process lt8 numbers trail standalone C probes (~680 vs 828 TFLOPS at
# 4096^3, ~1.65x vs ~2x at 8192^3): the GPU is already warm/power-throttled by
# the time the benchmark runs, and tensor-core power draw depends on operand
# bit patterns (this bench uses randn data; the C probe used all-ones). Compare
# like with like when quoting absolute numbers -- the ratios within one run are
# what is meaningful here.

function timef(f; warmup = 2, nruns = 5)
    for _ in 1:warmup; f(); end
    CUDA.synchronize()
    t = Inf
    for _ in 1:nruns; t = min(t, CUDA.@elapsed f()); end
    return t
end

function bench(M::Int, N::Int, K::Int; warmup = 2, nruns = 5)
    flops = 2.0 * M * N * K

    A8 = CuArray(F8.(randn(Float32, M, K) .* 0.05f0))
    B8 = CuArray(F8.(randn(Float32, K, N) .* 0.02f0))
    C8 = CUDA.zeros(Float32, M, N)
    t8  = timef(() -> fp8_gemm_cpp!(A8, B8, C8); warmup, nruns)
    tlt = timef(() -> fp8_gemm_cublaslt!(A8, B8, C8); warmup, nruns)

    A16 = CuArray(Float16.(Float32.(A8)))
    B16 = CuArray(Float16.(Float32.(B8)))
    C16 = CuArray{Float16}(undef, M, N)
    t16 = timef(() -> mul!(C16, A16, B16); warmup, nruns)

    CUDA.unsafe_free!(A8); CUDA.unsafe_free!(B8); CUDA.unsafe_free!(C8)
    CUDA.unsafe_free!(A16); CUDA.unsafe_free!(B16); CUDA.unsafe_free!(C16)
    return (fp8 = t8, lt8 = tlt, fp16 = t16, flops)
end

sizes = isempty(ARGS) ? [512, 1024, 2048, 4096, 8192] : parse.(Int, ARGS)

println("RTX 5090 — FP8 (mma.sync kernel & cuBLASLt, e4m3) vs FP16 (cuBLAS), square GEMM\n")
@printf "%6s | %10s %12s | %10s %12s | %10s %12s | %9s\n" "N" "fp8 ms" "fp8 TFLOPS" "lt8 ms" "lt8 TFLOPS" "fp16 ms" "fp16 TFLOPS" "lt8 vs 16"
println("-"^96)
for n in sizes
    (; fp8, lt8, fp16, flops) = bench(n, n, n)
    @printf "%6d | %10.3f %12.1f | %10.3f %12.1f | %10.3f %12.1f | %8.2fx\n" n fp8*1e3 flops/fp8/1e12 lt8*1e3 flops/lt8/1e12 fp16*1e3 flops/fp16/1e12 fp16/lt8
end
