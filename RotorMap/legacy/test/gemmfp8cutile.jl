# FP8 (e4m3) matrix multiplication in pure Julia with cuTile.jl: the kernel
# operates on tiles and is lowered to NVIDIA Tile IR, which tileiras compiles
# to cubin — the same tensor-core GEMM as gemmfp8cpp.jl's hand-written
# mma.sync PTX kernel, but with tile scheduling left to the compiler.
#
# How it works: cuTile kernels load multi-dimensional array fragments (tiles),
# and `muladd(a, b, acc)` on two 2-D tiles lowers to a Tile IR `mmaf`
# tensor-core op; tileiras picks the mma shape, shared-memory staging, TMA
# copies and warp schedule. The e4m3 element type comes from DLFP8Types
# (Float8_E4M3FN), which cuTile picks up through a package extension; tile
# `muladd` accepts e4m3 operands with an f16 or f32 accumulator. We use a
# Float32 accumulator (fp32 accumulation, like the PTX kernel's
# m16n8k32.row.col.f32.e4m3 mma). `muladd(...; fast_acc=true)` is an fp8-only
# hint, but it only changes codegen on Hopper (sm_90) and is ignored on our
# sm_120, so we don't set it. fp8 mma needs sm_90+; cuTile itself needs a
# driver with CUDA 13 (580+) and ships `tileiras` via CUDA.jl artifacts.
#
# Kernel shape: 1-D grid with 2-D block swizzle for L2 locality (cuTile's
# matmul example), K loop over TK-sized slabs. Loads use PaddingMode.Zero and
# stores are clipped, so any (M, N, K) works — unlike the PTX kernel, which
# needs M,N % 64 == 0 and K % 32 == 0. Benchmarks use the same multiples-of-64
# shapes for comparability.
#
# NB: don't quote absolute numbers across scripts — run gemmfp8cpp.jl and this
# file as separate processes and compare ratios within one run (GPU thermal
# state and operand bit patterns shift absolute TFLOPS by tens of percent).

using CUDA
using cuTile
import cuTile as ct
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Printf
using Random
using Test

const F8 = Float8_E4M3FN

# tile configs swept by the benchmark: (TM, TN, TK)
const TILE_CONFIGS = [(64, 64, 64), (128, 64, 64), (64, 128, 64), (128, 128, 64), (128, 128, 128)]

# --- kernel -------------------------------------------------------------------

# 2D block swizzle for L2 cache locality (from cuTile's matmul example):
# 1-indexed bid in, 1-indexed (bid_m, bid_n) out; modular arithmetic is done on
# the 0-indexed bid internally.
@inline function swizzle_2d(M, N, tm, tn, group_m, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(group_m) * num_bid_n
    bid0 = bid - Int32(1)
    group_id = fld(bid0, num_bid_in_group)
    first_bid_m = group_id * Int32(group_m)
    group_size_m = min(num_bid_m - first_bid_m, Int32(group_m))
    bid_m = first_bid_m + rem(bid0, group_size_m) + Int32(1)
    bid_n = fld(rem(bid0, num_bid_in_group), group_size_m) + Int32(1)
    return bid_m, bid_n
end

# D = A * B, all Julia column-major: A (M, K) e4m3, B (K, N) e4m3, D (M, N) f32.
# tm/tn/tk arrive as compile-time constants (wrapped in ct.Constant at the
# launch site), so each tile config compiles its own specialized kernel.
function fp8_gemm_kernel(A::ct.TileArray{Float8_E4M3FN,2}, B::ct.TileArray{Float8_E4M3FN,2},
                         D::ct.TileArray{Float32,2}, tm::Int, tn::Int, tk::Int)
    bid = ct.bid(1)
    M = size(A, 1)
    N = size(B, 2)
    bid_m, bid_n = swizzle_2d(M, N, tm, tn, 8, bid)

    num_k = ct.num_tiles(A, 2, (tm, tk))
    acc = zeros(Float32, tm, tn)                # this block's (M, N) f32 fragment
    for k in Int32(1):num_k
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)                 # tensor-core mmaf: e4m3 x e4m3 + f32
    end
    ct.store(D; index=(bid_m, bid_n), tile=acc) # partially out-of-bounds tiles clip
    return
end

function fp8_gemm_cutile!(D::CuMatrix{Float32}, A::CuMatrix{F8}, B::CuMatrix{F8};
                          tm::Int = 64, tn::Int = 64, tk::Int = 64)
    grid = cld(size(A, 1), tm) * cld(size(B, 2), tn)
    @cuda backend=cuTile blocks=grid fp8_gemm_kernel(A, B, D,
                                                     ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
    return D
end

# --- correctness ---------------------------------------------------------------

fp8_reference(A::CuMatrix{F8}, B::CuMatrix{F8}) =
    Float64.(Float32.(Array(A))) * Float64.(Float32.(Array(B)))

@testset "fp8 gemm via cuTile" begin
    Random.seed!(42)

    # same setup as gemmfp8cpp.jl: scaled randn, error measured against the
    # de-scaled f64 product of the quantized operands
    M, N, K = 256, 256, 256
    scale_a, scale_b = 0.05f0, 0.02f0
    A_gpu = CuArray(F8.(randn(Float32, M, K) .* scale_a))
    B_gpu = CuArray(F8.(randn(Float32, K, N) .* scale_b))
    D_gpu = CUDA.zeros(Float32, M, N)

    fp8_gemm_cutile!(D_gpu, A_gpu, B_gpu)
    A_ref = Float64.(Float32.(Array(A_gpu))) ./ scale_a
    B_ref = Float64.(Float32.(Array(B_gpu))) ./ scale_b
    C_ref = A_ref * B_ref
    relerr = norm(Array(D_gpu) ./ (scale_a * scale_b) .- C_ref) / norm(C_ref)
    @test relerr < 1e-6
    @show relerr

    # ragged shapes the PTX kernel cannot run: tiles are zero-padded on load
    # and clipped on store, so any (M, N, K) is handled
    for (M, N, K) in ((100, 77, 96), (1, 1, 1), (63, 65, 127))
        A_gpu = CuArray(F8.(randn(Float32, M, K)))
        B_gpu = CuArray(F8.(randn(Float32, K, N)))
        D_gpu = CUDA.zeros(Float32, M, N)
        fp8_gemm_cutile!(D_gpu, A_gpu, B_gpu; tm = 64, tn = 64, tk = 32)
        ref = fp8_reference(A_gpu, B_gpu)
        relerr = norm(Array(D_gpu) .- ref) / norm(ref)
        @test relerr < 1e-6
        @show (M, N, K) relerr
    end
end

# --- benchmark vs FP16 cuBLAS ---------------------------------------------------

function timef(f; warmup = 2, nruns = 5)
    for _ in 1:warmup; f(); end
    CUDA.synchronize()
    t = Inf
    for _ in 1:nruns; t = min(t, CUDA.@elapsed f()); end
    return t
end

# sanity-check a tile config at 256^3 before letting it into the benchmark
function config_is_correct(tm::Int, tn::Int, tk::Int)
    Random.seed!(1234)
    A = CuArray(F8.(randn(Float32, 256, 256)))
    B = CuArray(F8.(randn(Float32, 256, 256)))
    D = CUDA.zeros(Float32, 256, 256)
    fp8_gemm_cutile!(D, A, B; tm, tn, tk)
    ref = fp8_reference(A, B)
    ok = norm(Array(D) .- ref) / norm(ref) < 1e-5
    CUDA.unsafe_free!.((A, B, D))
    return ok
end

function bench(M::Int, N::Int, K::Int, configs; warmup = 2, nruns = 5)
    flops = 2.0 * M * N * K

    A8 = CuArray(F8.(randn(Float32, M, K) .* 0.05f0))
    B8 = CuArray(F8.(randn(Float32, K, N) .* 0.02f0))
    D8 = CUDA.zeros(Float32, M, N)
    cutile = [(cfg, timef(() -> fp8_gemm_cutile!(D8, A8, B8; tm = cfg[1], tn = cfg[2], tk = cfg[3]);
                          warmup, nruns)) for cfg in configs]

    A16 = CuArray(Float16.(Float32.(A8)))
    B16 = CuArray(Float16.(Float32.(B8)))
    D16 = CuArray{Float16}(undef, M, N)
    t16 = timef(() -> mul!(D16, A16, B16); warmup, nruns)

    CUDA.unsafe_free!.((A8, B8, D8, A16, B16, D16))
    return (cutile, t16, flops)
end

sizes = isempty(ARGS) ? [512, 1024, 2048, 4096, 8192] : parse.(Int, ARGS)

println("RTX 5090 — FP8 (e4m3) GEMM via cuTile.jl (Tile IR) vs FP16 (cuBLAS), square GEMM\n")
ct.versioninfo()

println("\nverifying tile configs at 256^3:")
configs = filter(TILE_CONFIGS) do cfg
    ok = config_is_correct(cfg...)
    @printf "  %-12s %s\n" join(cfg, "x") (ok ? "ok" : "WRONG — dropped")
    ok
end
isempty(configs) && error("no working tile config")

for n in sizes
    (cutile, t16, flops) = bench(n, n, n, configs)
    @printf "\n%6d\n" n
    best = Inf
    for (cfg, t) in cutile
        @printf "   cutile %-12s %10.3f ms %10.1f TFLOPS\n" join(cfg, "x") t * 1e3 flops / t / 1e12
        best = min(best, t)
    end
    @printf "   %-19s %10.3f ms %10.1f TFLOPS   (best cuTile / fp16 = %.2fx)\n" "fp16 cuBLAS" t16 * 1e3 flops / t16 / 1e12 t16 / best
end
