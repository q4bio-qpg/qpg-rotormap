# ==============================================================================
# fp8_cutile.jl -- FP8 (e4m3) matrix multiplication in pure Julia with
# cuTile.jl: SELF-CONTAINED kernel playground (no RotorMap deps).
#
# MERGED SOURCE: legacy/test/gemmfp8.jl (basic cuTile fp8 GEMM + its
# correctness test) + legacy/test/gemmfp8cutile.jl (the advanced cuTile study:
# 2-D block swizzle for L2 locality, tile-config sweep vs fp16 cuBLAS).
# Both are cuTile fp8 GEMM studies; their (few) top-level names do not
# collide -- `F8` is defined once here, the basic kernel keeps
# `fp8_gemm_kernel!`/`fp8_gemm!`, the advanced one `fp8_gemm_kernel`/
# `fp8_gemm_cutile!` (+ swizzle_2d/fp8_reference/timef/config_is_correct/
# bench/TILE_CONFIGS).
#
# RESTRUCTURE vs the sources: both legacy files ran a @testset at INCLUDE
# time (gemmfp8bench.jl included gemmfp8.jl exactly for that side effect).
# Here the tests are wrapped in `correctness_test()` (call it explicitly;
# experiments/bench_gemm_fp8.jl does, preserving the old include behavior),
# and the cutile benchmark driver is `bench_sweep()` (ran at top level under
# PROGRAM_FILE in the source; same dispatch kept via the guard below).
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
# Kernel shape (advanced kernel): 1-D grid with 2-D block swizzle for L2
# locality (cuTile's matmul example), K loop over TK-sized slabs. Loads use
# PaddingMode.Zero and stores are clipped, so any (M, N, K) works -- unlike
# the PTX kernel (lab/fp8_cpp.jl), which needs M,N % 64 == 0 and K % 32 == 0.
# Benchmarks use the same multiples-of-64 shapes for comparability.
#
# NB: don't quote absolute numbers across scripts -- run lab/fp8_cpp.jl and
# this file as separate processes and compare ratios within one run (GPU
# thermal state and operand bit patterns shift absolute TFLOPS by tens of
# percent).
# ==============================================================================

using CUDA
using cuTile
import cuTile as ct
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Printf
using Random
using Test

const F8 = Float8_E4M3FN

# ##############################################################################
# SECTION 1 -- the basic kernel (verbatim from legacy/test/gemmfp8.jl)
# ##############################################################################

# --- 1. Tile-based FP8 GEMM kernel -----------------------------------------

# C = A * B, with A::F8 (M, K), B::F8 (K, N), C::Float32 (M, N).
# FP8 tensor cores accumulate into Float32 tiles for numerical stability.
function fp8_gemm_kernel!(C::ct.TileArray{Float32, 2},
                          A::ct.TileArray{F8, 2},
                          B::ct.TileArray{F8, 2},
                          tm::Int, tn::Int, tk::Int)
    N = size(B, 2)
    # 1D grid decomposed into tile coordinates (1-indexed)
    num_n = cld(N, tn)
    bid0 = ct.bid(1) - Int32(1)
    bid_m = fld(bid0, Int32(num_n)) + Int32(1)
    bid_n = rem(bid0, Int32(num_n)) + Int32(1)

    acc = zeros(Float32, tm, tn)

    # Step through the K dimension in tile-sized strides
    num_k = ct.num_tiles(A, 2, (tm, tk))
    for k in Int32(1):num_k
        # Load Float8_E4M3FN tiles; Zero padding covers non-aligned sizes
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        # Hardware-accelerated FP8 matrix multiply-accumulate (acc in Float32)
        acc = muladd(a, b, acc)
    end

    ct.store(C; index=(bid_m, bid_n), tile=convert(ct.Tile{Float32}, acc))
    return nothing
end

function fp8_gemm!(A::CuMatrix{F8}, B::CuMatrix{F8}, C::CuMatrix{Float32};
                   tm::Int = 64, tn::Int = 64, tk::Int = 32)
    M, K = size(A)
    N = size(B, 2)
    grid = cld(M, tm) * cld(N, tn)
    @cuda backend=cuTile blocks=grid fp8_gemm_kernel!(
        C, A, B, ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
    return C
end

# ##############################################################################
# SECTION 2 -- the advanced cuTile kernel + benchmark machinery (verbatim from
# legacy/test/gemmfp8cutile.jl)
# ##############################################################################

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

# ##############################################################################
# Correctness -- both sources' top-level @testsets, verbatim, wrapped in
# correctness_test() (the legacy files ran these as an include side effect)
# ##############################################################################

"""
    correctness_test()

Run the two fp8 GEMM correctness testsets of the merged sources: the basic
kernel (`fp8_gemm!`, scaled-operand dequantization check, ~% tolerances) and
the advanced swizzled kernel (`fp8_gemm_cutile!`, exact-vs-f64 check plus the
ragged shapes the PTX kernel cannot run).  experiments/bench_gemm_fp8.jl calls
this right after including this file -- the old gemmfp8bench.jl relied on
gemmfp8.jl running its test at include time.
"""
function correctness_test()
    @testset "fp8 gemm" begin
        Random.seed!(42)

        M, N, K = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 64, 64, 32

        # Quantization scale factors (absorb the scale on the host side)
        scale_a = 0.05f0
        scale_b = 0.02f0

        # Random real matrices, scaled into the e4m3 range (max ~448)
        A32 = randn(Float32, M, K)
        B32 = randn(Float32, K, N)

        A_gpu = CuArray(F8.(A32 .* scale_a))
        B_gpu = CuArray(F8.(B32 .* scale_b))
        C_gpu = CuArray{Float32}(undef, M, N)

        fp8_gemm!(A_gpu, B_gpu, C_gpu; tm = TILE_M, tn = TILE_N, tk = TILE_K)
        CUDA.synchronize()

        # Reference: dequantized fp8 operands, accumulated in Float64
        A_ref = Float64.(Float32.(Array(A_gpu))) ./ scale_a
        B_ref = Float64.(Float32.(Array(B_gpu))) ./ scale_b
        # Dequantize: the kernel computed A8*B8 = (A*scale_a)*(B*scale_b)
        C_h = Array(C_gpu) ./ (scale_a * scale_b)
        C_ref = A_ref * B_ref
        # fp8 mantissa has 3 bits (~6% per-element quantization error); with random
        # signs the K-sum averages out, so require a few % relative Frobenius error
        relerr = norm(C_h .- C_ref) / norm(C_ref)
        @test relerr < 0.02
        @show relerr maximum(abs.(C_h .- C_ref))

        # Also check elementwise with generous fp8 tolerance
        @test isapprox(C_h, C_ref; rtol = 0.05, atol = 0.05 * norm(C_ref) / sqrt(length(C_ref)))
    end
    println("gemmfp8 test passed")
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
    return nothing
end

# ##############################################################################
# PROGRAM_FILE entry -- the cutile-vs-fp16 square-GEMM sweep (the legacy
# bottom-of-file driver of gemmfp8cutile.jl, verbatim, in bench_sweep())
# ##############################################################################

"""
    bench_sweep()

The advanced kernel's benchmark driver: verify each TILE_CONFIGS entry at
256^3, then sweep square sizes (ARGS, default 512 1024 2048 4096 8192)
comparing every cuTile tile config against the fp16 cuBLAS baseline.
"""
function bench_sweep()
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
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench_sweep()
end
