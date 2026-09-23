# FP8 (e4m3) matrix multiplication on GPU via cuTile
#
# Quantized A and B are stored as Float8_E4M3FN with scale factors applied on
# the host (A8 = A * scale_a). The kernel multiplies the fp8 tiles directly on
# the FP8 tensor cores, accumulating in Float32. The result is dequantized on
# the host with 1/(scale_a * scale_b) and checked against a Float64 reference.

using CUDA
using cuTile
import cuTile as ct
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Random
using Test

const F8 = Float8_E4M3FN

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

# --- 2. Host-side setup and verification ------------------------------------

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
