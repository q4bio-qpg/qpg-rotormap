# Benchmark: FP8 (e4m3) GEMM on cuTile tensor cores vs FP16 GEMM on cuBLAS.
#
# Run with:  julia --project=. test/gemmfp8bench.jl [sizes...]

using CUDA
using cuTile
import cuTile as ct
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Printf

include("gemmfp8.jl")   # brings fp8_gemm! (and runs its correctness test)

const F8 = Float8_E4M3FN

function bench_size(M::Int, N::Int, K::Int; tm = 64, tn = 64, tk = 32,
                    warmup = 2, nruns = 5)
    flops = 2.0 * M * N * K

    # --- FP8 path: quantized operands, cuTile kernel with Float32 accum ---
    A8 = CuArray(F8.(randn(Float32, M, K) .* 0.05f0))
    B8 = CuArray(F8.(randn(Float32, K, N) .* 0.02f0))
    C8 = CuArray{Float32}(undef, M, N)
    for _ in 1:warmup
        fp8_gemm!(A8, B8, C8; tm, tn, tk)
    end
    CUDA.synchronize()
    t8 = Inf
    for _ in 1:nruns
        t8 = min(t8, CUDA.@elapsed fp8_gemm!(A8, B8, C8; tm, tn, tk))
    end

    # --- FP16 path: cuBLAS via LinearAlgebra.mul! ---
    A16 = Float16.(Float32.(A8)) .* Float16(20.0f0)   # same values, dequantized
    B16 = Float16.(Float32.(B8)) .* Float16(50.0f0)
    A16 = CuArray(A16); B16 = CuArray(B16)
    C16 = CuArray{Float16}(undef, M, N)
    for _ in 1:warmup
        mul!(C16, A16, B16)
    end
    CUDA.synchronize()
    t16 = Inf
    for _ in 1:nruns
        t16 = min(t16, CUDA.@elapsed mul!(C16, A16, B16))
    end

    CUDA.unsafe_free!(A8); CUDA.unsafe_free!(B8); CUDA.unsafe_free!(C8)
    CUDA.unsafe_free!(A16); CUDA.unsafe_free!(B16); CUDA.unsafe_free!(C16)

    return (fp8 = t8, fp16 = t16, flops)
end

sizes = isempty(ARGS) ? [512, 1024, 2048, 4096, 8192] : parse.(Int, ARGS)

println("RTX 5090 — FP8 (cuTile, e4m3) vs FP16 (cuBLAS), square GEMM\n")
@printf "%6s | %10s %12s | %10s %12s | %8s\n" "N" "fp8 ms" "fp8 TFLOPS" "fp16 ms" "fp16 TFLOPS" "speedup"
println("-"^72)
for n in sizes
    (; fp8, fp16, flops) = bench_size(n, n, n)
    @printf "%6d | %10.3f %12.1f | %10.3f %12.1f | %7.2fx\n" n fp8*1e3 flops/fp8/1e12 fp16*1e3 flops/fp16/1e12 fp16/fp8
end
