# Testing and benchmarking of the SplitComplexMatrices

using Pkg

Pkg.activate(".")

try
    using CUDA
    # CUDA.allowscalar(true)
catch
    # no CUDA 
end

using RotorMap
using RotorMap.SplitComplexMatrices
using LinearAlgebra
using BenchmarkTools



function test(N)
    A = randn(ComplexF64, N, N) 
    B = randn(ComplexF64, N, N) 
    C = zeros(ComplexF64, N, N) 

    AS = A |> SplitComplexMatrix
    BS = B |> SplitComplexMatrix
    CS = C |> SplitComplexMatrix

    mul!(C, A, B)
    mul!(CS, AS, BS)

    CReal, CImag = reim(C)

    return norm(CReal-CS.Re) + norm(CImag-CS.Im)
    # return norm(C-CS.Re-im*CS.Im)
end

function benchmark_normal(N)
    A = randn(ComplexF64, N, N) 
    B = randn(ComplexF64, N, N) 
    C = zeros(ComplexF64, N, N) 
    
    return @benchmark mul!($C, $A, $B)
end

function benchmark_split(N)
    A = randn(ComplexF64, N, N) |> SplitComplexMatrix
    B = randn(ComplexF64, N, N) |> SplitComplexMatrix
    C = zeros(ComplexF64, N, N) |> SplitComplexMatrix
    
    return @benchmark mul!($C, $A, $B)
end

function benchmark_normal_cuda(N)
    A = randn(ComplexF64, N, N) |> cu
    B = randn(ComplexF64, N, N) |> cu
    C = zeros(ComplexF64, N, N) |> cu
    
    return @benchmark mul!($C, $A, $B)
end

function benchmark_split_cuda(N)
    A = randn(ComplexF64, N, N) |> SplitComplexMatrix |> cu
    B = randn(ComplexF64, N, N) |> SplitComplexMatrix |> cu
    C = zeros(ComplexF64, N, N) |> SplitComplexMatrix |> cu
    
    return @benchmark mul!($C, $A, $B)
end

# display(benchmark_normal(512))
# display(benchmark_split(512))

