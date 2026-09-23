module SplitComplexMatrices

export SplitComplexMatrix #, mul!, hcat, vcat, cu, *, adjoint

import LinearAlgebra: mul!
import Base: hcat, vcat, size, getindex, setindex!, *, adjoint

using CUDA 

struct SplitComplexMatrix
    Re::AbstractArray{<:Real}
    Im::AbstractArray{<:Real}

    function SplitComplexMatrix(A::AbstractArray{T},B::AbstractArray{T}) where {T<:Real}
        return new(A,B)
    end
    function SplitComplexMatrix(M::AbstractArray{T}) where {T<:Complex}
        p = reim(M)
        return new(p[1],p[2])
    end
    function SplitComplexMatrix(p::Tuple{AbstractArray{T},AbstractArray{T}}) where {T<:Real}
        return new(p[1],p[2])
    end
end

function hcat(A::SplitComplexMatrix, B::SplitComplexMatrix)
    return SplitComplexMatrix(hcat(A.Re, B.Re), hcat(A.Im, B.Im))
end

function vcat(A::SplitComplexMatrix, B::SplitComplexMatrix) 
    return SplitComplexMatrix(vcat(A.Re, B.Re), vcat(A.Im, B.Im))
end

function mul!(C::SplitComplexMatrix, A::SplitComplexMatrix, B::SplitComplexMatrix) 
    mul!(C.Re, A.Re, B.Re)
    mul!(C.Re, A.Im, B.Im, -1, 1)
    
    mul!(C.Im, A.Re, B.Im)
    mul!(C.Im, A.Im, B.Re, 1, 1)
    return C
end

function *(A::SplitComplexMatrix, B::SplitComplexMatrix) 
    CRe = A.Re*B.Re 
    mul!(CRe, A.Im, B.Im, -1, 1)

    CIm = A.Re*B.Im
    mul!(CIm, A.Im, B.Re, 1, 1)

    return SplitComplexMatrix(CRe, CIm)
end

function adjoint(A::SplitComplexMatrix)
    Re = adjoint(A.Re)
    Im = adjoint(-A.Im)

    return SplitComplexMatrix(Re, Im)
end

# 1. Required for display and many array functions
Base.size(A::SplitComplexMatrix) = size(A.Re) # Assumes Real and Imag have same size

# 2. Required to get elements (e.g., S[1,1])
function Base.getindex(A::SplitComplexMatrix, i::Int, j::Int)
    return A.Re[i, j] + im * A.Im[i, j]
end

# 3. Optional but good practice to set elements (e.g., S[1,1] = 5+2im)
function Base.setindex!(A::SplitComplexMatrix, val::Complex, i::Int, j::Int)
    Re, Im = reim(val)
    A.Re[i, j] = Re
    A.Im[i, j] = Im
    return A
end

try
    import CUDA: cu, CuArray 
    # CUDA.allowscalar(true) 
catch e
    @warn "no CUDA.jl"
end

# function __init__() # avoids precompilation
function cu(A::SplitComplexMatrix)
    return SplitComplexMatrix(cu(A.Re), cu(A.Im))
end

function CuArray(A::SplitComplexMatrix)
    return SplitComplexMatrix(CuArray(A.Re), CuArray(A.Im))
end

# end

end