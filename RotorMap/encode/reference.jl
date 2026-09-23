# ==============================================================================
# reference.jl -- CPU golden-reference rope encoders for verification.
#
# PURPOSE: deliberately naive float64 CPU encoders with the EXACT semantics of
#   the GPU fragment kernels (little-endian s-mer fields, same scramble key
#   and mixer, same G-bin zeroing convention, exp(+i*theta*p) phases,
#   normalize modes 0-3 with the kernels' epsilons) -- an independent
#   implementation, so shared bugs cannot hide.  The encoding is intentionally
#   NOT bitwise the 2-bit-packed batch kernels' one: the bin bijection and the
#   scramble key differ (the forward-fragment packing changes them), but it is
#   the same rope construction.
#
#   `_frag_codes`         forward fragment packing -> base codes 0:3
#   `_ref_rope_frag`      complex histogram + norms (the kernel's semantics)
#   `_ref_rope_frag_real` `_ref_rope_frag` wrapped with the split [Re; Im]
#                         flattening the real-output kernel emits
#
# SOURCES: legacy/test/ropeflow_v3.jl (_frag_codes, _ref_rope_frag) and
#   legacy/test/ropeflowreal.jl (_ref_rope_frag_real).
#
# DEPS: includes ropeencoder.jl (RopeEncoder).  A consumer must include
#   ropeencoder.jl first (or simply include this file, which does it).
#   No `using` needed (pure Base).
# ==============================================================================

include(joinpath(@__DIR__, "ropeencoder.jl"))

"""Forward fragment packing -> base codes 0:3 (the obvious decode)."""
function _frag_codes(words::AbstractVector{UInt32}, k::Int)
    codes = Vector{UInt8}(undef, k)
    for j in 1:k
        codes[j] = (words[(j - 1) >> 4 + 1] >> (2 * ((j - 1) & 15))) & UInt32(3)
    end
    return codes
end

function _ref_rope_frag(codes::Vector{UInt8}, re::RopeEncoder; normalize::Int)
    (; k, s, m, c) = re
    nbin = m * 4^c
    hist = zeros(ComplexF64, nbin)
    cbmask = UInt32(4^c - 1)
    nw = k - s + 1
    for p in 0:nw-1
        smer = UInt32(0)
        @inbounds for j in 0:s-1
            smer |= UInt32(codes[p+j+1]) << (2 * j) # little-endian field
        end
        lower = smer & cbmask
        upper = smer >> (2 * c)
        csmer = Int(lower ⊻ ((upper * 0x9E3779B1) & cbmask)) + 1
        for idm in 1:m
            θ = (m == 1 ? 1.0 : 2 * (idm - 1) / (m - 1) + 1) * (2π / k)
            hist[(csmer - 1) * m + idm] += cis(θ * p) # exp(+i*theta*p)
        end
    end
    allones = sum(UInt32(4)^i for i in 0:c-1)
    csmerG = 2 * allones + 1
    for idm in 1:m
        hist[(csmerG - 1) * m + idm] = 0.0
    end

    norms = zeros(m)
    if normalize == 0
        nrm = sqrt(sum(abs2, hist)) + 1e-16
        hist ./= nrm
        norms[1] = nrm
    elseif normalize == 1
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + 1e-7
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    elseif normalize == 2
        for bin in 1:nbin
            hist[bin] = hist[bin] / (abs(hist[bin]) + eps())
        end
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + eps()
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    else # 3
        for bin in 1:nbin
            hist[bin] = hist[bin] / sqrt(abs(hist[bin]) + eps())
        end
        for idm in 1:m
            nrm = sqrt(sum(abs2, @view(hist[idm:m:nbin]))) + eps()
            hist[idm:m:nbin] ./= nrm
            norms[idm] = nrm
        end
    end
    return hist, norms
end

"""_ref_rope_frag's complex histogram -> the kernel's split [Re; Im] real vector."""
function _ref_rope_frag_real(codes::Vector{UInt8}, re::RopeEncoder; normalize::Int)
    hist, norms = _ref_rope_frag(codes, re; normalize)
    D = length(hist)
    out = Vector{Float64}(undef, 2 * D)
    @inbounds for i in 1:D
        out[i] = real(hist[i])
        out[D+i] = imag(hist[i])
    end
    return out, norms
end
