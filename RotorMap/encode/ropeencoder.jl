# ==============================================================================
# ropeencoder.jl -- the minimal rope encoder parameter struct (vendored).
#
# PURPOSE: single source of the `RopeEncoder` struct and its `dim` accessor.
#   This is the parameter record only; the batch GPU encoders live in
#   encoder_v3.jl / kernel.jl and the CPU golden references in reference.jl.
#
# SOURCES: vendored verbatim from the legacy package (legacy/src/RopeEncoders.jl,
#   struct + `dim`).  The legacy package's many CPU/GPU encode functions are
#   NOT carried here -- they are superseded by the encode/ layer encoders.
#
# DEPS: none (pure Base).  Standalone file; a consumer need not include
#   anything before it.
# ==============================================================================

struct RopeEncoder
    k::Int64 # baseline angle denominator, should match the DNA length
    s::Int64 # size of the small s-mer
    m::Int64 # multiplicity; it should be m=2^q if we target a qubit system; m=1 is the default
    c::Int64 # how many s-mers to use in the compact version; 0 to skip compact version
    # e::Vector{ComplexF64} # cache for exponents

    function RopeEncoder(;k,s=5,m=1,c=0)
        # e = exp.(2π*im/k .* (0:k+s))
        # return new(k,s,m,c,e)
        return new(k,s,m,c)
    end
end

dim(re::RopeEncoder) = (re.c==0 ? re.m*4^re.s : re.m*4^re.c)
