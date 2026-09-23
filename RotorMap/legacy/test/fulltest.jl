using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

include(joinpath(@__DIR__,"full.jl"))

full(k=10^5, N=10^9, step=10^4, s=5, n=1000, err=0.15)