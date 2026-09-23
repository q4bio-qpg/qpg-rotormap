# __precompile__(true)
# full test of a workload, used for compilation

using Base.Threads
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using RotorMap.Mapper
using RotorMap.Utils

# if !isdefined(Main, :load)
# include(joinpath(@__DIR__,"../src/utils.jl"))
# end

"""
A function that runs a small but complete workload
"""
function full(; k=10^3, N=10^5, step=10^2, s=5, n=100, err=0.15, show_trace=false)
    if show_trace 
        @info "Running full test with parameters k=$k, N=$N, n=$n, err=$err"
    end

    if show_trace 
        @info "Generating a reference" 
    end
    dna_ref = generate_reference(N)

    if show_trace 
        @info "Generating reads" 
    end
    reads_arr, locs = generate_reads(dna_ref, k, n; err=err)
    reads = hcat(reads_arr...)

    re = RopeEncoder(k=k, s=s, m=1, c=64) # c not used
    ri = RopeIndexer(N=N, re=re, step=step)
    e = re.e |> cu

    if show_trace 
        @info "Constructing the index on GPU"     
    end
    cindex = index_m1_cuda(ri, dna_ref; e=e)
    # cindex = index_m1_cuda_optimized(ri, dna_ref; e=e)

    if show_trace 
        @info "Sending reads to GPU"    
    end
    creads = reads |> cu 

    if show_trace 
        @info "Encoding reads"
    end
    cropes = encode_batch_m1_cuda(re, creads, 1:k, e=e)

    if show_trace 
        @info "Mapping reads"
    end
    cinds = search_batch_m1_cuda(cindex, cropes)

    if show_trace 
        @info "Getting found locations"
    end
    inds = Array(cinds)

    if show_trace 
        @info "Comparing with true locations"
    end
    diffs = inds.*ri.step.-locs .|> abs 

    maxdiff = maximum(diffs) 
    avgdiff = sum(diffs)/n 

    mdp = maxdiff/k*100
    adp = avgdiff/k*100
    
    if show_trace 
        "Mapped precision in % of the read length: " |> println
        "Maximum: $mdp%" |> println
        "Average: $adp%" |> println
        "(Step size is $(ri.step*100/k)%)" |> println
    end
end