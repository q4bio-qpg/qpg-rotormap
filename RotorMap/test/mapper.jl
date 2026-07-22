using Revise

using Base.Threads
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using RotorMap.Mapper
using RotorMap.Utils


# test of the sliding kernel
function test2()
    @info "Loading index"   
    @time ri = load(joinpath(@__DIR__,"../indexs5m1.bin"))

    # @info "Loading reference"
    # dna_refs, heads = load(joinpath(@__DIR__,"../ref.bin"))

    # dna_ref = dna_refs[1] # todo: multiple refs

    # re = RopeEncoder(k=100_000, s=5, m=1, c=0)
    # ri = RopeIndexer(dna_ref; re=re, step=re.k÷10, err=0.3)
    
    # # @info "Constructing the index on GPU"     
    # CUDA.@time index_cuda_best_new!(ri)    
    # save(ri, "indexs5m1.bin")

    # throw("saved")

    @info "Loading reads"   
    # @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn10kl100ke30.bin"))
    @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn10kl100ke10.bin"))
    # @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn50kl100ke10.bin"))                   

    pos = [parse(Int, head[3:end]) for head in heads] # todo: what about descriptions?
    # @time reads = hcat(seqs...) # todo: it's more efficient to directly transfer to GPU
    @time reads = vcat(seqs...)
    
    @info "Sending reads to GPU"    
    # CUDA.@time creads_t = transpose(reads) |> CuArray 
    CUDA.@time creads = reads |> CuArray 
    # CUDA.@time creads = (@view reads[:,1:10000]) |> CuArray 

    starts = 1:ri.re.k:length(reads) |> collect
    stops = ri.re.k:ri.re.k:length(reads) |> collect
    if length(stops) < length(starts) 
        push!(stops, length(reads))
    end
    starts = starts .|> UInt32 |> CuArray
    stops = stops .|> UInt32 |> CuArray

    # re = RopeEncoder(k=ri.re.k, s=3, m=3, c=0) 
    # re2 = RopeEncoder(k=ri.re.k, s=6, m=16, c=0) 
    re2 = RopeEncoder(k=ri.re.k, s=5, m=64, c=0)
    # re2 = RopeEncoder(k=ri.re.k, s=6, m=16, c=0) 
    # re2 = RopeEncoder(k=ri.re.k, s=8, m=4, c=0) 

    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, e=ri.cu_e)
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads_t, e=ri.cu_e)
    # @show size(cropes)

    CUDA.@time locs = search_main(ri, re2, creads, starts, stops; pos=pos)  
end 