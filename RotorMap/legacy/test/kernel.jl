using Revise

using Base.Threads
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using RotorMap.Mapper
using RotorMap.Utils

# test of the base encoding kernel
function test() 
    @info "Loading index"   
    @time ri = load(joinpath(@__DIR__,"../indexs3.bin"))

    @info "Loading reads"   
    @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn10kl100ke10.bin"))                   
    # @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn50kl100ke10.bin"))                   

    # pos = [parse(Int, head[3:end]) for head in heads] # todo: what about descriptions?

    # @time reads = hcat(seqs...) # todo: it's more efficient to directly transfer to GPU
    @time reads = vcat(seqs...)
    # reads = seqs

    @info "Sending reads to GPU"    
    # CUDA.@time creads_t = transpose(reads) |> CuArray 
    CUDA.@time creads = reads |> CuArray 
    # CUDA.@time creads = (@view reads[:,1:5000]) |> CuArray 

    re = RopeEncoder(k=ri.re.k, s=5, m=8, c=0) 
    # re = RopeEncoder(k=ri.re.k, s=5, m=8, c=0) 
    # re = RopeEncoder(k=ri.re.k, s=7, m=4, c=0) 

    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, e=ri.cu_e)
    starts = 1:re.k:length(reads) |> collect
    stops = re.k:re.k:length(reads) |> collect
    if length(stops) < length(starts) 
        push!(stops, length(reads))
    end
    # starts = starts .|> UInt32 |> CuArray
    # stops = stops .|> UInt32 |> CuArray
    starts = starts |> CuArray
    stops = stops |> CuArray

    CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, starts = starts, stops = stops)
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads_t, e=ri.cu_e)
    # CUDA.@time cropes = encode_batch_m1_cuda(re, creads, 1:re.k, e=ri.cu_e)
    # CUDA.@time cropes, _ = encode_reads_batch_m1_cuda(re, creads, 1:re.k, e=ri.cu_e)
    @show size(cropes) 
end 
