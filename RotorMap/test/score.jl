using Revise

using Base.Threads
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using RotorMap.Mapper
using RotorMap.Utils
using RotorMap.RopeScores

# test of the base encoding kernel
function test3() 
    # @info "Loading index"   
    # @time ri = load(joinpath(@__DIR__,"../indexs3.bin"))

    # @info "Loading reads"   
    # @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn10kl100ke10.bin"))                   
    # @time seqs, heads = load_dnas(joinpath(@__DIR__,"../readsn50kl100ke10.bin"))                   

    # pos = [parse(Int, head[3:end]) for head in heads] # todo: what about descriptions?
    # @time reads = hcat(seqs...) # todo: it's more efficient to directly transfer to GPU

    # @info "Sending reads to GPU"    
    # CUDA.@time creads_t = transpose(reads) |> CuArray 
    # CUDA.@time creads = reads |> CuArray 
    # CUDA.@time creads = (@view reads[:,1:5000]) |> CuArray 

    re = RopeEncoder(k=20_000, s=5, m=4, c=0) 
    # re = RopeEncoder(k=ri.re.k, s=5, m=8, c=0) 
    # re = RopeEncoder(k=ri.re.k, s=7, m=4, c=0) 
    points(re, n=100)
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, e=ri.cu_e)
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads_t, e=ri.cu_e)
    # CUDA.@time cropes = encode_batch_m1_cuda(re, creads, 1:re.k, e=ri.cu_e)
    # CUDA.@time cropes, _ = encode_reads_batch_m1_cuda(re, creads, 1:re.k, e=ri.cu_e)
    # @show size(cropes)
end 
