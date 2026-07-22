using Pkg

Pkg.activate(".")
# Pkg.instantiate()

using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using RotorMap.Mapper
using RotorMap.Utils
# using RotorMap.SplitComplexMatrices
using LinearAlgebra
using Base.Threads
using BenchmarkTools

# generate data 
N = 10^9
k = 10^5
s = 4
dna_ref = rand(UInt8.(0:3),10^9)
re = RopeEncoder(k=k, s=s, m=1, c=64) 
ri = RopeIndexer(N=N, re=re, step=10^4)

using CUDA
# CUDA.allowscalar(true)
@info "CUDA loaded"

@info "Index construction: "
# @time ri(dna_ref) # gen index # SLOW AND INCOMPATIBLE; todo
e = re.e |> cu 
@time cindex = index_m1_cuda_optimized(ri, dna_ref; e=e)

n = 10000
err = 0.20
@info "Reads generation: "
@time reads, locs = generate_reads(dna_ref, k, n, err=err)

# ================= CUDA search ===========================

creads = reads |> cu 
# cindex = ri.index |> cu

@info "Reads encoding: "
@time cropes = encode_batch_m1_cuda(re, creads, 1:k, e=e)

@info "Search phase: "
@time cinds = search_batch_m1_cuda(cindex, cropes)

@info "Convert phase: "
@time inds = Array(cinds)

@time begin 
    # preds = [p[1] for p in Array(inds)][1,:]
    # results = preds.*ri.step.-locs .|> abs 
    results = inds.*ri.step.-locs .|> abs 
end

@show maximum(results) / (ri.step÷2), sum(results)/n / (ri.step÷2)

# ropes = zeros(ComplexF64, 4^re21.s, 1000)
# ropes_c = zeros(ComplexF64, re21.c, 1000)

# @threads for i = 1:1000 
#     rope, rope_c = re21(@view reads[:,i])
#     @views ropes[:,i] = rope
#     @views ropes_c[:,i] = rope_c
# end

# index_c = ri21.compact

# try
#     cropes = ropes_c |> cu 
#     cindex = ri21.compact |> cu 
# catch
# end

# sropes = ropes_c |> SplitComplexMatrix # |> cu
# sindex = ri21.compact |> SplitComplexMatrix # |> cu 

# try
#     csropes = sropes |> cu 
#     csindex = sindex |> cu 

#     cs16ropes = SplitComplexMatrix(Float16.(csropes.Re), Float16.(csropes.Im))
#     cs16index = SplitComplexMatrix(Float16.(csindex.Re), Float16.(csindex.Im))
# catch
# end

function test_search(ri::RopeIndexer, index, reads, locs) 
    re = ri.re 
    N = ri.N
    k = ri.k
    stp = ri.step

    n = size(reads)[2]
    results = zeros(Int, n) # how far the guessed location for each test 

    ropes = zeros(ComplexF64, 4^re.s, n)
    ropes_c = zeros(ComplexF64, re.c, n)
    
    @threads for i = 1:n 
        rope, rope_c = re(@view reads[:,i])
        @views ropes[:,i] = rope
        @views ropes_c[:,i] = rope_c
    end

    # indsb = basic_search_batch(ri, ri.compact, ropes_c).-1 # 0 based

    # indsb = basic_search_t1_batch(indext, ropes_c).-1 # 0 based
    # results = indsb.*stp.-locs .|> abs 

    # @threads 
    # @views
    for i = 1:n 
        # @views rope, rope_c = re(reads[:,i])
        # inds = basic_search(ri, ri.index, rope).-1 # 0 based

        @views rope, rope_c = ropes[:,i], ropes_c[:,i]
        inds = basic_search(ri, ri.compact, rope_c).-1 # 0 based
        
        # inds = indsb[:,i] 
    
        ret = inds.*stp.+(1-locs[i]) .|> abs |> minimum

        results[i] = ret
    end

    return maximum(results) / (ri21.step÷2), sum(results)/n / (ri21.step÷2)
end




function test_search_batch(ri::RopeIndexer, reads, locs) 
    re = ri.re 
    N = ri.N
    k = ri.k
    stp = ri.step

    n = size(reads)[2]
    results = zeros(Int, n) # how far the guessed location for each test 

    ropes = zeros(ComplexF64, 4^re.s, n)
    ropes_c = zeros(ComplexF64, re.c, n)
    
    @threads for i = 1:n 
        rope, rope_c = re(@view reads[:,i])
        @views ropes[:,i] = rope
        @views ropes_c[:,i] = rope_c
    end

    # indsb = basic_search_batch(ri, ri.compact, ropes_c).-1 # 0 based

    # indsb = basic_search_t1_batch(indext, ropes_c).-1 # 0 based
    # results = indsb.*stp.-locs .|> abs 

    # @threads 
    # @views
    # for i = 1:n 
    #     # @views rope, rope_c = re(reads[:,i])
    #     # inds = basic_search(ri, ri.index, rope).-1 # 0 based

    #     @views rope, rope_c = ropes[:,i], ropes_c[:,i]
    #     inds = basic_search(ri, ri.compact, rope_c).-1 # 0 based
        
    #     # inds = indsb[:,i] 
    
    #     ret = inds.*stp.+(1-locs[i]) .|> abs |> minimum

    #     results[i] = ret
    # end

    return maximum(results) / (ri21.step÷2), sum(results)/n / (ri21.step÷2)
end

# ret = test_search(ri21, indext, reads, locs)

# @show ret 



# try 
#     using CUDA

#     comp_cu = ri21.compact |> cu
# catch
# end

# function test_search_cuda(ri::RopeIndexer, reads, locs) 
#     re = ri.re 
#     N = ri.N
#     k = ri.k
#     stp = ri.step

#     n = size(reads)[2]
#     results = zeros(Int, n) # how far the guessed location for each test 

#     ropes = zeros(ComplexF64, 4^re.s, n)
#     ropes_c = zeros(ComplexF64, re.c, n)
    
#     @threads for i = 1:n 
#         rope, rope_c = re(@view reads[:,i])
#         @views ropes[:,i] = rope
#         @views ropes_c[:,i] = rope_c
#     end

#     # indsb = basic_search_batch(ri, ri.compact, ropes_c).-1 # 0 based

#     # @threads 
#     @views for i = 1:n 
#         # @views rope, rope_c = re(reads[:,i])
#         # inds = basic_search(ri, ri.index, rope).-1 # 0 based

#         @views rope, rope_c = ropes[:,i], ropes_c[:,i]
#         # inds = basic_search(ri, ri.compact, rope_c).-1 # 0 based
#         inds = basic_search_cuda(ri, comp_cu, rope_c |> cu).-1 # 0 based
        
#         # inds = indsb[:,i] 
    
#         ret = inds.*stp.+(1-locs[i]) .|> abs |> minimum

#         results[i] = ret
#     end

#     return maximum(results) / (ri21.step÷2), sum(results)/n / (ri21.step÷2)
# end