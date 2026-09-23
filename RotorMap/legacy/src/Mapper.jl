# __precompile__(true)

module Mapper

export search_batch_m1_cuda, search_batch_m1_cuda_split
export search_batch_m1_cuda_split_all, process_carts
export sliding_search_cuda
export search_cuda_split_max, search_main
export search_m1_cuda_split_all_batch
export search_carts
export search_batch_max
export search_batch_topk

try
    using CUDA
catch 
end
# CUDA.functional()

# using StaticArrays
using Base.Cartesian
using Base.Threads
using LinearAlgebra
using ProgressMeter
using ..RopeEncoders
using ..RopeIndexers
using ..SplitComplexMatrices

# =============== SEARCH PHASE =================

function search_main(ri::RopeIndexer, re2::RopeEncoder, creads::CuArray, starts::CuArray, stops::CuArray; pos)    
    # TODO use jagged,  find_ind, ind_starts

    re = ri.re

    @info "Encoding reads"
    # CUDA.@time cropes, _ = encode_batch_cuda_best(re, creads, e=ri.cu_e)
    # CUDA.@time creads_t = CuArray(transpose(creads))
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, e=ri.cu_e)
    # CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads_t, e=ri.cu_e)
    CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, creads, starts=starts, stops=stops)
    @show size(cropes)

    @show size(ri.cu_index)
    # throw("yo1")

    CUDA.@time begin 
        @info "Splitting complex matrices and converting to FP16"
        # cindex_split = SplitComplexMatrix(ComplexF32.(ri.cu_index))
        # cropes_split = SplitComplexMatrix(ComplexF32.(cropes))
        # cindex_split = SplitComplexMatrix(ComplexF16.(ri.cu_index))
        # cropes_split = SplitComplexMatrix(ComplexF16.(cropes))
        cindex_split = reshape(ri.cu_index, :, ri.n)
        cropes_split = reshape(cropes, :, length(starts))

        cindex_split = SplitComplexMatrix(ComplexF16.(cindex_split))
        cropes_split = SplitComplexMatrix(ComplexF16.(cropes_split))
    end 

    @info "Mapping reads, phase 1"    

    # CUDA.@time 
    # CUDA.@time 
    # fid = CUDA.CuArray{Float32}(undef, size(ri.cu_index, 2), size(cropes, 2))
    # CUDA.math_mode!(CUDA.FAST_MATH) # improves the product of FP32 complex matrices, still not faster than split fp16
    # CUDA.@time cinds = search_cuda_max(ri.cu_index, cropes) 
    
    # CUDA.@time cinds = search_cuda_split_max(cindex_split, cropes_split) 
    
    # CUDA.@time carts = search_batch_m1_cuda_split_all(cindex_split, cropes_split, cut = ri.cut)
    # @show size(carts)
    # ranges = process_carts_basic(carts; N=ri.N, k=re.k, n=size(cropes, 2), stp = ri.step)
    # @show size(ranges)
    # lr = length.(ranges) |> sort!
    # @show lr[1:10]
    # @show lr[end-9:end]
    # @show size(cinds)
    # throw("yo1")

    # CUDA.@time carts = search_m1_cuda_split_all_batch(cindex_split, cropes_split, cut = ri.cut)
    # CUDA.@time carts = search_m1_cuda_split_all_batch(ri::RopeIndexer, cindex_split, cropes_split, cut = ri.cut, batch_prod_size=2^12)
    # CUDA.@time carts = search_m1_cuda_split_all_batch(ri::RopeIndexer, cindex_split, cropes_split, cut = ri.cut, batch_prod_size=size(cindex_split.Re,2))
    # CUDA.@time carts = search_m1_cuda_split_all_batch(cindex_split, cropes_split, cut = ri.cut, batch_prod_size=2^4)
    # CUDA.@time carts = search_cuda_all_kernel(ri.cu_index, cropes, cut = ri.cut)
    # CUDA.@time carts = search_cuda_all_gemini(ri.cu_index, cropes, cut = ri.cut)
    # CUDA.@time carts = search_cuda_2d_tiled(ri.cu_index, cropes, cut = ri.cut)        
    @show ri.cut   
    # CUDA.@time ranges = search_m1_cuda_split_all_batch(ri::RopeIndexer, cindex_split, cropes_split, cut = 0.02463403f0, batch_prod_size=2^12)
         
    CUDA.@time ranges = search_m1_cuda_split_all_batch(re::RopeEncoder, stp, cindex_split, cropes_split, find_ind, ind_starts, cut = ri.cut, batch_prod_size=2^12)
    # CUDA.@time ranges = search_m1_cuda_split_all_batch(ri::RopeIndexer, cindex_split, cropes_split, cut = ri.cut, batch_prod_size=size(cindex_split.Re,2))
    # CUDA.@time ranges = search_m1_cuda_split_all_batch(ri::RopeIndexer, cindex_split, cropes_split, cut = ri.cut*0.9, batch_prod_size=2^12)
    @show size(ranges)
    # @show ranges[1:10]
    # @show ranges[end-9:end]
    lr = length.(ranges) |> sort!
    @show lr[1:10]
    @show lr[end-9:end]
    @show pos[1:10]
    @show sum(lr)
    # throw("yo3")
    # @show pos[1:10]
    # @info "Getting found locations from GPU"
    # @time inds = Array(cinds)
    # @show ranges[1][1]
    # preds = [sum(x[1])÷2 for x in ranges] 
    # preds = map(x->[sum(y)÷2 for y in x], ranges) # considers 1st found range for each read

    @info "Comparing with true locations"
    # inds = cinds |> Array
    # preds = inds.*ri.step
    avgdiff = 0.0
    maxdiff = 0.0
    found_n = 0
    n = length(pos)
    @show n 
    good_locs = [UInt32[] for i=1:n] 
    for i = 1:n
        p = pos[i]
        found = false
        for r in ranges[i]
            diff = (sum(r)÷2 - p) |> abs
            if diff <= re.k
                found = true                
                found_n += 1
                push!(good_locs[i], sum(r)÷2)
                avgdiff += diff 
                maxdiff = max(maxdiff, diff)
            end
        end
        # if !found
        #     missed += 1
        # end
    end
    avgdiff /= found_n
    @show found_n


    # @show good_locs[1:10]
    # @views diffs = preds.-pos[1:length(preds)] .|> abs 

    # maxdiff = maximum(diffs) 
    # avgdiff = sum(diffs)/length(diffs)

    mdp = maxdiff/re.k*100
    adp = avgdiff/re.k*100
    
    "Mapped precision in % of the read length (0.0 means exact match): " |> println
    "Maximum: $mdp%" |> println
    "Average: $adp%" |> println
    "(Step size is $(ri.step*100/re.k)%)" |> println

    # throw("skipping phase 2")

    mid_locs = [UInt32[] for i=1:n] 
    for i = 1:n
        for r in ranges[i]
            push!(mid_locs[i], sum(r)÷2)
        end
    end

    @info "Mapping reads, phase 2"
    # CUDA.@time locs = sliding_search_cuda(ri, re2, creads, re.k÷3, pos=preds)
    # CUDA.@time locs = sliding_search_cuda_ranges(ri, re2, creads, re.k÷2, found_ranges=ranges)
    @show mid_locs[1:10] 
    CUDA.@time locs, vals = sliding_search_cuda_locs(ri, re2, creads, starts, stops, mid_locs, ext_len = re2.k)
    @show vals[1:10] 
    @show locs[1:10] .|> Int32
    @show locs[end-10:end] .|> Int32

    # throw("check")
    # @views diffs = locs.-pos[1:length(preds)] .|> abs 

    # maxdiff = maximum(diffs) 
    # avgdiff = sum(diffs)/length(diffs)

    @show length(locs)
    @show sum(length.(mid_locs))

    avgdiff = 0.0
    maxdiff = 0.0
    tot = 0
    howm = 0
    for i = 1:n
        p = pos[i]
        best_match_val = typemax(UInt32)
        best_match_ind = 0
        for l in 1:length(mid_locs[i])                        
            diff = (locs[l+tot] - p) |> abs
            if diff < best_match_val
                best_match_ind = locs[l+tot]
                best_match_val = diff
            end
        end        

        if best_match_val > 100000 && howm < 10
            howm +=1
            @show best_match_val 
            @show p                      
            @show tot 
            # @show length(mid_locs[i])            
            @show mid_locs[i] .|> Int32
            @show locs[(1:length(mid_locs[i])).+tot] .|> Int32              
            @show vals[(1:length(mid_locs[i])).+tot] 
        end

        avgdiff += best_match_val 
        maxdiff = max(maxdiff, best_match_val)

        tot += length(mid_locs[i])
        # if !found
        #     missed += 1
        # end
    end
    avgdiff /= n
    @show avgdiff

    mdp = maxdiff/re.k*100
    adp = avgdiff/re.k*100
    
    "Mapped precision in % of the read length (Phase 2): " |> println
    "Maximum: $mdp%" |> println
    "Average: $adp%" |> println
    "(Step size is $(ri.step*100/re.k)%)" |> println

    # return locs
    return 1
end

function search_cuda_split_max(index::SplitComplexMatrix, ropes::SplitComplexMatrix)
    # CUDA.@time ips = CUDA.zeros(eltype(index), size(index, 2), size(ropes, 2))
    # CUDA.@time mul!(ips, index', ropes)
    # CUDA.@time 
    ips = index'*ropes
    begin
        A, B = ips.Re, ips.Im
        # CUDA.@time 
        fid = A.^2
        # CUDA.@time A .*= A
        # CUDA.@time B .*= B
        # CUDA.@time 
        fid .+= B.^2
        
    end 
    # CUDA.@time 
    vals, carts = findmax(fid, dims=1)
    # @show vals[1:100]
    # CUDA.@time 
    cret = map(x -> x[1], carts) |> vec 
    
    cret .-= 1 # make them 0-based
    return cret
end 

function search_cuda_max(index::CuArray, ropes::CuArray)#, fid::CuArray)
    ni = size(index)[2]
    n = size(ropes)[2]

    # @show T 
    # CUDA.@time ips = CUDA.zeros(eltype(index), size(index, 2), size(ropes, 2))
    # CUDA.@time mul!(ips, index', ropes)
    # CUDA.@time 
    ips = index'*ropes
    # @time begin
    #     A, B = reim(ips)
    #     A .*= A
    #     B .*= B
    #     A .+= B
    #     fid = A
    # end 
    # @time 
    begin
        # fid = CUDA.CuArray{T}(undef, ni, n)
        # fid .= abs2.(ips)
        # CUDA.@time 
        fid = abs.(ips)
    end 

    # fid = pairwise(Euclidean(), index, ropes, dims=2)
    # sq_norm_A = vec(sum(abs2.(index), dims=1)) 
    # sq_norm_B = vec(sum(abs2.(ropes), dims=1)) 
    # fid = sq_norm_A .+ sq_norm_B' .- 2 .* real(ips)
    # ret = findmin(fid, dims=1)

    # inds = Array{Int}(undef, n)

    # CUDA.@allowscalar begin 
    #     ret = getindex.(findmax(fid, dims=1)[2], 1)
    # end

    # CUDA.@time 
    vals, carts = findmax(fid, dims=1)
    # @time 
    # ret = getindex.(carts, 1)
    # CUDA.@time 
    cret = map(x -> x[1], carts) |> vec 
    # @time ret = cret |> Array

    # ret = findmax_linear_dims1(fid)
    # ret = argmax_along_dims_1(fid)
    
    cret .-= 1 # make them 0-based
    return cret
end 

function search_batch_m1_cuda(index::CuArray{Complex{T}}, ropes::CuArray{Complex{T}}) where T 
    ni = size(index)[2]
    n = size(ropes)[2]

    # @show T 

    # CUDA.@time 
    ips = index'*ropes
    # @time begin
    #     A, B = reim(ips)
    #     A .*= A
    #     B .*= B
    #     A .+= B
    #     fid = A
    # end 
    # @time 
    begin
        # fid = CUDA.CuArray{T}(undef, ni, n)
        # fid .= abs2.(ips)
        # CUDA.@time 
        fid = abs2.(ips)
    end 

    # fid = pairwise(Euclidean(), index, ropes, dims=2)
    # sq_norm_A = vec(sum(abs2.(index), dims=1)) 
    # sq_norm_B = vec(sum(abs2.(ropes), dims=1)) 
    # fid = sq_norm_A .+ sq_norm_B' .- 2 .* real(ips)
    # ret = findmin(fid, dims=1)

    # inds = Array{Int}(undef, n)

    # CUDA.@allowscalar begin 
    #     ret = getindex.(findmax(fid, dims=1)[2], 1)
    # end

    # CUDA.@time 
    vals, carts = findmax(fid, dims=1)
    # @time 
    # ret = getindex.(carts, 1)
    # CUDA.@time 
    cret = map(x -> x[1], carts) |> vec 
    # @time ret = cret |> Array

    # ret = findmax_linear_dims1(fid)
    # ret = argmax_along_dims_1(fid)
    
    cret .-= 1 # make them 0-based
    return cret
end 

function search_batch_m1_cuda_split(index::SplitComplexMatrix, ropes::SplitComplexMatrix)
    ni = size(index)[2]
    n = size(ropes)[2]

    # @show size(index)
    # @show size(ropes)


    # CUDA.@time 
    ips = index'*ropes
    # CUDA.@time 
    begin
        A, B = ips.Re, ips.Im
        A .*= A
        B .*= B
        A .+= B
        fid = A
    end 
    # begin
    #     CUDA.@time fid = abs2.(ips)
    # end 

    # fid = pairwise(Euclidean(), index, ropes, dims=2)
    # sq_norm_A = vec(sum(abs2.(index), dims=1)) 
    # sq_norm_B = vec(sum(abs2.(ropes), dims=1)) 
    # fid = sq_norm_A .+ sq_norm_B' .- 2 .* real(ips)
    # ret = findmin(fid, dims=1)

    # inds = Array{Int}(undef, n)

    # CUDA.@allowscalar begin 
    #     ret = getindex.(findmax(fid, dims=1)[2], 1)
    # end
    
    # CUDA.@time 
    vals, carts = findmax(fid, dims=1)
    # @show vals[1:100]
    # CUDA.@time 
    cret = map(x -> x[1], carts) |> vec 

    # CUDA.@time mask = fid .> cut
    # CUDA.@time carts = CUDA.findall(mask)

    # ret = getindex.(carts, 1)
    # @time ret = cret |> Array

    # ret = findmax_linear_dims1(fid)
    # ret = argmax_along_dims_1(fid)
    
    cret .-= 1 # make them 0-based
    return cret
end 


function search_cuda_all_kernel(index::CuArray, ropes::CuArray; cut)
    # ips = index'*ropes  # Dense complex matrix multiplication
    # fid = abs2.(ips)    # Element-wise squared magnitude
    # mask = fid .> cut    # Element-wise comparison
    
    # The three lines above can be combined into one more efficient kernel
    d, ni = size(index)
    _, n = size(ropes)
    mask = CUDA.zeros(Bool, ni, n)

    # Define the kernel
    function compute_mask_kernel(index, ropes, mask, cut::R) where {R<:Real}
        # Thread and block indices
        i = (blockIdx().y - 1) * blockDim().y + threadIdx().y # Column for index
        j = (blockIdx().x - 1) * blockDim().x + threadIdx().x # Column for ropes
        d_common = size(index, 1)

        # Check boundaries
        if i > size(index, 2) || j > size(ropes, 2)
            return
        end

        # Compute the dot product
        # dot_prod = sum(conj(index[:, i]) .* ropes[:, j])
        # Let's unroll the loop manually for clarity and potential optimization
        my_dot = zero(ComplexF32)
        for k in 1:d_common
            my_dot += conj(index[k, i]) * ropes[k, j]
        end
        
        # Compare squared magnitude to cut and write to mask
        mask[i, j] = abs2(my_dot) > cut
        return
    end

    # Configure and launch the kernel
    # (n, ni) for the grid dimensions, not (ni, n), to match j, i mapping
    # threads = (16, 16) 
    # threads = (1, 32) 
    # threads = (32, 32)
    threads = (8, 8)  
    blocks = ceil.(Int, (n, ni) ./ threads)
    
    CUDA.@sync begin
        CUDA.@cuda blocks=blocks threads=threads compute_mask_kernel(index, ropes, mask, cut)
    end

    # The rest of the logic remains the same, using the highly optimized CUDA.jl functions
    CUDA.@time carts = CUDA.findall(mask)
    if length(carts) > 0
        CUDA.@time CUDA.sort!(carts)
    end
    return carts
end

function search_cuda_all(index::CuArray, ropes::CuArray; cut)
    ni = size(index, 2)
    n = size(ropes, 2)

    CUDA.@time ips = index'*ropes
    fid = abs2.(ips)

    CUDA.@time mask = fid .> cut
    CUDA.@time carts = CUDA.findall(mask)
    if length(carts) > 0
        CUDA.@time CUDA.sort!(carts)
    end

    return carts
end 

# not memory efficient
function search_batch_m1_cuda_split_all(index::SplitComplexMatrix, ropes::SplitComplexMatrix; cut)
    ni = size(index)[2]
    n = size(ropes)[2]

    # @show T 

    CUDA.@time ips = index'*ropes
    CUDA.@time begin
        A, B = ips.Re, ips.Im
        A .*= A
        B .*= B
        A .+= B
        fid = A
    end 
    # begin
    #     CUDA.@time fid = abs2.(ips)
    # end 

    # fid = pairwise(Euclidean(), index, ropes, dims=2)
    # sq_norm_A = vec(sum(abs2.(index), dims=1)) 
    # sq_norm_B = vec(sum(abs2.(ropes), dims=1)) 
    # fid = sq_norm_A .+ sq_norm_B' .- 2 .* real(ips)
    # ret = findmin(fid, dims=1)

    # inds = Array{Int}(undef, n)

    # CUDA.@allowscalar begin 
    #     ret = getindex.(findmax(fid, dims=1)[2], 1)
    # end
    
    # CUDA.@time vals, carts = findmax(fid, dims=1)
    # CUDA.@time cret = map(x -> x[1], carts) |> vec 


    # CUDA.@time for i in 1:n
    #     # Select the column view. `@view` is efficient and the result is still a CuArray.
    #     col = @view fid[:, i]

    #     # 1. Perform the comparison on the GPU.
    #     #    This creates a CuArray of booleans without transferring data.
    #     mask = col .> cut

    #     # 2. Find the indices where the mask is true.
    #     #    CUDA.findall is a GPU-accelerated version of Base.findall.
    #     #    It returns a CuArray of integers.
    #     col_inds = CUDA.findall(mask) 

    #     # 3. Move the result from GPU memory to CPU memory.
    #     #    `Array(x)` is the standard way to transfer a CuArray `x` to a regular `Array`.
    #     # result_indices[col] = Array(gpu_indices)
    # end

    CUDA.@time mask = fid .> cut
    CUDA.@time carts = CUDA.findall(mask)
    if length(carts) > 0
        CUDA.@time CUDA.sort!(carts)
    end
    # CUDA.@time carts = CUDA.findall(x -> x>cut, fid)

    # @show length(carts)
    # ret = getindex.(carts, 1)
    # @time ret = cret |> Array

    # ret = findmax_linear_dims1(fid)
    # ret = argmax_along_dims_1(fid)
    
    # cret .-= 1 # make them 0-based
    # return cret

    # @show carts[1:100]

    return carts |> Array
    # return 1
end 

# memory efficient   
"""
The main search routine. Compares encodings in the index with given encodings (ropes). 
Does this in batches to reduce memory requirement. 

Returns found_ranges - the list of lists of pairs (also represented as 2-element lists). 
The top list has the same number of elements as the number of given encodings. 
"""
function search_m1_cuda_split_all_batch(re::RopeEncoder, stp, index::SplitComplexMatrix, ropes::SplitComplexMatrix, find_ind, ind_starts; cuts, batch_prod_size=2^12, euc=false)
    ni = size(index.Re, 2)
    n = size(ropes.Re, 2)
    # @show ni, n
    @show size(index.Re)

    CRe = CUDA.zeros(Float16, batch_prod_size, n)
    CIm = CUDA.zeros(Float16, batch_prod_size, n)
    mask = CUDA.zeros(Bool, batch_prod_size, n)

    # carts_all_cpu = []
    # found_ranges = [UnitRange{Int64}[] for i=1:n] 
    # found_ranges = [Vector{Int64}[] for i=1:n] 
    found_locs = [Tuple{Int64,Int64}[] for i=1:n] 

    # bi = 0 
    for i = 1:batch_prod_size:ni-batch_prod_size        
        # @show i
        # ARe = @view index.Re[i:i+batch_prod_size-1]
        # AIm = @view index.Im[i:i+batch_prod_size-1] # *-1
        # ARe = (@view index.Re[:, i:min(ni, i+batch_prod_size-1)]) |> transpose
        # AIm = (@view index.Im[:, i:min(ni, i+batch_prod_size-1)]) |> transpose # *-1

        ips = SplitComplexMatrix((@view index.Re[:, i:i+batch_prod_size-1]), (@view index.Im[:, i:i+batch_prod_size-1]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        # ARe = (@view index.Re[:, i:i+batch_prod_size-1]) |> transpose
        # AIm = (@view index.Im[:, i:i+batch_prod_size-1]) |> transpose # *-1

        # BRe = ropes.Re
        # BIm = ropes.Im 

        # mul!(CRe, ARe, BRe)
        # mul!(CRe, AIm, BIm)

        # mul!(CIm, ARe, BIm)
        # mul!(CIm, AIm, BRe, -1, 1)

        if euc==false
            CIm .*= CIm
            CRe .*= CRe
            CRe .+= CIm
            fid = CRe
        else 
            fid = CRe
        end

        # mask .= fid .> cut
        @views mask .= fid .> cuts[i:i+batch_prod_size-1]

        carts = CUDA.findall(mask) 
        # if length(carts) > 0
        #     CUDA.sort!(carts)
        # end

        # append!(carts_all_cpu, Array(carts))
        # @show i, bi*batch_prod_size
        # process_carts!(found_ranges, ri, Array(carts), bi*batch_prod_size)
        # process_carts!(found_ranges, re, stp, Array(carts), i-1, find_ind, ind_starts)
        process_carts_simple!(found_locs, re, stp, Array(carts), i-1, find_ind, ind_starts)
        # bi += 1
        # append!(found_ranges, found_ranges_local)
    end

    last_i = last(1:batch_prod_size:ni-batch_prod_size) 

    mask = CUDA.zeros(Bool, ni-last_i+1, n)
    if last_i <= ni
        i = last_i
        # @show i:ni
        # ARe = @view index.Re[i:i+batch_prod_size-1]
        # AIm = @view index.Im[i:i+batch_prod_size-1] # *-1
        # ARe = (@view index.Re[:, i:min(ni, i+batch_prod_size-1)]) |> transpose
        # AIm = (@view index.Im[:, i:min(ni, i+batch_prod_size-1)]) |> transpose # *-1
        # ARe = (@view index.Re[:, i:ni]) |> transpose
        # AIm = (@view index.Im[:, i:ni]) |> transpose # *-1
        # ARe = index.Re |> transpose
        # AIm = index.Im |> transpose # *-1
        ips = SplitComplexMatrix((@view index.Re[:, i:ni]), (@view index.Im[:, i:ni]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        # BRe = ropes.Re
        # BIm = ropes.Im 

        # CRe = ARe * BRe
        # # CUDA.synchronize()
        # # mul!(CRe, ARe, BRe)
        # mul!(CRe, AIm, BIm)
        # # CUDA.synchronize()

        # CIm = ARe * BIm
        # # CUDA.synchronize()
        # # mul!(CIm, ARe, BIm)
        # mul!(CIm, AIm, BRe, -1, 1)
        # # CUDA.synchronize()

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe
        # CUDA.synchronize()

        # mask = fid .> cut
        @views mask .= fid .> cuts[i:ni]

        carts = CUDA.findall(mask) # todo last batch
        # if length(carts) > 0
        #     CUDA.sort!(carts)
        # end

        # @show (bi-1)*batch_prod_size
        # append!(carts_all_cpu, Array(carts))
        # process_carts!(found_ranges, ri, Array(carts), (bi-1)*batch_prod_size)
        # process_carts!(found_ranges, re, stp, Array(carts), i-1, find_ind, ind_starts)
        process_carts_simple!(found_locs, re, stp, Array(carts), i-1, find_ind, ind_starts)
    end

    # return carts_all_cpu |> sort!
    found_ranges = find_ranges(found_locs, re.k)
    return found_ranges, found_locs
end 

function search_carts(re::RopeEncoder, stp, index::SplitComplexMatrix, ropes::SplitComplexMatrix, find_ind, ind_starts; cuts, batch_prod_size=2^12, euc=false)
    ni = size(index.Re, 2)
    n = size(ropes.Re, 2)
    # @show ni, n
    @show size(index.Re)

    CRe = CUDA.zeros(Float16, batch_prod_size, n)
    CIm = CUDA.zeros(Float16, batch_prod_size, n)
    mask = CUDA.zeros(Bool, batch_prod_size, n)

    carts_all = cu(CartesianIndex{2}[])
    # bi = 0 
    for i = 1:batch_prod_size:ni-batch_prod_size        
        # @show i
        # ARe = @view index.Re[i:i+batch_prod_size-1]
        # AIm = @view index.Im[i:i+batch_prod_size-1] # *-1
        # ARe = (@view index.Re[:, i:min(ni, i+batch_prod_size-1)]) |> transpose
        # AIm = (@view index.Im[:, i:min(ni, i+batch_prod_size-1)]) |> transpose # *-1

        ips = SplitComplexMatrix((@view index.Re[:, i:i+batch_prod_size-1]), (@view index.Im[:, i:i+batch_prod_size-1]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        if euc==false
            CIm .*= CIm
            CRe .*= CRe
            CRe .+= CIm
            fid = CRe
        else 
            fid = CRe
        end

        # mask .= fid .> cut
        @views mask .= fid .> cuts[i:i+batch_prod_size-1]

        carts = CUDA.findall(mask) 
        carts_all = vcat(carts_all, carts)
    end

    last_i = last(1:batch_prod_size:ni-batch_prod_size) 

    mask = CUDA.zeros(Bool, ni-last_i+1, n)
    if last_i <= ni
        i = last_i

        ips = SplitComplexMatrix((@view index.Re[:, i:ni]), (@view index.Im[:, i:ni]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe
        # CUDA.synchronize()

        # mask = fid .> cut
        @views mask .= fid .> cuts[i:ni]

        carts = CUDA.findall(mask) 
        carts_all = vcat(carts_all, carts)
    end

    return carts_all
end 

"""
Process all found_locs to generate found_ranges. 
"""
function find_ranges(found_locs::Vector{Vector{Tuple{Int64,Int64}}}, k)::Vector{Vector{Tuple{Int64, Vector{Int64}}}}
    n = length(found_locs)
    found_ranges = [Tuple{Int64, Vector{Int64}}[] for i=1:n] 
    
    for i = 1:n         
        ni = length(found_locs[i])
        piece = 0
        loc = 0
        for j = 1:ni
            tup = found_locs[i][j]        
            cpiece, cloc = tup
            if cpiece > piece # new piece
                ranges = (cpiece, [cloc, cloc])
                push!(found_ranges[i], ranges)
            else # same piece
                if cloc - loc < k # same region 
                    found_ranges[i][end][2][2] = cloc
                else # new region
                    ranges = (cpiece, [cloc, cloc])
                    push!(found_ranges[i], ranges)                
                end
            end 
            piece = cpiece
            loc = cloc 
        end        
    end
     
    return found_ranges
end
# test: 
# found_locs = [[(1,10),(1,15)], [(3,10), (4,10)], [(1,10), (1,20)]]; k=6
# found_ranges = [[(1,[10,15])], [(3,[10,10]), (4,[10,10])], [(1,[10,10]), (1,[20,20])]]

"""
Process sorted Cartesian Indices found by main search phase. 

The main problem: found indices are relative withing a batch, 
however we should output a location relative to the dna piece in the index. 

The simple version doesn't try to combine found locations into ranges. 
"""
function process_carts_simple!(found_locs, re, stp, carts, offset, find_ind, ind_starts) 
    for c in carts
        loc, i = c.I
        global_loc = loc+offset # global location in the entire index
        piece = find_ind[global_loc] # index of the piece 
        piece_loc = (global_loc - ind_starts[piece])*stp+1 # exact location within a dna piece
        push!(found_locs[i], (piece, piece_loc))
    end

    return 1
end

function process_carts!(found_ranges, re, stp, carts, offset, find_ind, ind_starts) # there is a bug somewhere
    # if  1227238 - 2^12 < offset < 1227238
    # if offset==1224704
    #     @show offset
    #     @show carts
    # end

    k = re.k 
    # stp = ri.step

    # last_piece = find_ind[global_loc]
    for c in carts
        loc, i = c.I
        # new_loc = (loc-1+offset)*stp+1 # TODO!!! there is a bug since we use jagged array
        global_loc = loc+offset # global location in the entire index
        new_loc = (global_loc-ind_starts[find_ind[global_loc]])*stp+1 # location within a dna piece, it could be a new piece!
        # if new_loc > 10^9 
        #     @show c 
        # end
        if length(found_ranges[i]) > 0 # check if this is a range extension
            # @show found_ranges[i]
            prev_end = found_ranges[i][end][2]
            if new_loc-prev_end < k # same region # todo: split if too long 
                found_ranges[i][end][2] = new_loc
            else
                push!(found_ranges[i], [new_loc, new_loc])
            end
        else # new region            
            push!(found_ranges[i], [new_loc, new_loc])
        end
    end

    return 1
end

function process_carts_basic(carts; n, N, k, stp)        
    ret = [UnitRange{Int64}[] for i=1:n]
    cur_i = 0
    cur_loc = 0
    loc_start = 0
    for c in carts
        loc, i = c.I
        if i > cur_i 
            # finish previous if i > 1           
            if i > 1 
                loc_end = cur_loc * stp
                loc_end = min(loc_end, N-k)
                push!(ret[i-1], loc_start+1:loc_end+k)
            end 

            loc_start = (loc-2)*stp # starting location of the match 
            loc_start = max(loc_start, 0)
            cur_loc = loc
            cur_i = i
        else # i == cur_i
            if loc - cur_loc < 10 # same region
                cur_loc = loc
            else # new region 
                # finish previous 
                loc_end = cur_loc * stp
                loc_end = min(loc_end, N-k)
                push!(ret[i], loc_start+1:loc_end+k)

                # start new 
                loc_start = (loc-2)*stp # starting location of the match 
                loc_start = max(loc_start, 0)
                cur_loc = loc
            end 
        end
    end

    if cur_i > 0
        loc_end = cur_loc * stp
        loc_end = min(loc_end, N-k)
        push!(ret[cur_i], loc_start+1:loc_end+k)
    end

    return ret
end

function search_batch_max(re::RopeEncoder, stp, index::SplitComplexMatrix, ropes::SplitComplexMatrix, find_piece, first_kmer; batch_prod_size=2^12)
    ni = size(index.Re, 2)
    n = size(ropes.Re, 2)

    # CRe = CUDA.zeros(Float16, batch_prod_size, n)
    # CIm = CUDA.zeros(Float16, batch_prod_size, n)

    vals = CUDA.zeros(Float16, n)
    locs = CUDA.zeros(Int32, n)
    mask = CUDA.zeros(Bool, n)

    @showprogress for i = 1:batch_prod_size:ni-batch_prod_size        
        ips = SplitComplexMatrix((@view index.Re[:, i:i+batch_prod_size-1]), (@view index.Im[:, i:i+batch_prod_size-1]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe

        vals_i, carts = findmax(fid, dims=1)
        vals_i = vals_i |> vec
        locs_i = map(x -> x[1]+i-1, carts) |> vec
        mask .= vals_i .> vals
        locs .= ifelse.(mask, locs_i, locs)
        vals .= ifelse.(mask, vals_i, vals)
    end

    last_i = last(1:batch_prod_size:ni) 
    if last_i <= ni
        i = last_i

        ips = SplitComplexMatrix((@view index.Re[:, i:ni]), (@view index.Im[:, i:ni]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe
        
        vals_i, carts = findmax(fid, dims=1)
        vals_i = vals_i |> vec
        locs_i = map(x -> x[1]+i-1, carts) |> vec
        mask .= vals_i .> vals
        locs .= ifelse.(mask, locs_i, locs)
        vals .= ifelse.(mask, vals_i, vals)
    end

    found_locs = map(x -> (find_piece[x], 1+(x-first_kmer[find_piece[x]])*stp), Array{Int64}(locs))

    return found_locs, vals
end 

function search_batch_topk(index::SplitComplexMatrix, ropes::SplitComplexMatrix; k=10, batch_prod_size=2^12)
    ni = size(index.Re, 2)
    n = size(ropes.Re, 2)

    # vals = CUDA.zeros(Float16, n)
    # locs = CUDA.zeros(Int32, n)
    # mask = CUDA.zeros(Bool, n)

    CRe = CUDA.zeros(Float16, batch_prod_size, n)
    CIm = CUDA.zeros(Float16, batch_prod_size, n)
    C = SplitComplexMatrix(CRe, CIm)

    vals_topk = CUDA.zeros(Float16, (n, k))
    locs_topk = CUDA.zeros(Int32, (n, k))

    vals_topk_i = CUDA.zeros(Float16, (n, k))
    locs_topk_i = CUDA.zeros(Int32, (n, k))

    vals_topk_m = CUDA.zeros(Float16, (n, k))
    locs_topk_m = CUDA.zeros(Int32, (n, k))

    threads = 256
    blocks = cld(n, threads)    

    # @showprogress 
    @inbounds for i = 1:batch_prod_size:ni-batch_prod_size        
        # ips = SplitComplexMatrix((@view index.Re[:, i:i+batch_prod_size-1]), (@view index.Im[:, i:i+batch_prod_size-1]))'*ropes        
        # CRe = ips.Re 
        # CIm = ips.Im
        C = mul!(C, SplitComplexMatrix((@view index.Re[:, i:i+batch_prod_size-1]), (@view index.Im[:, i:i+batch_prod_size-1]))', ropes)
        CRe = C.Re 
        CIm = C.Im

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe

        # @show i
        @cuda threads=threads blocks=blocks topk_kernel!(vals_topk_i, locs_topk_i, fid', k)
        # @show i, locs_topk_i[8193, :]
        @cuda threads=threads blocks=blocks merge_topk_kernel!(vals_topk_m, locs_topk_m, vals_topk, locs_topk, vals_topk_i, locs_topk_i, i-1)
        vals_topk .= vals_topk_m
        locs_topk .= locs_topk_m
    end

    last_i = last(1:batch_prod_size:ni) 
    if last_i <= ni
        i = last_i

        ips = SplitComplexMatrix((@view index.Re[:, i:ni]), (@view index.Im[:, i:ni]))'*ropes
        CRe = ips.Re 
        CIm = ips.Im

        # C = mul!(C, SplitComplexMatrix((@view index.Re[:, i:ni]), (@view index.Im[:, i:ni]))', ropes)
        # CRe = C.Re 
        # CIm = C.Im

        CIm .*= CIm
        CRe .*= CRe
        CRe .+= CIm
        fid = CRe
        # @show i
        @cuda threads=threads blocks=blocks topk_kernel!(vals_topk_i, locs_topk_i, fid', k)
        # @show i, locs_topk_i[8193, :]
        @cuda threads=threads blocks=blocks merge_topk_kernel!(vals_topk_m, locs_topk_m, vals_topk, locs_topk, vals_topk_i, locs_topk_i, i-1)
        vals_topk .= vals_topk_m
        locs_topk .= locs_topk_m
    end
    
    # found_locs = map(x -> (find_piece[x], 1+(x-first_kmer[find_piece[x]])*stp), Array{Int64}(locs_topk))

    return Array(locs_topk), vals_topk
end 

# ======================= SLIDING SEARCH ================================================

@generated function kernel_super_v2(s, ::Val{M}, k, dnas, reads_encodings, ips, vals, locs, e) where M    
    # function kernel(s, k, dna_ref, reads_encodings, ips, locs, ws, we)
    quote
        n = reads_encodings |> length
        N = dnas[1] |> length

        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x
        if idy > n
            return
        end

        # ips_sums = @cuDynamicSharedMem(Float32, blockDim().x)  
        ips_sums = @cuDynamicSharedMem(Float32, (blockDim().x, blockDim().y))  
        ip_new = ips[idm, idy]

        # # Each thread writes its partial sum to shared memory.
        # # Indexing is 1-based for Julia arrays, including shared memory.
        # sdata[threadIdx().x + 1] = thread_sum
        
        # # Ensure all threads have finished writing to shared memory before proceeding.
        # sync_threads()
        
        # ips_sums[idm] = abs2(ips[idm, idy])
        ips_sums[idm, threadIdx().y] = abs2(ips[idm, idy])
        sync_threads()

        # # Perform the reduction. The number of active threads is halved in each step.
        # # This requires the number of threads per block to be a power of 2.
        stride = blockDim().x ÷ 2
        while stride > 0
            # Only the first `stride` threads participate in this reduction step.
            if idm <= stride
                ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
            end
            # Wait for all threads in the block to complete this step.
            sync_threads()
            stride ÷= 2
        end

        val = ips_sums[1, threadIdx().y]        

        smer_start = UInt32(0)
        smer_end = UInt32(0)
        for i = 1 : s
            smer_start <<= 2
            smer_start += dnas[idy][i]

            smer_end <<= 2
            smer_end += dnas[idy][i-s+k]            
        end

        bmask = UInt32(4^s - 1)

        ws_idm = e[idm + 1]
        we_idm = e[(k-s)*idm % k + 1]

        loc = 1
        for i = s+1 : N-k+s 
            smer_end <<= 2
            smer_end &= bmask
            smer_end += dnas[idy][i-s+k]            

            # ip_new = (ip_new - reads_encodings[idy][idm, smer_start+1])*ws_idm + reads_encodings[idy][idm, smer_end+1]*we_idm 
            ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws_idm + reads_encodings[idm, smer_end+1, idy]*we_idm 

            ips_sums[idm, threadIdx().y] = abs2(ip_new)
            sync_threads()

            stride = blockDim().x ÷ 2
            while stride > 0
                if idm <= stride
                    ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
                end
                sync_threads()
                stride ÷= 2
            end
            fid = ips_sums[1, threadIdx().y]

            # if fid > val 
            #     val = fid
            #     loc = i-s+1
            # end
            
            # val = is_better ? fid : val
            # loc = is_better ? i-s+1 : loc
            # val, loc = fid > val ? (fid, i-s+1) : (val, loc)

            is_better = fid > val
            val = CUDA.ifelse(is_better, fid, val) # ifelse not necessary as warp sees same data 
            loc = CUDA.ifelse(is_better, i-s+1, loc)
            # val = max(fid, val)

            smer_start <<= 2
            smer_start &= bmask
            smer_start += dnas[idy][i]
        end

        vals[idy] = val 
        locs[idy] = loc
        
        return
    end
end

# todo norm recomp
@generated function kernel_super_noshift(s, ::Val{M}, k, dnas, reads_encodings, ips, vals, locs, e) where M    
    # function kernel(s, k, dna_ref, reads_encodings, ips, locs, ws, we)
    quote
        n = size(reads_encodings, 3)
        N = size(dnas, 1)

        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x
        if idy > n
            return
        end

        # ips_sums = @cuDynamicSharedMem(Float32, blockDim().x)  
        ips_sums = @cuDynamicSharedMem(ComplexF32, (blockDim().x, blockDim().y))  
        ip_new = ips[idm, idy]

        # # Each thread writes its partial sum to shared memory.
        # # Indexing is 1-based for Julia arrays, including shared memory.
        # sdata[threadIdx().x + 1] = thread_sum
        
        # # Ensure all threads have finished writing to shared memory before proceeding.
        # sync_threads()
        
        ips_sums[idm, threadIdx().y] = ips[idm, idy]
        sync_threads()

        # # Perform the reduction. The number of active threads is halved in each step.
        # # This requires the number of threads per block to be a power of 2.
        stride = blockDim().x ÷ 2
        while stride > 0
            # Only the first `stride` threads participate in this reduction step.
            if idm <= stride
                ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
            end
            # Wait for all threads in the block to complete this step.
            sync_threads()
            stride ÷= 2
        end

        val = abs2(ips_sums[1, threadIdx().y])        

        smer_start = UInt32(0)
        smer_end = UInt32(0)
        for i = 1 : s
            smer_start <<= 2
            smer_start += dnas[i, idy]

            smer_end <<= 2
            smer_end += dnas[i-s+k, idy]            
        end

        bmask = UInt32(4^s - 1)

        # @nexprs $M mi -> (ip_new_{mi} = ips[mi, idx]) 
        

        # CUDA.@cuprintf("CUDA kernel %d %f\n", idx, ip_new_1)

        
        # val = abs2.(ip_new) |> sum
        # val = abs2(ip_new_1)
        # val = 0
        # @nexprs $M mi -> (val += abs2(ip_new_{mi}))

        # if idx < 32
            # CUDA.@cuprintf("CUDA kernel %d %d\n", idx, $M1)
            # CUDA.@cuprintln(idx, $M1)
            # CUDA.@cuprintf("CUDA kernel %d %f %f %f %f\n", idx, abs2(ip_new_1), abs2(ip_new_2), abs2(ip_new_3), abs2(ip_new_4))
        # end

        # ws = e[2]
        # we = e[k-s+1]
        ws_idm = e[idm + 1]
        we_idm = e[(k-s)*idm % k + 1]

        loc = 1
        for i = s+1 : N-k+s 
            smer_end <<= 2
            smer_end &= bmask
            smer_end += dnas[i-s+k, idy]            

            # @nexprs $M mi -> (ws_0 = ws_0*ws; we_0 = we_0*we; ip_new_{mi} = (ip_new_{mi} - reads_encodings[mi, smer_start+1, idx])*ws_0 + reads_encodings[mi, smer_end+1, idx]*we_0)
            # ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws^idm + reads_encodings[idm, smer_end+1, idy]*we^idm # todo optimize ^idm
            ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws_idm + reads_encodings[idm, smer_end+1, idy]*we_idm 

            # fid = 0.0
            # @nexprs $M mi -> (fid += abs2(ip_new_{mi}))
            ips_sums[idm, threadIdx().y] = ip_new
            sync_threads()

            stride = blockDim().x ÷ 2
            while stride > 0
                if idm <= stride
                    ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
                end
                sync_threads()
                stride ÷= 2
            end
            fid = abs2(ips_sums[1, threadIdx().y])

            # if fid > val 
            #     val = fid
            #     loc = i-s+1
            # end
            
            # val = is_better ? fid : val
            # loc = is_better ? i-s+1 : loc
            # val, loc = fid > val ? (fid, i-s+1) : (val, loc)

            is_better = fid > val
            val = CUDA.ifelse(is_better, fid, val) # ifelse not necessary as warp sees same data 
            loc = CUDA.ifelse(is_better, i-s+1, loc)
            # val = max(fid, val)

            smer_start <<= 2
            smer_start &= bmask
            smer_start += dnas[i, idy]
        end

        vals[idy] = val 
        locs[idy] = loc
        
        return
    end
end

"""
dnas - a single combined vector of dna parts. 
srats and stops - correspond to ranges around found good_locs.  
reads_encodings - ropedim × n matrix 
ips - a single combined vector of inner products - 
"""
@generated function kernel_super_new(s, k, dnas, dstarts, dstops, reads_encodings, reads_inds, ips, vals, locs)   
    # function kernel(s, k, dna_ref, reads_encodings, ips, locs, ws, we)
    quote
        n = size(reads_inds, 1)
        # N = size(dnas, 1)

        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x
        if idy > n
            return
        end

        read_ind = reads_inds[idy]

        # ips_sums = @cuDynamicSharedMem(Float32, blockDim().x)  
        ips_sums = @cuDynamicSharedMem(Float32, (blockDim().x, blockDim().y))  
        ip_new = ips[idm, idy]

        # # Each thread writes its partial sum to shared memory.
        # # Indexing is 1-based for Julia arrays, including shared memory.
        # sdata[threadIdx().x + 1] = thread_sum
        
        # # Ensure all threads have finished writing to shared memory before proceeding.
        # sync_threads()
        
        # ips_sums[idm] = abs2(ips[idm, idy])
        ips_sums[idm, threadIdx().y] = abs2(ips[idm, idy])
        sync_threads()

        # # Perform the reduction. The number of active threads is halved in each step.
        # # This requires the number of threads per block to be a power of 2.
        stride = blockDim().x ÷ 2
        while stride > 0
            # Only the first `stride` threads participate in this reduction step.
            if idm <= stride
                ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
            end
            # Wait for all threads in the block to complete this step.
            sync_threads()
            stride ÷= 2
        end

        val = ips_sums[1, threadIdx().y]        

        region_start = dstarts[idy]
        region_stop = dstops[idy]

        smer_start = UInt32(0)
        smer_end = UInt32(0)
        # for i = 1 : s
        for i = region_start : region_start + s-1
            smer_start <<= 2
            # smer_start += dnas[i, idy]
            smer_start += dnas[i]

            smer_end <<= 2
            smer_end += dnas[i-s+k]            
        end

        bmask = UInt32(4^s - 1)

        # @nexprs $M mi -> (ip_new_{mi} = ips[mi, idx]) 
        

        # CUDA.@cuprintf("CUDA kernel %d %f\n", idx, ip_new_1)

        
        # val = abs2.(ip_new) |> sum
        # val = abs2(ip_new_1)
        # val = 0
        # @nexprs $M mi -> (val += abs2(ip_new_{mi}))

        # if idx < 32
            # CUDA.@cuprintf("CUDA kernel %d %d\n", idx, $M1)
            # CUDA.@cuprintln(idx, $M1)
            # CUDA.@cuprintf("CUDA kernel %d %f %f %f %f\n", idx, abs2(ip_new_1), abs2(ip_new_2), abs2(ip_new_3), abs2(ip_new_4))
        # end

        # ws = e[2]
        # we = e[k-s+1]
        # ws_idm = e[idm + 1]
        # we_idm = e[(k-s)*idm % k + 1]
        ws_idm = exp(idm/k*2π*im) 
        we_idm = exp((k-s)/k*idm*2π*im) 
        
        loc = region_start
        # for i = s+1 : N-k+s 
        for i = region_start + s : region_stop - k + s 
            smer_end <<= 2
            smer_end &= bmask
            # smer_end += dnas[i-s+k, idy]            
            smer_end += dnas[i-s+k]            

            # @nexprs $M mi -> (ws_0 = ws_0*ws; we_0 = we_0*we; ip_new_{mi} = (ip_new_{mi} - reads_encodings[mi, smer_start+1, idx])*ws_0 + reads_encodings[mi, smer_end+1, idx]*we_0)
            # ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws^idm + reads_encodings[idm, smer_end+1, idy]*we^idm # todo optimize ^idm
            # ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws_idm + reads_encodings[idm, smer_end+1, idy]*we_idm 
            ip_new = (ip_new - reads_encodings[idm, smer_start+1, read_ind])*ws_idm + reads_encodings[idm, smer_end+1, read_ind]*we_idm 

            # fid = 0.0
            # @nexprs $M mi -> (fid += abs2(ip_new_{mi}))
            ips_sums[idm, threadIdx().y] = abs2(ip_new)
            sync_threads()

            stride = blockDim().x ÷ 2
            while stride > 0
                if idm <= stride
                    ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
                end
                sync_threads()
                stride ÷= 2
            end
            fid = ips_sums[1, threadIdx().y]

            # if fid > val 
            #     val = fid
            #     loc = i-s+1
            # end
            
            # val = is_better ? fid : val
            # loc = is_better ? i-s+1 : loc
            # val, loc = fid > val ? (fid, i-s+1) : (val, loc)

            is_better = fid > val
            val = CUDA.ifelse(is_better, fid, val) # ifelse not necessary as warp sees same data 
            loc = CUDA.ifelse(is_better, i-s+1, loc)
            # val = max(fid, val)

            smer_start <<= 2
            smer_start &= bmask
            # smer_start += dnas[i, idy]
            smer_start += dnas[i]
        end

        vals[idy] = val 
        locs[idy] = loc
        
        return
    end
end


# @generated function kernel_super(s, ::Val{M}, k, dnas, reads_encodings, ips, vals, locs, ws, we) where M
@generated function kernel_super(s, ::Val{M}, k, dnas, reads_encodings, ips, vals, locs, e) where M    
    # function kernel(s, k, dna_ref, reads_encodings, ips, locs, ws, we)
    quote
        n = size(reads_encodings, 3)
        N = size(dnas, 1)

        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x
        if idy > n
            return
        end

        # ips_sums = @cuDynamicSharedMem(Float32, blockDim().x)  
        ips_sums = @cuDynamicSharedMem(Float32, (blockDim().x, blockDim().y))  
        ip_new = ips[idm, idy]

        # # Each thread writes its partial sum to shared memory.
        # # Indexing is 1-based for Julia arrays, including shared memory.
        # sdata[threadIdx().x + 1] = thread_sum
        
        # # Ensure all threads have finished writing to shared memory before proceeding.
        # sync_threads()
        
        # ips_sums[idm] = abs2(ips[idm, idy])
        ips_sums[idm, threadIdx().y] = abs2(ips[idm, idy])
        sync_threads()

        # # Perform the reduction. The number of active threads is halved in each step.
        # # This requires the number of threads per block to be a power of 2.
        stride = blockDim().x ÷ 2
        while stride > 0
            # Only the first `stride` threads participate in this reduction step.
            if idm <= stride
                ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
            end
            # Wait for all threads in the block to complete this step.
            sync_threads()
            stride ÷= 2
        end

        val = ips_sums[1, threadIdx().y]        

        smer_start = UInt32(0)
        smer_end = UInt32(0)
        for i = 1 : s
            smer_start <<= 2
            smer_start += dnas[i, idy]

            smer_end <<= 2
            smer_end += dnas[i-s+k, idy]            
        end

        bmask = UInt32(4^s - 1)

        # @nexprs $M mi -> (ip_new_{mi} = ips[mi, idx]) 
        

        # CUDA.@cuprintf("CUDA kernel %d %f\n", idx, ip_new_1)

        
        # val = abs2.(ip_new) |> sum
        # val = abs2(ip_new_1)
        # val = 0
        # @nexprs $M mi -> (val += abs2(ip_new_{mi}))

        # if idx < 32
            # CUDA.@cuprintf("CUDA kernel %d %d\n", idx, $M1)
            # CUDA.@cuprintln(idx, $M1)
            # CUDA.@cuprintf("CUDA kernel %d %f %f %f %f\n", idx, abs2(ip_new_1), abs2(ip_new_2), abs2(ip_new_3), abs2(ip_new_4))
        # end

        # ws = e[2]
        # we = e[k-s+1]
        ws_idm = e[idm + 1]
        we_idm = e[(k-s)*idm % k + 1]

        loc = 1
        for i = s+1 : N-k+s 
            smer_end <<= 2
            smer_end &= bmask
            smer_end += dnas[i-s+k, idy]            

            # @nexprs $M mi -> (ws_0 = ws_0*ws; we_0 = we_0*we; ip_new_{mi} = (ip_new_{mi} - reads_encodings[mi, smer_start+1, idx])*ws_0 + reads_encodings[mi, smer_end+1, idx]*we_0)
            # ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws^idm + reads_encodings[idm, smer_end+1, idy]*we^idm # todo optimize ^idm
            ip_new = (ip_new - reads_encodings[idm, smer_start+1, idy])*ws_idm + reads_encodings[idm, smer_end+1, idy]*we_idm 

            # fid = 0.0
            # @nexprs $M mi -> (fid += abs2(ip_new_{mi}))
            ips_sums[idm, threadIdx().y] = abs2(ip_new)
            sync_threads()

            stride = blockDim().x ÷ 2
            while stride > 0
                if idm <= stride
                    ips_sums[idm, threadIdx().y] += ips_sums[idm + stride, threadIdx().y]
                end
                sync_threads()
                stride ÷= 2
            end
            fid = ips_sums[1, threadIdx().y]

            # if fid > val 
            #     val = fid
            #     loc = i-s+1
            # end
            
            # val = is_better ? fid : val
            # loc = is_better ? i-s+1 : loc
            # val, loc = fid > val ? (fid, i-s+1) : (val, loc)

            is_better = fid > val
            val = CUDA.ifelse(is_better, fid, val) # ifelse not necessary as warp sees same data 
            loc = CUDA.ifelse(is_better, i-s+1, loc)
            # val = max(fid, val)

            smer_start <<= 2
            smer_start &= bmask
            smer_start += dnas[i, idy]
        end

        vals[idy] = val 
        locs[idy] = loc
        
        return
    end
end


@generated function kernel(s, ::Val{M}, k, dnas, reads_encodings, ips, vals, locs, ws, we) where M
    # function kernel(s, k, dna_ref, reads_encodings, ips, locs, ws, we)
    M1 = M+1
    quote
        n = size(reads_encodings, 3)
        N = size(dnas, 1)

        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if idx > n
            return
        end

        # CUDA.@cuprintf("CUDA kernel %d %d\n", idx, $M1)

        # s = re.s 
        # k = re.k

        smer_start = UInt32(0)
        smer_end = UInt32(0)
        for i = 1 : s
            smer_start <<= 2
            smer_start += dnas[i, idx]

            smer_end <<= 2
            smer_end += dnas[i-s+k, idx]            
        end

        bmask = UInt32(4^s - 1)
        
        # ip_new = [ips[mi, idx] for mi = 1:m]
        # ip_new = MVector{m, Float32}(undef)
        # for mi = 1:m
        #     ip_new[mi] = ips[mi, idx] 
        # end

        # Base.Cartesian.@nexprs 4 i -> y_i = x[i]
        @nexprs $M mi -> (ip_new_{mi} = ips[mi, idx]) 

        # CUDA.@cuprintf("CUDA kernel %d %f\n", idx, ip_new_1)

        loc = 1
        # val = abs2.(ip_new) |> sum
        # val = abs2(ip_new_1)
        val = 0
        @nexprs $M mi -> (val += abs2(ip_new_{mi}))

        if idx < 32
            # CUDA.@cuprintf("CUDA kernel %d %d\n", idx, $M1)
            # CUDA.@cuprintln(idx, $M1)
            # CUDA.@cuprintf("CUDA kernel %d %f %f %f %f\n", idx, abs2(ip_new_1), abs2(ip_new_2), abs2(ip_new_3), abs2(ip_new_4))
        end

        # ws = e[2]
        # we = e[k-s+1]
        for i = s+1 : N-k+s 
            smer_end <<= 2
            smer_end &= bmask
            smer_end += dnas[i-s+k, idx]            

            # # ip_new = ip_new*ws - reads_encodings[smer_start+1, idx]*ws + reads_encodings[smer_end+1, idx]/we

            # ip_new = ip_new*ws - reads_encodings[idx, smer_start+1]*ws + reads_encodings[idx, smer_end+1]*we # transposed
            # ip_new = ip_new*ws - reads_encodings[smer_start+1, idx]*ws + reads_encodings[smer_end+1, idx]*we # todo fix norm?
            # for mi = 1:m 
            #     ip_new[mi] = ip_new[mi]*ws - reads_encodings[mi, smer_start+1, idx]*ws + reads_encodings[mi, smer_end+1, idx]*we # todo fix norm?
            #     fid = abs2.(ip_new) |> sum
            # end 
            ws_0 = 1.0 + 0*im
            we_0 = 1.0 + 0*im
            # @nexprs $M mi -> (ip_new_{mi} = ip_new_{mi}*ws^mi - reads_encodings[mi, smer_start+1, idx]*ws^mi + reads_encodings[mi, smer_end+1, idx]*we^mi)
            # @nexprs $M mi -> (ws_{mi} = ws_{mi-1}*ws; we_{mi} = we_{mi-1}*we; ip_new_{mi} = ip_new_{mi}*ws_{mi} - reads_encodings[mi, smer_start+1, idx]*ws_{mi} + reads_encodings[mi, smer_end+1, idx]*we_{mi})
            @nexprs $M mi -> (ws_0 = ws_0*ws; we_0 = we_0*we; ip_new_{mi} = (ip_new_{mi} - reads_encodings[mi, smer_start+1, idx])*ws_0 + reads_encodings[mi, smer_end+1, idx]*we_0)
            fid = 0.0
            @nexprs $M mi -> (fid += abs2(ip_new_{mi}))

            # if fid > val 
            #     val = fid
            #     loc = i-s+1
            # end
            
            # val = is_better ? fid : val
            # loc = is_better ? i-s+1 : loc
            # val, loc = fid > val ? (fid, i-s+1) : (val, loc)

            is_better = fid > val
            val = CUDA.ifelse(is_better, fid, val)
            loc = CUDA.ifelse(is_better, i-s+1, loc)
            # val = max(fid, val)

            smer_start <<= 2
            smer_start &= bmask
            smer_start += dnas[i, idx]
        end

        vals[idx] = val 
        locs[idx] = loc
        
        return
    end
end

function sliding_search_cuda(
    ri::RopeIndexer, 
    re2::RopeEncoder, 
    # dna_ref::CuArray, 
    reads::CuArray,
    # range::UnitRange{Int}; 
    # e::CuArray
    ext_len = re.k ÷ 10
    ;pos
) 
    re = re2
    # re = ri.re 
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s, m=64, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+1, m=16, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+2, m=4, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=1, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=4, c=ri.re.c) # not enough mem on 3090

    e = ri.cu_e
    dna_ref = ri.cu_dna_ref

    # encode reads
    nr = size(reads, 2)
    # @show n2

    # @show length(dna_ref)
    
    # CUDA.@time reads_encodings, reads_encodings_norm = encode_reads_batch_m1_cuda(re, reads, 1:re.k-re.s+1, e=e)
    @info "Recomputing reads encodings with higher dim"
    CUDA.@time reads_encodings, reads_encodings_norm = encode_batch_cuda_all_best(re, reads, e=e)
    @show size(reads_encodings)

    # encode initial segment # todo how to to speed up?  
    # CUDA.@time head_encoding, head_encoding_norm = encode_reads_batch_m1_cuda(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)

    # ext_len = re.k ÷ 10 # how much more ips we want to compute
    # ext_len = re.k

    # @show ri.N

    # the fixed range covers the total part we need from dna_ref, it should have length re.k + ext_len
    function fixed_range(p::Int) # todo be sure ri.N is large enough
        ran = p + 1 : p + re.k + ext_len
        ran = ran .- (ext_len ÷ 2)
        if ran.start < 1 
            ran = ran .+ (1-ran.start)
        elseif ran.stop > ri.N
            ran = ran .+ (ri.N-ran.stop)
        end            
        return ran
    end
    # ranges = [ (p + 1 : p + win_size + re.k) .- min(p, win_size÷2) for p in pos] # todo: check ranges
    ranges = [fixed_range(p) for p in pos] 

    # @show ranges |> length
    # @show maximum(pos)
    # @show ranges[argmax(pos)]


    dnas = CUDA.zeros(UInt8, re.k + ext_len, nr)
    for i in 1:nr 
        @views dnas[:, i] .= dna_ref[ranges[i]]
    end

    @info "Recomputing index encodings with higher dim"
    CUDA.@time heads_encodings, heads_encodings_norm = encode_batch_cuda_all_best(re, dnas, e=ri.cu_e, start=1, stop=re.k-re.s+1, normalize = false)
    # CUDA.@time head_encoding, head_encoding_norm = encode_batch_cuda_best(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)
    # head_encoding, head_encoding_norm = he[1][1], h2[1][1]
    @show size(heads_encodings)

    # CUDA.@time ips = heads_encodings'.*reads_encodings
    @info "Recomputing inner products"
    CUDA.@time ips = sum(conj.(heads_encodings) .* reads_encodings, dims=2) # todo: use FP16?
    @show size(ips)

    # @show abs2.(ips[1:re.m, 1, 1:10])

    # hm = sort(vec(Array(abs2.(ips))))
    # @show hm[1:100]
    # @show hm[end-100:end]

    # hm2 = sort(vec(Array(reads_encodings_norm)))
    # @show hm2[1:100]
    # @show hm2[end-100:end]

    vals = CUDA.zeros(Float32, nr)
    locs = CUDA.zeros(Int32, nr) 

    CUDA.@allowscalar begin 
        ws = ri.cu_e[2]
        we = ri.cu_e[re.k-re.s+1]'
    end

    threads = min(2^5, nr)
    blocks = ceil(Int, nr / threads)

    # CUDA.@time reads_encodings_t = CuArray(transpose(reads_encodings))
    # CUDA.@time ips2 = CuArray(ips)

    @info "Sliding search kernel run"
    CUDA.@time @cuda blocks=blocks threads=threads kernel(
        re.s,
        Val(re.m),
        re.k,
        dnas,
        reads_encodings, 
        ips, 
        vals, 
        locs, 
        ws,
        we
    )

    # @show vals[1:100]
    cpu_locs = Array(locs)
    return [ranges[i].start + cpu_locs[i]  - 1 for i in 1:nr]
end


function sliding_search_cuda_locs(
    ri::RopeIndexer, 
    re2::RopeEncoder, 
    # dna_ref::CuArray, 
    reads::CuArray,
    starts::CuArray,
    stops::CuArray,
    # range::UnitRange{Int}; 
    # e::CuArray
    found_locs    
    ;ext_len = re2.k ÷ 2
) 
    re = re2
    # re = ri.re 
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s, m=64, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+1, m=16, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+2, m=4, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=1, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=4, c=ri.re.c) # not enough mem on 3090

    # e = ri.cu_e
    dna_ref = ri.cu_dna_ref

    # encode reads
    rstarts = starts 
    rstops = stops
    nr = length(rstarts)
    ntotal_locs = sum(length.(found_locs))
    # @show n2

    # @show length(dna_ref)
    
    # CUDA.@time reads_encodings, reads_encodings_norm = encode_reads_batch_m1_cuda(re, reads, 1:re.k-re.s+1, e=e)
    @info "Recomputing reads encodings with higher dim"
    creads = reads
    CUDA.@time reads_encodings, reads_encodings_norm = encode_batch_cuda_all_best(re, creads, starts=rstarts, stops=rstops)
    @show size(reads_encodings)

    # encode initial segment # todo how to to speed up?  
    # CUDA.@time head_encoding, head_encoding_norm = encode_reads_batch_m1_cuda(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)

    # ext_len = re.k ÷ 10 # how much more ips we want to compute
    # ext_len = re.k

    # @show ri.N

    # the fixed range covers the total part we need from dna_ref, it should have length re.k + ext_len
    function search_range(loc; blow=1, bhigh=ri.N) 
        ran = loc + 1 : loc + re.k + ext_len
        ran = ran .- (ext_len ÷ 2)
        if ran.start < 1 
            ran = ran .+ (1-ran.start)
        elseif ran.stop > ri.N
            ran = ran .+ (ri.N-ran.stop)
        end            
        return ran.start, ran.stop
    end
    
    # dstarts = UInt32[]
    # dstops = UInt32[]
    dstarts = Int32[]
    dstops = Int32[]
    dstops1 = Int32[]
    reads_inds = UInt32[]
    for i = 1:length(found_locs)
        locs = found_locs[i]
        append!(reads_inds, UInt32(i)*ones(UInt32, length(locs)))
        for l in locs 
            a, b = search_range(l)
            push!(dstarts, a)
            push!(dstops, b)
            push!(dstops1, a+re.k-1)
        end
    end

    @show length(reads_inds)
    # @show dstarts[1:10] 
    # @show dstops[1:10] 
    # @show dstops1[1:10] 

    # throw("check ds")

    @show length(dstarts) - ntotal_locs
    dstarts = CuArray(dstarts)
    dstops = CuArray(dstops)
    dstops1 = CuArray(dstops1)
    # ranges = [fixed_range(p[1]) for p in found_ranges] # taking 1st range for each read

    # @show ranges |> length
    # @show maximum(pos)
    # @show ranges[argmax(pos)]

    # n_ranges_total = sum(length.(found_ranges))
    # n_ranges_total = nr


    # dnas = CUDA.zeros(UInt8, re.k + ext_len, n_ranges_total)
    # cropes = CUDA.zeros(ComplexF32, 4^re2.s * re2.m, n_ranges_total)

    # for i in 1:nr 
    #     @views dnas[:, i] .= dna_ref[ranges[i]]
    # end
    # dnas = [(@view dna_ref[ranges[i]]) for i in 1:nr]

    dnas = ri.cu_dna_ref

    @info "Recomputing index encodings with higher dim"
    CUDA.@time heads_encodings, heads_encodings_norm = encode_batch_cuda_all_best(re, dnas, starts=dstarts, stops=dstops1, normalize = false)
    # CUDA.@time head_encoding, head_encoding_norm = encode_batch_cuda_best(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)
    # head_encoding, head_encoding_norm = he[1][1], h2[1][1]
    @show size(heads_encodings)

    # CUDA.@time ips = heads_encodings'.*reads_encodings
    @info "Recomputing inner products"
    # CUDA.@time ips = sum(conj.(heads_encodings) .* reads_encodings, dims=2) # todo: use FP16?
    
    # gathered_reads = reads_encodings[:, :, reads_inds]
    # CUDA.@time ips = vec(sum(conj(heads_encodings) .* gathered_reads, dims=(1, 2)))

    function batch_dot_kernel!(ips, heads_encodings, reads_encodings, reads_inds)
        # Each thread will compute one element of `ips`.
        # i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x
        if idy > length(reads_inds)
            return
        end

        # if i <= length(ips)
        read_ind = reads_inds[idy]
        dot_val = zero(eltype(ips))

        # Manually compute the dot product for the i-th element
        # This loop runs on a single GPU thread
        @inbounds for k in 1:size(heads_encodings, 2)
            # for j in 1:size(heads_encodings, 1)
                dot_val += conj(heads_encodings[idm, k, idy]) * reads_encodings[idm, k, read_ind]
            # end
        end
        ips[idm, idy] = dot_val
        # end
        return
    end

    # threads = 32
    # blocks = ceil(Int, ntotal_locs / threads)

    threads = (re.m, max(1, (32÷re.m)*1))
    blocks = (1, cld(ntotal_locs, max(1, (32÷re.m)*1)))

    ips = CUDA.zeros(ComplexF32, re.m, ntotal_locs)
    CUDA.@time @cuda threads=threads blocks=blocks batch_dot_kernel!(ips, heads_encodings, reads_encodings, cu(reads_inds))

    # ips = CUDA.zeros(ComplexF32, ntotal_locs)
    # CUDA.@time for i = 1:ntotal_locs
    #     read_i = reads_inds[i]
    #     CUDA.@allowscalar ips[i] = vec(heads_encodings[:,:,i])'*vec(reads_encodings[:,:,read_i])
    # end
    @show size(ips)

    # throw("check")

    # @show abs2.(ips[1:re.m, 1, 1:10])

    # hm = sort(vec(Array(abs2.(ips))))
    # @show hm[1:100]
    # @show hm[end-100:end]

    # hm2 = sort(vec(Array(reads_encodings_norm)))
    # @show hm2[1:100]
    # @show hm2[end-100:end]

    vals = CUDA.zeros(Float32, ntotal_locs)
    better_locs = CUDA.zeros(UInt32, ntotal_locs)

    # CUDA.@allowscalar begin 
    #     ws = ri.cu_e[2]
    #     we = ri.cu_e[re.k-re.s+1]'
    # end

    # threads = min(2^5, nr)
    # blocks = ceil(Int, nr / threads)

    # threads = (32, 1)
    # blocks = (1, nr)

    threads = (re.m, max(1, (32÷re.m)*1))
    blocks = (1, cld(ntotal_locs, max(1, (32÷re.m)*1)))

    # CUDA.@time reads_encodings_t = CuArray(transpose(reads_encodings))
    # CUDA.@time ips2 = CuArray(ips)

    @info "Sliding search kernel run"
    res = CUDA.@profile trace=true begin 
        # CUDA.@time 
        # @cuda blocks=blocks threads=threads kernel(
        # @cuda blocks=blocks threads=threads shmem=4*re.m*max(1,(32÷re.m)*1) kernel_super_v2(
        @cuda blocks=blocks threads=threads shmem=4*re.m*max(1,(32÷re.m)*1) kernel_super_new(
        # @cuda blocks=blocks threads=threads shmem=2*4*re.m*max(1,(32÷re.m)*1) kernel_super_noshift(
            re.s,
            # Val(re.m),
            re.k,
            dnas,
            dstarts,
            dstops,
            reads_encodings, 
            cu(reads_inds),
            ips, 
            vals, 
            better_locs, 
            # ri.cu_e
            # ws,
            # we
        )
    end 
    display(res)

    # @show vals[1:100]
    # cpu_locs = Array(locs)
    # return [ranges[i].start + cpu_locs[i]  - 1 for i in 1:nr]
    return Array(better_locs), Array(vals)
end


function sliding_search_cuda_ranges(
    ri::RopeIndexer, 
    re2::RopeEncoder, 
    # dna_ref::CuArray, 
    reads::CuArray,
    # range::UnitRange{Int}; 
    # e::CuArray
    ext_len = re.k ÷ 2
    ;found_ranges
) 


    re = re2
    # re = ri.re 
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s, m=64, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+1, m=16, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+2, m=4, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=1, c=ri.re.c)
    # re = RopeEncoder(k=ri.re.k, s=ri.re.s+3, m=4, c=ri.re.c) # not enough mem on 3090

    e = ri.cu_e
    dna_ref = ri.cu_dna_ref

    # encode reads
    nr = size(reads, 2)
    # @show n2

    # @show length(dna_ref)
    
    # CUDA.@time reads_encodings, reads_encodings_norm = encode_reads_batch_m1_cuda(re, reads, 1:re.k-re.s+1, e=e)
    @info "Recomputing reads encodings with higher dim"
    CUDA.@time reads_encodings, reads_encodings_norm = encode_batch_cuda_all_best(re, reads, e=e)
    @show size(reads_encodings)

    # encode initial segment # todo how to to speed up?  
    # CUDA.@time head_encoding, head_encoding_norm = encode_reads_batch_m1_cuda(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)

    # ext_len = re.k ÷ 10 # how much more ips we want to compute
    # ext_len = re.k

    # @show ri.N

    # the fixed range covers the total part we need from dna_ref, it should have length re.k + ext_len
    function fixed_range(found_range) 
        mid = sum(found_range)÷2 # todo: !important! we could lose some if frange is too long; solution: split frange if too long?
        ran = mid + 1 : mid + re.k + ext_len
        ran = ran .- (ext_len ÷ 2)
        if ran.start < 1 
            ran = ran .+ (1-ran.start)
        elseif ran.stop > ri.N
            ran = ran .+ (ri.N-ran.stop)
        end            
        return ran
    end
    ranges = [fixed_range(p[1]) for p in found_ranges] # taking 1st range for each read

    # @show ranges |> length
    # @show maximum(pos)
    # @show ranges[argmax(pos)]

    # n_ranges_total = sum(length.(found_ranges))
    n_ranges_total = nr


    dnas = CUDA.zeros(UInt8, re.k + ext_len, n_ranges_total)
    # cropes = CUDA.zeros(ComplexF32, 4^re2.s * re2.m, n_ranges_total)

    for i in 1:nr 
        @views dnas[:, i] .= dna_ref[ranges[i]]
    end
    # dnas = [(@view dna_ref[ranges[i]]) for i in 1:nr]

    @info "Recomputing index encodings with higher dim"
    CUDA.@time heads_encodings, heads_encodings_norm = encode_batch_cuda_all_best(re, dnas, e=ri.cu_e, start=1, stop=re.k-re.s+1, normalize = false)
    # CUDA.@time head_encoding, head_encoding_norm = encode_batch_cuda_best(re, hcat(dna_ref), 1:re.k-re.s+1, e=e)
    # head_encoding, head_encoding_norm = he[1][1], h2[1][1]
    @show size(heads_encodings)

    # CUDA.@time ips = heads_encodings'.*reads_encodings
    @info "Recomputing inner products"
    CUDA.@time ips = sum(conj.(heads_encodings) .* reads_encodings, dims=2) # todo: use FP16?
    @show size(ips)

    # @show abs2.(ips[1:re.m, 1, 1:10])

    # hm = sort(vec(Array(abs2.(ips))))
    # @show hm[1:100]
    # @show hm[end-100:end]

    # hm2 = sort(vec(Array(reads_encodings_norm)))
    # @show hm2[1:100]
    # @show hm2[end-100:end]

    vals = CUDA.zeros(Float32, nr)
    locs = CUDA.zeros(Int32, nr) 

    CUDA.@allowscalar begin 
        ws = ri.cu_e[2]
        we = ri.cu_e[re.k-re.s+1]'
    end

    # threads = min(2^5, nr)
    # blocks = ceil(Int, nr / threads)

    # threads = (32, 1)
    # blocks = (1, nr)

    threads = (re.m, max(1, (32÷re.m)*1))
    blocks = (1, cld(nr, max(1, (32÷re.m)*1)))

    # CUDA.@time reads_encodings_t = CuArray(transpose(reads_encodings))
    # CUDA.@time ips2 = CuArray(ips)

    @info "Sliding search kernel run"
    res = CUDA.@profile trace=true begin 
        # CUDA.@time 
        # @cuda blocks=blocks threads=threads kernel(
        # @cuda blocks=blocks threads=threads shmem=4*re.m*max(1,(32÷re.m)*1) kernel_super_v2(
        @cuda blocks=blocks threads=threads shmem=4*re.m*max(1,(32÷re.m)*1) kernel_super(
        # @cuda blocks=blocks threads=threads shmem=2*4*re.m*max(1,(32÷re.m)*1) kernel_super_noshift(
            re.s,
            Val(re.m),
            re.k,
            dnas,
            reads_encodings, 
            ips, 
            vals, 
            locs, 
            ri.cu_e
            # ws,
            # we
        )
    end 
    display(res)

    # @show vals[1:100]
    cpu_locs = Array(locs)
    return [ranges[i].start + cpu_locs[i]  - 1 for i in 1:nr]
end




function topk_kernel!(out_vals, out_inds, A, k)
    # Map 1 thread to 1 row
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M, N = size(A)

    if row <= M
        # Initialize the top-k arrays for this row
        for i in 1:k
            out_vals[row, i] = typemin(eltype(A))
            out_inds[row, i] = 0
        end

        # Iterate across the columns
        for col in 1:N
            val = A[row, col]
            
            # If we find a value bigger than our current smallest top-k
            if val > out_vals[row, k]
                pos = k
                
                # Bubble it up into the correct sorted position (Insertion Sort)
                while pos > 1 && val > out_vals[row, pos-1]
                    out_vals[row, pos] = out_vals[row, pos-1]
                    out_inds[row, pos] = out_inds[row, pos-1]
                    pos -= 1
                end
                
                # Insert the new value and its index
                out_vals[row, pos] = val
                out_inds[row, pos] = col
            end
        end
    end
    return nothing
end

function get_topk(A::CuMatrix{T}, k::Int) where T
    M, N = size(A)
    
    # Pre-allocate output arrays on the GPU
    out_vals = CUDA.fill(typemin(T), (M, k))
    out_inds = CUDA.zeros(Int32, (M, k)) # Int32 is usually sufficient and saves VRAM

    # Define thread and block counts
    threads = 256
    blocks = cld(M, threads)

    # Launch the kernel
    @cuda threads=threads blocks=blocks topk_kernel!(out_vals, out_inds, A, k)

    return out_vals, out_inds
end

function merge_topk_kernel!(out_vals, out_inds, valsA, indsA, valsB, indsB, offsetB)
    # Map 1 thread to 1 row
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M = size(valsA, 1)
    k = size(valsA, 2)

    if row <= M
        pA = 1 # Pointer for matrix A
        pB = 1 # Pointer for matrix B
        
        # We only need the top k elements from the combined 2k elements
        for j in 1:k
            # Grab current values, defaulting to the minimum possible if a pointer exceeds k
            valA = pA <= k ? valsA[row, pA] : typemin(eltype(valsA))
            valB = pB <= k ? valsB[row, pB] : typemin(eltype(valsB))

            # Pick the larger value and advance its respective pointer
            if valA >= valB
                out_vals[row, j] = valA
                out_inds[row, j] = indsA[row, pA]
                pA += 1
            else
                out_vals[row, j] = valB
                
                # Only apply the offset if the index is valid (not a 0-padded placeholder)
                idxB = indsB[row, pB]
                out_inds[row, j] = idxB > 0 ? idxB + offsetB : 0
                # out_inds[row, j] = idxB + offsetB 
                pB += 1
            end
        end
    end
    return nothing
end

# Wrapper function for ease of use
function merge_topk(valsA::CuMatrix{T}, indsA::CuMatrix{I}, 
                    valsB::CuMatrix{T}, indsB::CuMatrix{I}; 
                    offsetB::Integer=0) where {T, I}
    
    M, k = size(valsA)
    
    # Pre-allocate output arrays
    out_vals = CUDA.similar(valsA)
    out_inds = CUDA.similar(indsA)
    
    threads = 256
    blocks = cld(M, threads)
    
    # Pass the offset as the correct integer type to avoid type instability in the kernel
    offsetB_typed = I(offsetB)

    @cuda threads=threads blocks=blocks merge_topk_kernel!(out_vals, out_inds, valsA, indsA, valsB, indsB, offsetB_typed)
    
    return out_vals, out_inds
end


end