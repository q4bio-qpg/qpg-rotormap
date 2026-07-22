# __precompile__(true)

module RopeIndexers

export RopeIndexer, index_m1_cuda, index_m1_cuda_optimized
export index_m1_cuda_nobatch, index_m1_cuda_v2, index_m1_cuda_v2_t, index_m1_cuda_new
export index_cuda_best!, index_cuda_best_new!, index_cuda_final
export select_kmers

using CUDA
using Base.Threads
using LinearAlgebra
using ProgressMeter
using ..SplitComplexMatrices
using ..Utils
using ..RopeEncoders
using ..RopeScores

struct RopeIndexer    
    re::RopeEncoder
    N::Int64 # length of the reference sequence    
    k::Int64 # length of the sliding window, should match re.k
    step::Int64 # step size of the window shift 
    n::Int64 # number of rope encodings
    cu_dna_ref::CuArray # original reference
    cu_index::CuArray # normal index
    cu_norms::CuArray # norms of encodings
    cu_e::CuArray # cache for exponents
    cut::Real # cutting threshold
    # index::Union{Array{ComplexF64, 3}, Array{ComplexF64, 2}} # re.m x 4^re.s x n
    # index::Union{Array{ComplexF32, 3}, Array{ComplexF32, 2}} # re.m x 4^re.s x n
    # index::Union{Array{ComplexF16, 3}, Array{ComplexF16, 2}} # re.m x 4^re.s x n
    # compact::Array{ComplexF64, 2} # compact index 
    # counts::Vector{Int64} # total counts of s-mers

    function RopeIndexer(dna_ref; re, step, err=0.30)        
        N = length(dna_ref)
        cu_dna_ref = dna_ref |> CuArray
        
        cu_e = exp.(2π*im/re.k .* (0:re.k+re.s)) |> cu

        n = length(0:step:N-re.k) # n = 1+(N-re.k)/step |> floor |> Int

        cu_index = CUDA.zeros(ComplexF32, re.m, 4^re.s, n)
        cu_norms = CUDA.zeros(Float32, re.m, n)

        # if re.m > 1
        #     index = zeros(ComplexF32, 4^re.s, re.m, n)
        # else
        #     index = zeros(ComplexF32, 4^re.s, n)
        # end

        # if re.c == 0
        #     compact = zeros(ComplexF32, 0, 0)
        # else 
        #     compact = zeros(ComplexF32, re.c, n)
        # end

        # counts = zeros(Int64, 4^re.s)

        @info "Estimating cutting threshold for $err error"
        CUDA.@time cut, _, _ = cutting_threshold_new(re, err+0.1, step/re.k/2)
        
        @show cut
        # cutm *= 0.9

        return new(re, N, re.k, step, n, cu_dna_ref, cu_index, cu_norms, cu_e, cut)
    end
end

# =============== INDEXING PHASE =================

using CUDA

function index_m1_cuda(self::RopeIndexer, dna_ref::AbstractVector{T}; e, batch_size = 1024) where T
    index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)

    batch = Array{T}(undef, self.k, batch_size)
    batch_gpu = CUDA.zeros(T, self.k, batch_size)
    batch_partial = 0
    batch_number = 0

    # p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...", barglyphs=BarGlyphs("[=> ]"))
    p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...")

    for loc in 0:self.step:self.N-self.k
        batch_partial += 1
        if batch_partial == batch_size # batch is filled         
            index_view = @view index[:, batch_number*batch_size+1:(batch_number+1)*batch_size]
            # @time 
            copyto!(batch_gpu, batch)
            # @show batch |> typeof,  batch_gpu |> typeof
            # @time embeds = encode_batch_m1_cuda(self.re, batch_gpu, 1:self.k, e = e)
            encode_batch_m1_cuda!(index_view, self.re, batch_gpu, 1:self.k, e = e)
            # @views index[:, batch_number*batch_size+1:(batch_number+1)*batch_size] .= embeds
            batch_partial = 0
            batch_number += 1
            next!(p)
            # update!(p, position(file))
        else 
            @views batch[:, batch_partial] .= dna_ref[loc+1:loc+self.k]
        end    
    end
    if batch_partial>0 # last partial batch
        index_view = @view index[:, batch_number*batch_size+1:self.n]
        # copyto!(batch_gpu, batch)
        # batch_view = @view batch[:, 1:batch_partial]
        batch_gpu = CuArray(view(batch, 1:self.k, 1:batch_partial))
        encode_batch_m1_cuda!(index_view, self.re, batch_gpu, 1:self.k, e = e)
    end
    
    # copyto!(self.index, index)
    return index
end

function index_m1_cuda_new(self::RopeIndexer, dna_ref::CuArray{T}; e) where T
    # index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)

    # dnas = CuArray(@view dna_ref[loc+1:loc+self.k] for loc in 0:self.step:self.N-self.k)
    dnas = collect(CuArray, (@view dna_ref[loc+1:loc+self.k] for loc in 0:self.step:self.N-self.k)) # won't work, need ranges 
    index, norms = encode_m1_cuda_new(self.re, dnas, e=e)

    return index
end

function index_m1_cuda_v2(self::RopeIndexer, dna_ref::CuArray{T}; e, batch_size = 2^12) where T
    index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)

    # batch = Array{T}(undef, self.k, batch_size)
    # batch_gpu = CUDA.zeros(T, batch_size, self.k) # transpose
    batch_gpu = CUDA.zeros(T, self.k, batch_size)
    batch_partial = 0
    batch_number = 0

    # p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...", barglyphs=BarGlyphs("[=> ]"))
    p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...")

    for loc in 0:self.step:self.N-self.k
        batch_partial += 1
        if batch_partial == batch_size # batch is filled         
            index_view = @view index[:, batch_number*batch_size+1:(batch_number+1)*batch_size]
            # @time 
            # copyto!(batch_gpu, batch)
            # @show batch |> typeof,  batch_gpu |> typeof
            # @time embeds = encode_batch_m1_cuda(self.re, batch_gpu, 1:self.k, e = e)
            # CUDA.@time 
            encode_batch_m1_cuda!(index_view, self.re, batch_gpu, 1:self.k, e = e)
            # @views index[:, batch_number*batch_size+1:(batch_number+1)*batch_size] .= embeds
            batch_partial = 0
            batch_number += 1
            next!(p)
            # update!(p, position(file))
        else 
            # CUDA.@allowscalar @views copyto!(batch_gpu[batch_partial, :], dna_ref[loc+1:loc+self.k])
            # batch_gpu[batch_partial, :] .= dna_ref[loc+1:loc+self.k]
            batch_gpu[:, batch_partial] .= dna_ref[loc+1:loc+self.k]
        end    
    end
    if batch_partial>0 # last partial batch
        index_view = @view index[:, batch_number*batch_size+1:self.n]
        # copyto!(batch_gpu, batch)
        # batch_view = @view batch[:, 1:batch_partial]
        # batch_gpu_view = @view batch_gpu[1:batch_partial, :]
        batch_gpu_view = @view batch_gpu[:, 1:batch_partial]
        encode_batch_m1_cuda!(index_view, self.re, batch_gpu_view, 1:self.k, e = e)
    end
    
    # copyto!(self.index, index)
    return index
end

function index_m1_cuda_v2_t(self::RopeIndexer, dna_ref::CuArray{T}; e, batch_size = 2^12) where T
    index = CUDA.zeros(ComplexF32, self.n, 4^self.re.s)

    # batch = Array{T}(undef, self.k, batch_size)
    # batch_gpu = CUDA.zeros(T, batch_size, self.k) # transpose
    batch_gpu = CUDA.zeros(T, self.k, batch_size)
    batch_partial = 0
    batch_number = 0

    # p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...", barglyphs=BarGlyphs("[=> ]"))
    p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...")

    for loc in 0:self.step:self.N-self.k
        batch_partial += 1
        if batch_partial == batch_size # batch is filled         
            index_view = @view index[batch_number*batch_size+1:(batch_number+1)*batch_size, :]
            # @time 
            # copyto!(batch_gpu, batch)
            # @show batch |> typeof,  batch_gpu |> typeof
            # @time embeds = encode_batch_m1_cuda(self.re, batch_gpu, 1:self.k, e = e)
            CUDA.@time encode_batch_m1_cuda_t2!(index_view, self.re, batch_gpu, 1:self.k, e = e)
            # @views index[:, batch_number*batch_size+1:(batch_number+1)*batch_size] .= embeds
            batch_partial = 0
            batch_number += 1
            next!(p)
            # update!(p, position(file))
        else 
            # CUDA.@allowscalar @views copyto!(batch_gpu[batch_partial, :], dna_ref[loc+1:loc+self.k])
            # batch_gpu[batch_partial, :] .= dna_ref[loc+1:loc+self.k]
            batch_gpu[:, batch_partial] .= dna_ref[loc+1:loc+self.k]
        end    
    end
    if batch_partial>0 # last partial batch
        index_view = @view index[batch_number*batch_size+1:self.n, :]
        # copyto!(batch_gpu, batch)
        # batch_view = @view batch[:, 1:batch_partial]
        # batch_gpu_view = @view batch_gpu[1:batch_partial, :]
        batch_gpu_view = @view batch_gpu[:, 1:batch_partial]
        encode_batch_m1_cuda_t2!(index_view, self.re, batch_gpu_view, 1:self.k, e = e)
    end
    
    # copyto!(self.index, index)
    return index
end


function index_m1_cuda_nobatch(self::RopeIndexer, dna_ref::AbstractVector{T}; e) where T
    index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)

    # ranges = [loc+1:loc+self.k for loc in 0:self.step:self.N-self.k]  
    # encode_ranges_m1_cuda_v1!(index, self.re, CuArray(dna_ref), CuArray(ranges), e = e)
    ranges_start = [loc+1 for loc in 0:self.step:self.N-self.k] |> CuArray
    ranges_end = [loc+self.k for loc in 0:self.step:self.N-self.k] |> CuArray
    encode_ranges_m1_cuda_v3!(index, self.re, dna_ref, ranges_start, ranges_end, e = e)

    return index
end

# Assumes you can modify or create an in-place version of your encoding function.
# The new function is named encode_batch_m1_cuda! and takes the destination array as the first argument.
# function encode_batch_m1_cuda!(dest::CuArray, re, batch, k_range; e)
#   ... computation ...
#   # Writes results directly into `dest` instead of returning a new array.
# end

function index_m1_cuda_optimized(self::RopeIndexer, dna_ref::AbstractVector{T}; e, batch_size = 1024) where T
    # --- 1. Pre-allocations (done once) ---
    # Final output on GPU (unavoidable allocation)
    index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)
    
    # GPU buffer for a single batch of k-mers (unavoidable allocation)
    gpu_batch = CUDA.zeros(T, self.k, batch_size)
    
    # --- KEY CHANGE: CPU buffer to gather data before transfer ---
    # This one allocation replaces thousands of small GPU allocations.
    cpu_batch = Array{T}(undef, self.k, batch_size)

    # --- 2. Restructured Looping ---
    # Loop over batches of indices, not k-mers, to manage CPU/GPU transfer efficiently
    kmer_indices = 0:self.step:self.N-self.k
    total_kmers = length(kmer_indices)

    for batch_start in 1:batch_size:total_kmers
        # Determine the actual size of the current batch (for the final partial batch)
        batch_end = min(batch_start + batch_size - 1, total_kmers)
        current_batch_size = batch_end - batch_start + 1

        # --- 3. Gather K-mers into the CPU Buffer ---
        # This is a fast, CPU-to-CPU memory operation
        for i in 1:current_batch_size
            # Get the location of the i-th k-mer of this batch
            loc_in_dna = kmer_indices[batch_start + i - 1]
            
            # Copy the k-mer into the CPU batch buffer
            @views cpu_batch[:, i] .= dna_ref[loc_in_dna+1 : loc_in_dna+self.k]
        end

        # --- 4. Bulk Transfer from CPU to GPU ---
        # This is MUCH faster than `batch_size` individual small transfers.
        copyto!(gpu_batch, cpu_batch)

        # --- 5. In-place GPU Computation ---
        # Define the slice of the final index where results will be written
        index_view = @view index[:, batch_start:batch_end]
        
        # Take a view of the part of the gpu_batch we actually filled (for the last batch)
        gpu_batch_view = @view gpu_batch[:, 1:current_batch_size]

        # Call the in-place version of the function. It writes directly to index_view.
        # This eliminates the allocation of the `embeds` temporary variable.
        encode_batch_m1_cuda!(index_view, self.re, gpu_batch_view, 1:self.k; e=e)
    end
    
    return index
end

function (self::RopeIndexer)(dna_ref)
    if self.re.m>1 
        index = parallel_indexer(self, dna_ref)
        index_normal = zeros(ComplexF64, self.re.m * 4^self.re.s, self.n)

        @views for i=1:self.n 
            v = vec(index[:,:,i])
            v /= norm(v)        
            index_normal[:, i] = v 
        end

        return index_normal 
    else
        self.index .*= 0
        self.compact .*= 0
        self.counts .*= 0
        parallel_indexer_m1(self, dna_ref)

        if self.re.c > 0
            @views for i=1:self.n 
                w = self.index[1:self.re.c,i]
                nw = norm(w)
                self.compact[:,i] = w/nw 
            end
        end

        @views for i=1:self.n 
            v = self.index[:,i]
            nv = norm(v)
            self.index[:,i] /= nv
        end

        return 
    end 
end

function parallel_indexer_m1(self::RopeIndexer, dna) # parallel
    # @assert self.re.m==1
    # index = zeros(ComplexF64, 4^self.re.s, self.n)

    nthr = nthreads()
    p = [self.n/nthr*i for i=0:nthr] .|> floor .|> Int
    lrs = [p[i]:p[i+1]-1 for i=1:nthr]		

    @threads for i=1:nthr
        better_indexer_m1(self, dna, lrs[i])
    end

    return 
end

function better_indexer_m1(self::RopeIndexer, dna, lr::UnitRange{Int64}) # partial application
    # lr is the 0-based indices of ropes that we want to compute
    # @assert self.re.m==1
    # index = zeros(ComplexF64, 4^self.re.s, lr)
    # n = length(lr)

    start = 1 + lr.start * self.step
    stop = self.k + lr.stop * self.step 

    ind = reduce((part, digit) -> 4*part+digit, dna[start-1+self.re.s:-1:start]; init=0) # converts a sequence of base-4 digits to a number, little endian
    shift = 2*(self.re.s-1) # used for efficient recomputation of ind in the loop

    @inbounds @fastmath @views for i = self.re.s + start : stop + 1
        l1 = (i-self.re.s-self.k+self.step-1) ÷ self.step
        l1 = max(lr.start,l1)
        l2 = (i-self.re.s-1) ÷ self.step
        l2 = min(lr.stop, l2)

        off = i-self.re.s-(1+self.step*l1) # 0-based offset in the first window
        t = (off > self.k-self.re.s ? 1 : 0) # skip tail in the first window, assume step > s
        for l in l1+t:l2 
            self.index[ind+1,l+1] += self.re.e[1+(off-self.step*(l-l1))]
        end

        if i > stop 
            break
        end

        self.counts[ind+1] += 1
        ind = (ind >> 2) + (dna[i] * (1 << shift))                  
    end

    return 
end

# todo: fix m > 1

function parallel_indexer(self::RopeIndexer, dna) # parallel
    # @assert self.re.m==1
    index = zeros(ComplexF64, self.re.m, 4^self.re.s, self.n)
    e = [exp(2pi*im/self.k*i) for i=1:self.k] # todo: use more efficient method

    nthr = nthreads()
    p = [self.n/nthr*i for i=0:nthr] .|> floor .|> Int
    lrs = [p[i]:p[i+1]-1 for i=1:nthr]		

    @threads for i=1:nthr
        better_indexer(self, dna, lrs[i], index, e)
    end

    return index
end


function better_indexer(self::RopeIndexer, dna, lr::UnitRange{Int64}, index, e) # partial application
    # lr is the 0-based indices of ropes that we want to compute
    # @assert self.re.m==1
    # index = zeros(ComplexF64, 4^self.re.s, lr)
    # n = length(lr)

    start = 1 + lr.start * self.step
    stop = self.k + lr.stop * self.step 

    # e = [exp(2pi*im/self.k*i) for i=1:self.k]

    ind = reduce((part, digit) -> 4*part+digit, dna[start-1+self.re.s:-1:start]; init=0) # converts a sequence of base-4 digits to a number, little endian
    shift = 2*(self.re.s-1) # used for efficient recomputation of ind in the loop

    # @inbounds 
    @fastmath @views for i = self.re.s + start : stop + 1
        l1 = (i-self.re.s-self.k+self.step-1) ÷ self.step
        l1 = max(lr.start,l1)
        l2 = (i-self.re.s-1) ÷ self.step
        l2 = min(lr.stop, l2)

        off = i-self.re.s-(1+self.step*l1) # 0-based offset in the first window
        l0 = (off > self.k-self.re.s ? 1 : 0) # skip tail in the first window, assume step > s
        for l in l1+l0:l2 
            for t=1:self.re.m 
                index[t,ind+1,l+1] += e[mod1((off-self.step*(l-l1))*t, self.k)]
            end
        end

        if i > stop 
            break
        end
        ind = (ind >> 2) + (dna[i] * (1 << shift))                  
    end

    return 
end

function index_cuda_best_new!(self::RopeIndexer; batch_size = 2^12)
    # index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)
    # index_norms = CUDA.zeros(Float32, self.n)

    index = self.cu_index
    index_norms = self.cu_norms
    dna_ref = self.cu_dna_ref

    starts = 1:self.step:self.N-self.k+1 |> collect
    stops = self.re.k:self.step:self.N |> collect
    # if length(stops) < length(starts) 
    #     push!(stops, length(dna_ref))
    # end
    starts = starts .|> UInt32 |> CuArray
    stops = stops .|> UInt32 |> CuArray

    CUDA.@time encode_batch_cuda_all_best!(index, index_norms, self.re, dna_ref, starts = starts, stops = stops)

    return 1
end

"""
Takes a jagged array of dna pieces. In each piece, we select k-mers with the defined step. 
Selected k-mers are specified by their start and stop positions in the jagged array; the arrays all_starts and all_stops are returned. 
Additionally, helper arrays are returned:
    find_piece - given an index in the all_starts array, it returns the index of the corresponding dna piece
    first_kmer - given an index of a dna piece, returns the index in all_starts of the first k-mer taken from that piece

Thus, (i-first_kmer[find_piece[i]])*step = relative position of the k-mer (with index i in all_starts) within a dna piece. 
"""
function select_kmers(dnas::JaggedArray, k::Int; stp=k÷10)
    all_starts = Int[]
    all_stops = Int[]
    find_piece = Int[]
    first_kmer = zeros(Int, length(dnas.starts))
    current_pos = 1
    for i = 1:length(dnas.starts) # loop through pieces
        if dnas.stops[i] - dnas.starts[i] + 1 < k # check if the piece is too short
            continue
        end 
        starts = dnas.starts[i]:stp:dnas.stops[i]-k+1 |> collect # starts of kmers within the piece
        stops = dnas.starts[i]+k-1:stp:dnas.stops[i] |> collect
        if length(stops) < length(starts) # could be less by 1
            push!(stops, dnas.stops[i])
        end
        append!(all_starts, starts)
        append!(all_stops, stops)
        append!(find_piece, i*ones(length(starts)))
        first_kmer[i] = current_pos
        current_pos += length(starts)
    end

    return all_starts, all_stops, find_piece, first_kmer
end

""" 
The main indexing routine. Selects kmers from a jagged array of dna pieces and then computes their encodings. 
"""
function index_cuda_final(re::RopeEncoder, dnas::JaggedArray; stp=re.k÷10, normalize=1, mode = :best)
    # index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)
    # index_norms = CUDA.zeros(Float32, self.n)

    # all_starts = UInt32[]
    # all_stops = UInt32[]
    # find_ind = UInt32[]
    # all_starts = Int[]
    # all_stops = Int[]
    # find_ind = Int[]
    # ind_starts = zeros(Int, length(dnas.stops))
    # ind_sum = 1
    # for i = 1:length(dnas.stops)
    #     if dnas.stops[i] - dnas.starts[i] + 1 < re.k 
    #         continue
    #     end 
    #     starts = dnas.starts[i]:stp:dnas.stops[i]-re.k+1 |> collect
    #     stops = dnas.starts[i]+re.k-1:stp:dnas.stops[i] |> collect
    #     if length(stops) < length(starts) # could be less by 1
    #         push!(stops, dnas.stops[i])
    #     end
    #     append!(all_starts, starts)
    #     append!(all_stops, stops)
    #     append!(find_ind, i*ones(length(starts)))
    #     ind_starts[i] = ind_sum
    #     ind_sum += length(starts)
    # end 

    all_starts, all_stops, find_piece, first_kmer = select_kmers(dnas, re.k; stp=stp)

    ntotal = all_stops |> length # total number of selected kmers

    index = CUDA.zeros(ComplexF32, re.m, 4^re.c, ntotal)
    # index = CUDA.ones(ComplexF32, re.m, 4^re.c, ntotal)
    norms = CUDA.zeros(Float32, re.m, ntotal)

    # @show ntotal
    # @show length(dnas.vect)
    # @show all_starts[end-10:end] .|> Int
    # @show all_stops[end-10:end] .|> Int
    
    # throw("check")

    all_starts = all_starts  |> CuArray
    all_stops = all_stops |> CuArray

    # CUDA.@time 
    encode_batch_cuda_all_best!(index, norms, re, dnas.vect, starts = all_starts[1:end], stops = all_stops[1:end], normalize=normalize, mode=mode)

    return index, norms, all_starts, all_stops, find_piece, first_kmer
end

function index_cuda_best!(self::RopeIndexer; batch_size = 2^12)
    # index = CUDA.zeros(ComplexF32, 4^self.re.s, self.n)
    # index_norms = CUDA.zeros(Float32, self.n)

    index = self.cu_index
    index_norms = self.cu_norms
    dna_ref = self.cu_dna_ref
    e = self.cu_e

    batch_gpu = CUDA.zeros(UInt8, self.k, batch_size)
    batch_partial = 0
    batch_number = 0

    p = Progress(self.n ÷ batch_size, dt=1, desc="Generating index...")

    for loc in 0:self.step:self.N-self.k
        batch_partial += 1
        if batch_partial == batch_size # batch is filled         
            index_view = @view index[:, batch_number*batch_size+1:(batch_number+1)*batch_size]
            index_norms_view = @view index_norms[batch_number*batch_size+1:(batch_number+1)*batch_size]
            # @time 
            # copyto!(batch_gpu, batch)
            # @show batch |> typeof,  batch_gpu |> typeof
            # @time embeds = encode_batch_m1_cuda(self.re, batch_gpu, 1:self.k, e = e)
            # CUDA.@time 
            encode_batch_cuda_best!(index_view, index_norms_view, self.re, batch_gpu, e = e)
            # @views index[:, batch_number*batch_size+1:(batch_number+1)*batch_size] .= embeds
            batch_partial = 0
            batch_number += 1
            next!(p)
            # update!(p, position(file))
        else 
            # CUDA.@allowscalar @views copyto!(batch_gpu[batch_partial, :], dna_ref[loc+1:loc+self.k])
            # batch_gpu[batch_partial, :] .= dna_ref[loc+1:loc+self.k]
            @views batch_gpu[:, batch_partial] .= dna_ref[loc+1:loc+self.k]
        end    
    end
    if batch_partial>0 # last partial batch
        index_view = @view index[:, batch_number*batch_size+1:self.n]
        index_norms_view = @view index_norms[batch_number*batch_size+1:self.n]
        # copyto!(batch_gpu, batch)
        # batch_view = @view batch[:, 1:batch_partial]
        # batch_gpu_view = @view batch_gpu[1:batch_partial, :]
        batch_gpu_view = @view batch_gpu[:, 1:batch_partial]
        encode_batch_cuda_best!(index_view, index_norms_view, self.re, batch_gpu_view, e = e)
    end
    
    # copyto!(self.index, index)
    # return index, index_norms
    return 1
end

end