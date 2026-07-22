# __precompile__(true)

module RopeEncoders

export RopeEncoder, encode_batch_m1_cuda, encode_batch_m1_cuda!
export encode_ranges_m1_cuda!, encode_ranges_m1_cuda_v1!, encode_reads_batch_m1_cuda!, encode_reads_batch_m1_cuda
export encode_ranges_m1_cuda_v3!, encode_batch_m1_cuda_t!, encode_batch_m1_cuda_t2!
export encode_m1_cuda_new, encode_m1_cuda_new!
export encode_batch_cuda_best!, encode_batch_cuda_best, encode_batch_cuda_all_best!, encode_batch_cuda_all_best
export dim 

using Base.Cartesian
using Base.Threads
using Random
using LinearAlgebra
# using StaticArrays, Setfield

# using Chain

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


function (self::RopeEncoder)(dna; complex=true, invariant=true, pure=false, parallel=false) 
    # dna = [0,1,2,3,0,2,1...]
    @assert length(dna)==self.k 

    # todo: compat mode; requires permutation

    if !parallel
        if self.m==1
            embeds = basic_encoding_m1_v2(self, dna, 1:self.k)
            # embeds = basic_encoding(self, dna, 1:self.k)
        else
            embeds = basic_encoding(self, dna, 1:self.k)
        end
    else
        if self.m==1
            embeds = parallel_encoding_m1_v2(self, dna)
            # embeds = parallel_encoding(self, dna)
        else
            embeds = parallel_encoding(self, dna)
        end
    end
    
    if pure 
        return embeds
    end
    
    if self.c > 0 # a simple compact case
        if self.m > 1 # todo
            ret = Array{ComplexF64}(undef, self.m, self.c)
            @views for i in 1:self.c
                ret[:, i] = embeds[:, self.perm[i]]
            end 
            return ret/norm(vec(ret))
        else 
            # ret = Array{ComplexF64}(undef, self.c)
            # ret = Array{ComplexF64}(undef, 4^self.s)
            # for i in 1:self.c
            #     ret[i] = embeds[self.perm[i]]
            # end 

            # ret = embeds
            # ret .= @view embeds[self.perm]
            # return ret/norm(ret)
            if complex 
                rope = embeds/norm(embeds)
                @views rope_c = embeds[1:self.c]/norm(embeds[1:self.c])
                return rope, rope_c
            else
                res = reim(embeds)
                rembeds = hcat(res[1], res[2])' #|> vec

                rope = rembeds/norm(rembeds)
                @views rope_c = rembeds[1:2*self.c]/norm(rembeds[1:2*self.c])
                return rope, rope_c
            end 
        end
    end

    result = vec(embeds) 
    rnorm = norm(result)     
    
    if invariant
        # @views for t=1:self.m
        #     nr = norm(embeds[t,:])
        #     if nr > 0 
        #         embeds[t,:]/=nr 
        #     end
        # end

        return embeds/rnorm
    end 

    result /= rnorm

    if !complex # use the real and imag parts for each number instead; we don't normalize it again since this mode goes to angular encoding anyway
        res = reim(result)
        result = hcat(res[1], res[2])' |> vec
    end
    
    return result
end

"""
dna - a dna read is an array of elements 0:3

version for use when m=1 

range is 1-based
"""
function encode_single_m1(self::RopeEncoder, dna::AbstractVector{T}, range::UnitRange{Int}) where T
    # @assert self.m==1
    embed = zeros(ComplexF64, 4^self.s)

    smer = 0 # 0-based big endian number representations of the current s-mer in the loop

    @views for i = range.start : range.start+self.s-1
        smer <<= 2
        smer += dna[i] 
    end

    bmask = 4^self.s - 1
    ie = range.start # exponent power 
    @inbounds @views for i = self.s + range.start : range.stop
        smer += 1
        embed[smer] += self.e[ie]
        smer -= 1
        smer <<= 2
        smer &= bmask 
        smer += dna[i] 
        ie += 1
    end
    @views for i = range.stop + 1 : range.stop + self.s
        smer += 1
        embed[smer] += self.e[ie]
        smer -= 1
        if i >= self.k 
            break
        end
        smer <<= 2
        smer &= bmask 
        smer += dna[i] 
        ie += 1
    end

    return embed
end

basic_encoding_m1_v2 = encode_single_m1

"""
dnas - a row major matrix of dna reads where each dna is an array of elements 0:3

version for use when m=1 

range is 1-based
"""
function encode_batch_m1(self::RopeEncoder, dnas::AbstractMatrix{T}, range::UnitRange{Int}) where T 
    n = size(dnas)[1] # number of reads
    embeds = zeros(ComplexF64, 4^self.s, n)

    smers = zeros(Int, n) # 0-based big endian number representations of current s-mers in the loop

    @views for i = range.start : range.start+self.s-1
        smers .<<= 2
        smers .+= dnas[:, i] 
    end

    bmask = 4^self.s - 1
    ie = range.start # exponent power 
    cart = CartesianIndex.(smers, 1:n)
    @inbounds @views for i = self.s + range.start : range.stop
        smers .+= 1
        cart .= CartesianIndex.(smers, 1:n)
        embeds[cart] .+= self.e[ie]
        smers .-= 1
        smers .<<= 2
        smers .&= bmask 
        smers .+= dnas[:, i] 
        ie += 1
    end
    @views for i = range.stop + 1 : range.stop + self.s
        smers .+= 1
        cart .= CartesianIndex.(smers, 1:n)
        embeds[cart] .+= self.e[ie]
        smers .-= 1
        if i >= self.k 
            break
        end
        smers .<<= 2
        smers .&= bmask
        smers .+= dnas[:, i] 
        ie += 1 
    end

    return embeds
end

function basic_encoding(self::RopeEncoder, dna, range::UnitRange{Int}) # partial application    
    embeds = zeros(ComplexF64, self.m, 4^self.s)

    ind = reduce((part, digit) -> 4*part+digit, dna[range.start-1+self.s:-1:range.start]; init=0) # converts a sequence of base-4 digits to a number, little endian
    shift = 2*(self.s-1) # used for efficient recomputation of ind in the loop

    # todo: use precomputed exponents, add compact logic

    v = [exp(2pi*im/self.k*i) for i=1:self.m]
    vi = [exp((range.start-1)*2pi*im/self.k*i) for i=1:self.m]
    # @inbounds 
    @views for i = self.s + range.start : range.stop
        embeds[:, ind+1] .+= vi         
        vi .*= v
        ind = (ind >> 2) + (dna[i] * (1 << shift))                
    end
    @views for i = range.stop + 1 : range.stop + self.s
        embeds[:, ind+1] .+= vi         
        if i >= self.k 
            break
        end
        vi .*= v
        # ind = (ind >> 2) + (dna[mod1(i, self.k)] << shift)                 
        ind = (ind >> 2) + (dna[i] * (1 << shift))                 
    end

    return embeds
end

function basic_encoding_v2(self::RopeEncoder, dna, range::UnitRange{Int}) # partial application    
    embeds = zeros(ComplexF64, 4^self.s, self.m)

    ind = reduce((part, digit) -> 4*part+digit, dna[range.start-1+self.s:-1:range.start]; init=0) # converts a sequence of base-4 digits to a number, little endian
    shift = 2*(self.s-1) # used for efficient recomputation of ind in the loop

    # todo: use precomputed exponents, add compact logic

    # v = [exp(2pi*im/self.k*i) for i=1:self.m]
    # vi = [exp((range.start-1)*2pi*im/self.k*i) for i=1:self.m]
    ie = range.start-1 # exp power 
    @inbounds @views for i = self.s + range.start : range.stop
        for t = 1:self.m
            # embeds[ind+1, t] .+= ω^(ie*t)
            id, ir = divrem(ie*t, 2^self.elog)
            # id = (ie*t) >> self.elog
            # ir = (ie*t) & (self.elog-1)
            embeds[ind+1, t] += self.ediv[id+1]*self.erem[ir+1]
        end
        ie += 1
        ind = (ind >> 2) + (dna[i] * (1 << shift))                
    end
    @views for i = range.stop + 1 : range.stop + self.s
        for t = 1:self.m
            # embeds[ind+1, t] .+= ω^(ie*t)
            # id, ir = divrem(ie*t, 2^elog)
            id = ie*t >> self.elog
            ir = ie*t & (self.elog-1)
            embeds[ind+1, t] += self.ediv[id+1]*self.erem[ir+1]
        end       
        if i >= self.k 
            break
        end
        ie += 1
        # ind = (ind >> 2) + (dna[mod1(i, self.k)] << shift)                 
        ind = (ind >> 2) + (dna[i] * (1 << shift))                 
    end

    return embeds
end

function basic_encoding_m1(self::RopeEncoder, dna, range::UnitRange{Int}) # more efficient if m=1
    # @assert self.m==1
    embeds = zeros(ComplexF64, 4^self.s)

    ind = reduce((part, digit) -> 4*part+digit, dna[range.start-1+self.s:-1:range.start]; init=0) # converts a sequence of base-4 digits to a number, little endian
    shift = 2*(self.s-1) # used for efficient recomputation of ind in the loop

    v = exp(2pi*im/self.k)
    vi = exp((range.start-1)*2pi*im/self.k)
    # @inbounds 
    @views for i = self.s + range.start : range.stop
        embeds[ind+1] += vi         
        vi *= v
        ind = (ind >> 2) + (dna[i] * (1 << shift))  
    end
    @views for i = range.stop + 1 : range.stop + self.s
        embeds[ind+1] += vi         
        if i >= self.k 
            break
        end
        vi *= v
        # ind = (ind >> 2) + (dna[mod1(i, self.k)] << shift)                 
        ind = (ind >> 2) + (dna[i] * (1 << shift))               
    end

    return embeds
end



function parallel_encoding(self::RopeEncoder, dna) 
    # embeds = zeros(ComplexF64, self.m, 4^self.s)

    nthr = nthreads()
    p = [self.k/nthr*i for i=0:nthr] .|> floor .|> Int
    ranges = [p[i]+1:p[i+1] for i=1:nthr]		

    embeds_parts = [zeros(ComplexF64, self.m, 4^self.s) for i=1:nthr]
    @threads for i=1:nthr
        embeds_parts[i] = basic_encoding(self, dna, ranges[i])
    end
    embeds = sum(embeds_parts)

    return embeds
end

function parallel_encoding_m1(self::RopeEncoder, dna) # more efficient if m=1
    # embeds = zeros(ComplexF64, self.m, 4^self.s)

    nthr = nthreads()
    p = [self.k/nthr*i for i=0:nthr] .|> floor .|> Int
    ranges = [p[i]+1:p[i+1] for i=1:nthr]		

    embeds_parts = [zeros(ComplexF64, 4^self.s) for i=1:nthr]
    @threads for i=1:nthr
        embeds_parts[i] = basic_encoding_m1(self, dna, ranges[i])
    end
    embeds = sum(embeds_parts)

    return embeds
end


# =============================== CUDA ===============================

using CUDA 
# CUDA.allowscalar(true) 

# uses re.m 
function encode_batch_cuda_all_best(
    re::RopeEncoder, 
    dnas::CuArray
    ;
    # e::CuArray,
    # start = 1,
    # stop = re.k,
    starts::CuArray,
    stops::CuArray,
    normalize = 1,
    mode = :best,
    max_shm_size = 48
)
    # n = size(dnas, 2)
    n = length(starts)

    # dest = CUDA.zeros(ComplexF32, re.m, 4^re.s, n)
    # if double
    #     dest = CUDA.zeros(ComplexF32, re.m, 2*(4^re.c), n)
    # else


    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, n) 
    # dest = CUDA.ones(ComplexF32, re.m, 4^re.c, n) 

    # end 
    # dest = CUDA.zeros(ComplexF32, 4^re.s, re.m, n)
    # dest = CUDA.zeros(ComplexF32, n, re.m, 4^re.s)
    dest_norms = CUDA.zeros(Float32, re.m, n)  

    encode_batch_cuda_all_best!(
        dest,
        dest_norms,
        re, 
        dnas, 
        # e = e,
        starts = starts,
        stops = stops, 
        normalize = normalize,
        mode = mode,
        max_shm_size = max_shm_size
    )
    return dest, dest_norms
end

function encode_batch_cuda_all_best!(
    dest::CuArray{ComplexF32}, 
    dest_norms::CuArray{Float32}, 
    re::RopeEncoder, 
    dnas::CuArray
    ;
    # e::CuArray,
    # start = 1,
    # stop = re.k,
    starts::CuArray,
    stops::CuArray,
    normalize = 1,
    mode = :best,
    max_shm_size = 48
)  
    n = length(starts) # Number of dnas
    
    begin 
        # threads = (4, 4, 32)
        # blocks = (1, 1, cld(n, 32))
        # threads = (re.m, max(1, (32÷re.m)*4))
        # blocks = (1, cld(n, max(1, (32÷re.m)*4)))

        # trace=true
        # res = CUDA.@profile trace=true begin 
            # @cuda blocks=blocks threads=threads shmem=2^(8+2*re.s)*re.m kernel_cool_cache(
        if mode==:best
            threads = (re.m, max(1, (32÷re.m)*16))
            blocks = (1, cld(n, max(1, (32÷re.m)*16)))

            @cuda blocks=blocks threads=threads kernel(
                dest,            
                dest_norms,
                dnas, 
                re.s,
                re.m,
                re.c,
                re.k, 
                starts,
                stops,
                normalize
            )
        elseif mode==:freq 
            threads = (1, 32*16)
            blocks = (1, cld(n, 32*16))

            @cuda blocks=blocks threads=threads kernel_freq(
                dest,            
                dest_norms,
                dnas, 
                re.s,
                re.m,
                re.c,
                re.k, 
                starts,
                stops,
                normalize
            )
        elseif mode==:default
            threads = (1, 32*16)
            blocks = (1, cld(n, 32*16))

            @cuda blocks=blocks threads=threads kernel_default(
                dest,            
                dest_norms,
                dnas, 
                re.s,
                re.m,
                re.c,
                re.k, 
                starts,
                stops,
                normalize
            )
        end
            # CUDA.synchronize()
        # end    
    end

    # display(res)
    
    return dest, dest_norms
end

"""
Here dnas is not a matrix but contiguous array. We use offsets (range_start, range_stop) to track them. 
"""
@generated function kernel_freq(embeds, embeds_norms, dnas, s, m, c, k, ranges_start, ranges_stop, normalize)
    quote 
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        idm = threadIdx().x

        if idy > length(ranges_start)
            return
        end
        bmask = UInt32(4^s - 1)
        cbmask = UInt32(4^c - 1)
        mixer = 0x9E3779B1  # Based on the golden ratio

        smer = UInt32(0)
        # @inbounds 
        for i = ranges_start[idy] : ranges_start[idy] + s - UInt32(1)
            smer <<= 2
            smer += dnas[i]
        end

        # sc2 = 2*(s-c)
        for i = s + ranges_start[idy] : ranges_stop[idy]

            # csmer = smer+1
            lower = smer & cbmask
            upper = smer >> 2c
            scramble_key = (upper * mixer) & cbmask
            csmer = (lower ⊻ scramble_key) + 1

            embeds[idm, csmer, idy] += 1

            smer <<= 2
            smer &= bmask
            smer += dnas[i]
        end
        lower = smer & cbmask
        upper = smer >> 2c
        scramble_key = (upper * mixer) & cbmask
        csmer = (lower ⊻ scramble_key) + 1
        
        embeds[idm, csmer, idy] += 1

        # begin # Ignore repeating letters
            # allones = UInt32(0)
            # for i = 0:c-1
            #     allones += UInt32(4^i)
            # end

            # csmerA = 1
            # csmerC = allones + 1
            # csmerG = UInt32(2) * allones + 1
            # csmerT = UInt32(3) * allones + 1

            # embeds[idm, csmerA, idy] = 0.0
            # embeds[idm, csmerC, idy] = 0.0
            # embeds[idm, csmerG, idy] = 0.0
            # embeds[idm, csmerT, idy] = 0.0
        # end

        for i in 1:4^c               
            embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
        end
        
        embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
        # embeds_norms[idm, idy] += eps()
        embeds_norms[idm, idy] += 1e-16

        for i in 1:4^c
            embeds[idm, i, idy] /= embeds_norms[idm, idy]
        end        

        return
    end
end



"""
Here dnas is not a matrix but contiguous array. We use offsets (range_start, range_stop) to track them. 
"""
@generated function kernel(embeds, embeds_norms, dnas, s, m, c, k, ranges_start, ranges_stop, normalize)
# @generated function kernel_cool(embeds, embeds_norms, dnas, s, m, k, e, range_start, range_stop, normalize)
    quote 
        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        # x dim is for m factor
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        # idy = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        # idm = threadIdx().x+4*(threadIdx().y-1)
        idm = threadIdx().x

        # golden_ratio = (sqrt(5)-1)/2 
        # chaos = exp(golden_ratio * 2π*im)

        # if idy > size(dnas, 2)
        # if idx > size(dnas, 1)
        if idy > length(ranges_start)
            return
        end
        bmask = UInt32(4^s - 1)
        cbmask = UInt32(4^c - 1)
        cfactor = UInt32(2654435769)

        mixer = 0x9E3779B1  # Based on the golden ratio

        # range_start = 1
        # range_stop = k-s+1
        # CUDA.@cuprintln(idm, " ", idy, " ", ranges_start[1])
        # CUDA.@cuprintln()
        # CUDA.@cuprintf("CUDA kernel %d %d %d\n", idm, idy, ranges_start[end])

        smer = UInt32(0)
        # @inbounds 
        for i = ranges_start[idy] : ranges_start[idy] + s - UInt32(1)
            smer <<= 2
            # smer += dnas[i, idy]
            smer += dnas[i]
            # smer += dnas[idy][i]
            # smer += dnas[idx, i]
        end

        # CUDA.@cuprintln("ok1")

        # ω_x = exp(idm * 2π*im/(k)) 
        # ω_x = exp((4idm+1)/4 * 2π*im/(k)) # repeats fixed
        # ω_x = exp((idm*(idm+1)+1)/(idm+1) * 2π*im/(k)) # repeats fixed
        ω_x = exp((2*(idm-1)/(m-1)+1) * 2π*im/(k)) # repeats fixed
        if m==1 
            ω_x = exp(2π*im/(k))
        end 

        # ω_i = ω_x^(range_start - 1)
        # ω_i = ω_x^(range_start - 1) # this is used if we split encoding into parts, not needed in practice
        ω_i = 1
        # @inbounds 
        # chaos_i = chaos
        sc2 = 2*(s-c)
        for i = s + ranges_start[idy] : ranges_stop[idy]
            # embeds[threadIdx().x, smer+1, idy] += ω_i
            # csmer = ((smer * cfactor) >> 24) & cbmask + 1

            # csmer = smer+1
            lower = smer & cbmask
            upper = smer >> 2c
            scramble_key = (upper * mixer) & cbmask
            csmer = (lower ⊻ scramble_key) + 1

            # csmer = ((smer ⊻ (smer >> sc2)) & cbmask) + 1
            embeds[idm, csmer, idy] += ω_i #* chaos_i
            # embeds[idm, smer+1, idy] += ω_i
            # @nexprs $M mi -> (ω_i_0 *= ω_i; embeds[mi, smer+1, idx] += ω_i_0)
            smer <<= 2
            smer &= bmask
            # smer += dnas[i, idy]
            smer += dnas[i]
            # smer += dnas[idy][i]
            ω_i *= ω_x
            # chaos_i *= chaos
        end
        # csmer = smer+1
        lower = smer & cbmask
        upper = smer >> 2c
        scramble_key = (upper * mixer) & cbmask
        csmer = (lower ⊻ scramble_key) + 1
        
        embeds[idm, csmer, idy] += ω_i #* chaos_i
        # embeds[idm, smer+1, idy] += ω_i

        # CUDA.@cuprintln("ok2")

        begin # Ignore repeating letters
            allones = UInt32(0)
            for i = 0:c-1
                allones += UInt32(4^i)
            end

            # csmerA = ((1 ⊻ (1 >> sc2)) & cbmask) + 1
            # csmerC = ((allones ⊻ (allones >> sc2)) & cbmask) + 1
            # csmerG = (((UInt32(2) * allones) ⊻ ((UInt32(2) * allones) >> sc2)) & cbmask) + 1
            # csmerT = (((UInt32(3) * allones) ⊻ ((UInt32(3) * allones) >> sc2)) & cbmask) + 1

            csmerA = 1
            csmerC = allones + 1
            csmerG = UInt32(2) * allones + 1
            csmerT = UInt32(3) * allones + 1

            # embeds[idm, csmerA, idy] = 0.0
            # embeds[idm, csmerC, idy] = 0.0
            embeds[idm, csmerG, idy] = 0.0
            # embeds[idm, csmerT, idy] = 0.0
        end

        # @inbounds for i = range_stop[idy] + 1 : range_stop[idy] + s # todo: should we loop till range_stop+s-1 ? Seems no, as we don't use the last dnas[i, idx]
        #     # embeds[threadIdx().x, smer+1, idy] += ω_i
        #     embeds[idm, smer+1, idy] += ω_i
        #     if i >= k
        #         break
        #     end

        #     smer <<= 2
        #     smer &= bmask            
        #     # smer += dnas[i, idy]
        #     smer += dnas[i]
        #     # smer += dnas[idy][i]
        #     ω_i *= ω_x
        # end

        # norm_embeds_idx = zeros(m)
        # @inbounds 
        # for i in 1:4^s

        # for i in 1:4^c
        #     # @nexprs $M mi -> (embeds_norms[mi, idx] += abs2(embeds[mi, i, idx]))
        #     # embeds_norms[threadIdx().x, idy] += abs2(embeds[threadIdx().x, i, idy])
        #     embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
        # end

        # CUDA.@cuprintln("ok3")

        # embeds_norms[threadIdx().x, idy] = sqrt(embeds_norms[threadIdx().x, idy])
        # embeds_norms[threadIdx().x, idy] += eps()

        # embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
        # embeds_norms[idm, idy] += eps()

        # CUDA.@cuprintln("ok4")
        # @nexprs $M mi -> (embeds_norms[mi, idx] = sqrt(embeds_norms[mi, idx]); embeds_norms[mi, idx] += eps())
        if normalize==0
            for i in 1:4^c               
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end

            sync_threads()
            if idm==m 
                for i_m in 2:m
                    embeds_norms[1, idy] += embeds_norms[i_m, idy]
                end
            end
            embeds_norms[1, idy] = sqrt(embeds_norms[1, idy])
            embeds_norms[1, idy] += 1f-7

            sync_threads()
            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= embeds_norms[1, idy]
            end
        elseif normalize==1
            # @inbounds 
            # for i in 1:4^s
            for i in 1:4^c               
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end
            
            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            # embeds_norms[idm, idy] += eps()
            embeds_norms[idm, idy] += 1f-7

            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end        
        elseif normalize==2 # another version - each smer is treated equally
            # for i in 1:4^s
            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= abs(embeds[idm, i, idy]) + eps()
                # embeds[idm, i, idy] /= 2^s #* sqrt(blockDim().x)
                # embeds[idm, i, idy] /= 2^c #* sqrt(blockDim().x)
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end         

            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            embeds_norms[idm, idy] += eps()

            for i in 1:4^c               
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end 
        elseif normalize==3 # use sqrt 
            # for i in 1:4^s
            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]                
                embeds[idm, i, idy] /= sqrt(abs(embeds[idm, i, idy]) + eps())
                # embeds[idm, i, idy] /= 2^s #* sqrt(blockDim().x)
                # embeds[idm, i, idy] /= 2^c #* sqrt(blockDim().x)
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end           

            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            embeds_norms[idm, idy] += eps()

            for i in 1:4^c               
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end 
        end

        return
    end
end



"""
Here dnas is not a matrix but contiguous array. We use offsets (range_start, range_stop) to track them. 
"""
@generated function kernel_default(embeds, embeds_norms, dnas, s, m, c, k, ranges_start, ranges_stop, normalize)
# @generated function kernel_cool(embeds, embeds_norms, dnas, s, m, k, e, range_start, range_stop, normalize)
    quote 
        # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        # x dim is for m factor
        idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        # idy = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        # idm = threadIdx().x+4*(threadIdx().y-1)
        idm = threadIdx().x

        # golden_ratio = (sqrt(5)-1)/2 
        # chaos = exp(golden_ratio * 2π*im)

        # if idy > size(dnas, 2)
        # if idx > size(dnas, 1)
        if idy > length(ranges_start)
            return
        end
        bmask = UInt32(4^s - 1)
        cbmask = UInt32(4^c - 1)
        cfactor = UInt32(2654435769)

        mixer = 0x9E3779B1  # Based on the golden ratio

        # range_start = 1
        # range_stop = k-s+1
        # CUDA.@cuprintln(idm, " ", idy, " ", ranges_start[1])
        # CUDA.@cuprintln()
        # CUDA.@cuprintf("CUDA kernel %d %d %d\n", idm, idy, ranges_start[end])

        smer = UInt32(0)
        # @inbounds 
        for i = ranges_start[idy] : ranges_start[idy] + s - UInt32(1)
            smer <<= 2
            # smer += dnas[i, idy]
            smer += dnas[i]
            # smer += dnas[idy][i]
            # smer += dnas[idx, i]
        end

        # CUDA.@cuprintln("ok1")

        ω_x = exp(idm * 2π*im/(k)) 
        # ω_x = exp((4idm+1)/4 * 2π*im/(k)) # repeats fixed
        # ω_x = exp((idm*(idm+1)+1)/(idm+1) * 2π*im/(k)) # repeats fixed
        # ω_x = exp((2*(idm-1)/(m-1)+1) * 2π*im/(k)) # repeats fixed

        # ω_i = ω_x^(range_start - 1)
        # ω_i = ω_x^(range_start - 1) # this is used if we split encoding into parts, not needed in practice
        ω_i = 1
        # @inbounds 
        # chaos_i = chaos
        sc2 = 2*(s-c)
        for i = s + ranges_start[idy] : ranges_stop[idy]
            # embeds[threadIdx().x, smer+1, idy] += ω_i
            # csmer = ((smer * cfactor) >> 24) & cbmask + 1

            csmer = smer+1
            # lower = smer & cbmask
            # upper = smer >> 2c
            # scramble_key = (upper * mixer) & cbmask
            # csmer = (lower ⊻ scramble_key) + 1

            # csmer = ((smer ⊻ (smer >> sc2)) & cbmask) + 1
            embeds[idm, csmer, idy] += ω_i #* chaos_i
            # embeds[idm, smer+1, idy] += ω_i
            # @nexprs $M mi -> (ω_i_0 *= ω_i; embeds[mi, smer+1, idx] += ω_i_0)
            smer <<= 2
            smer &= bmask
            # smer += dnas[i, idy]
            smer += dnas[i]
            # smer += dnas[idy][i]
            ω_i *= ω_x
            # chaos_i *= chaos
        end
        csmer = smer+1
        # lower = smer & cbmask
        # upper = smer >> 2c
        # scramble_key = (upper * mixer) & cbmask
        # csmer = (lower ⊻ scramble_key) + 1
        
        embeds[idm, csmer, idy] += ω_i #* chaos_i
        # embeds[idm, smer+1, idy] += ω_i

        # CUDA.@cuprintln("ok2")

        begin # Ignore repeating letters
            allones = UInt32(0)
            for i = 0:c-1
                allones += UInt32(4^i)
            end

            # csmerA = ((1 ⊻ (1 >> sc2)) & cbmask) + 1
            # csmerC = ((allones ⊻ (allones >> sc2)) & cbmask) + 1
            # csmerG = (((UInt32(2) * allones) ⊻ ((UInt32(2) * allones) >> sc2)) & cbmask) + 1
            # csmerT = (((UInt32(3) * allones) ⊻ ((UInt32(3) * allones) >> sc2)) & cbmask) + 1

            csmerA = 1
            csmerC = allones + 1
            csmerG = UInt32(2) * allones + 1
            csmerT = UInt32(3) * allones + 1

            # embeds[idm, csmerA, idy] = 0.0
            # embeds[idm, csmerC, idy] = 0.0
            # embeds[idm, csmerG, idy] = 0.0
            # embeds[idm, csmerT, idy] = 0.0
        end

        # @inbounds for i = range_stop[idy] + 1 : range_stop[idy] + s # todo: should we loop till range_stop+s-1 ? Seems no, as we don't use the last dnas[i, idx]
        #     # embeds[threadIdx().x, smer+1, idy] += ω_i
        #     embeds[idm, smer+1, idy] += ω_i
        #     if i >= k
        #         break
        #     end

        #     smer <<= 2
        #     smer &= bmask            
        #     # smer += dnas[i, idy]
        #     smer += dnas[i]
        #     # smer += dnas[idy][i]
        #     ω_i *= ω_x
        # end

        # norm_embeds_idx = zeros(m)
        # @inbounds 
        # for i in 1:4^s

        # for i in 1:4^c
        #     # @nexprs $M mi -> (embeds_norms[mi, idx] += abs2(embeds[mi, i, idx]))
        #     # embeds_norms[threadIdx().x, idy] += abs2(embeds[threadIdx().x, i, idy])
        #     embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
        # end

        # CUDA.@cuprintln("ok3")

        # embeds_norms[threadIdx().x, idy] = sqrt(embeds_norms[threadIdx().x, idy])
        # embeds_norms[threadIdx().x, idy] += eps()

        # embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
        # embeds_norms[idm, idy] += eps()

        # CUDA.@cuprintln("ok4")
        # @nexprs $M mi -> (embeds_norms[mi, idx] = sqrt(embeds_norms[mi, idx]); embeds_norms[mi, idx] += eps())
        if normalize==0
            for i in 1:4^c               
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end

            if idm==m 
                for i_m in 2:m
                    embeds_norms[1, idy] += embeds_norms[i_m, idy]
                end
            end
            embeds_norms[1, idy] = sqrt(embeds_norms[1, idy])
            embeds_norms[1, idy] += 1e-16

            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= embeds_norms[1, idy]
            end
        elseif normalize==1
            # @inbounds 
            # for i in 1:4^s
            for i in 1:4^c               
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end
            
            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            # embeds_norms[idm, idy] += eps()
            embeds_norms[idm, idy] += 1e-16

            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end        
        elseif normalize==2 # another version - each smer is treated equally
            # for i in 1:4^s
            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]
                embeds[idm, i, idy] /= abs(embeds[idm, i, idy]) + eps()
                # embeds[idm, i, idy] /= 2^s #* sqrt(blockDim().x)
                # embeds[idm, i, idy] /= 2^c #* sqrt(blockDim().x)
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end         

            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            embeds_norms[idm, idy] += eps()

            for i in 1:4^c               
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end 
        elseif normalize==3 # use sqrt 
            # for i in 1:4^s
            for i in 1:4^c
                # @nexprs $M mi -> (embeds[mi, i, idx] /= embeds_norms[mi, idx])
                # embeds[threadIdx().x, i, idy] /= embeds_norms[threadIdx().x, idy]                
                embeds[idm, i, idy] /= sqrt(abs(embeds[idm, i, idy]) + eps())
                # embeds[idm, i, idy] /= 2^s #* sqrt(blockDim().x)
                # embeds[idm, i, idy] /= 2^c #* sqrt(blockDim().x)
                embeds_norms[idm, idy] += abs2(embeds[idm, i, idy])
            end           

            embeds_norms[idm, idy] = sqrt(embeds_norms[idm, idy])
            embeds_norms[idm, idy] += eps()

            for i in 1:4^c               
                embeds[idm, i, idy] /= embeds_norms[idm, idy]
            end 
        end

        return
    end
end




end