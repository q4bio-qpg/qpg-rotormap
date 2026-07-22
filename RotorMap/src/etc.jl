# GPU memory helper 

GC.gc(); CUDA.reclaim(); CUDA.memory_status()


# useful pieces of discarded code

cache_3 = @cuDynamicSharedMem(ComplexF32, 32, 64) 
cache_1 = @cuStaticSharedMem(ComplexF32, (32, 64)) 

@nexprs $M mi -> (ω_i_0 *= ω_i; embeds[mi, smer+1, idx] += ω_i_0)
@nexprs 3 mi -> (ω_i_0 *= ω_i; cache_{mi}[threadIdx().x, smer+1] += ω_i_0)

@nexprs $M mi -> cache_{mi} = @cuDynamicSharedMem(ComplexF32, 32, 4^s)

for j = 1:bmask+1 
    @nexprs $M mi -> cache_{mi}[threadIdx().x, j] = 0.0
end

@generated function kernel_cool_cached(embeds, embeds_norms, dnas, s, ::Val{M}, k, e, range_start, range_stop, normalize) where M
# @generated function kernel_cool(embeds, embeds_norms, dnas, s, m, k, e, range_start, range_stop, normalize)
    quote 
    end
end

CUDA.@cuprintln()
CUDA.@cuprintf("CUDA kernel %d %d %d\n", idm, idy, ranges_start[end])

norm_sums = @cuDynamicSharedMem(Float32, (blockDim().x, blockDim().y))  

stride = blockDim().x ÷ 2
while stride > 0
    if idm <= stride
        norm_sums[idm, threadIdx().y] += norm_sums[idm + stride, threadIdx().y]
    end
    sync_threads()
    stride ÷= 2
end
norm_sum = norm_sums[1, threadIdx().y]

@cuda blocks=blocks threads=threads shmem=2^(8+2*re.s)*re.m kernel_cool_cache()


open("ranges_pos.json", "w") do io
    show(io, "text/plain", ranges)
end