@enum RoPEVersion begin
    default
    finetuned
end

function RoPE(dna::Vector{UInt8},s::Int,m::Int,t::Int; version::RoPEVersion=default)
        N = length(dna)
        c = zeros(ComplexF64, m, 4^t)

        if version == default || m==1
            ωs = [exp(k * 2π*im/N) for k=1:m]
        else
            ωs = [exp((2*(k-1)/(m-1)+1) * 2π*im/N) for k=1:m]
        end

        bmask = UInt32(4^s - 1)
        tbmask = UInt32(4^t - 1)
        mixer = 0x9E3779B1 # used for 4^s-to-4^t scrambled mapping 

        smer = UInt32(0) # integer encoding of an s-mer
        for i = 1:s
            smer <<= 2
            smer += dna[i]
        end

        ωs_i = ones(ComplexF64, m)
        @inbounds for i = s+1:length(dna)
            lower = smer & tbmask
            upper = smer >> 2t
            scramble_key = (upper * mixer) & tbmask
            tsmer = (lower ⊻ scramble_key) + 1 
            c[:, tsmer] .+= ωs_i

            smer <<= 2
            smer &= bmask
            smer += dna[i]
            ωs_i .*= ωs
        end
        begin # the last s-mer
            lower = smer & tbmask
            upper = smer >> 2t
            scramble_key = (upper * mixer) & tbmask
            tsmer = (lower ⊻ scramble_key) + 1        
            c[:, tsmer] .+= ωs_i
        end 

        c/=sqrt(sum(abs2.(c))+1e-16) # normalization

    return vec(c)
end

dna = rand(UInt8.(0:3), 20_000)
v = RoPE(dna, 8,4,4)