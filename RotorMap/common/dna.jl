# =============================================================================
# dna.jl -- vendored subset of the legacy package's Utils.jl (legacy/src/Utils.jl):
# the DNA-sequence utilities the workbench actually uses. These BARE names
# replace the old `Utils.X` calls everywhere in the new tree:
#   save / load          Julia Serialization of arbitrary data (index/reference .bin files)
#   save_fasta           write DNA (Vector{UInt8} codes) as a fasta, optional heads
#   dna_letters          UInt8 code -> letter LUT
#   generate_reference   random ACGT reference of N bases (seeded)
#   generate_reads       sample n reads of length k + mutate them (the provenance pattern)
#   mutate               substitutions/insertions/deletions via a TreeArray (rope tree)
#   mutate_shift         left-shift by floor(n*err/2) bases + random tail
# REQUIRES: common/treearrays.jl included first (this file does `using .TreeArrays`).
# Sources: Utils.jl lines 91-104 (save/load/dna_letters), 147-192 (save_fasta),
#          892-990 (generate_reference/generate_reads/mutate), 996-1005
#          (mutate_shift). Verbatim (one noted fix in mutate_shift).
# =============================================================================
using Random
using Base.Threads
using ProgressMeter: @showprogress
using Serialization # save/load (Julia-serialized .bin files)
using .TreeArrays

function save(data, file)
    open(file, "w") do f
        serialize(f, data)
    end
end

function load(file)
    data = open(file, "r") do f
        deserialize(f)
    end
    return data
end

const dna_letters = UInt8.(['A', 'C', 'G', 'T'])

function save_fasta(dnas::AbstractArray, file::String; heads=[], pos=[], chunk_size=2^12)
	n = length(dnas)

	if heads==[]
		if pos != [] 
			heads = ["> " * repr(pos[i]) for i in eachindex(dnas)]
		else
			heads = ["> " for i in eachindex(dnas)]
		end 
    end

	write_buffer = Vector{UInt8}(undef, chunk_size)

	open(file, "w") do io
        # Use zip for cleaner, more direct iteration over sequences and their headers
        @showprogress for (dna, head) in zip(dnas, heads)
            # Write the header for the current sequence
            println(io, head)

            # Process the current DNA sequence in chunks
            for i_start in 1:chunk_size:length(dna)
                # Determine the end of the current chunk, correctly handling the last one
                i_end = min(i_start + chunk_size - 1, length(dna))
                
                # The number of bytes to actually write from the buffer
                chunk_len = i_end - i_start + 1

                # Fill the pre-allocated buffer in-place.
                # This is the core of the allocation reduction.
                @inbounds for k in 1:chunk_len
                    # dna is 0-indexed (0-3), so we add 1 for 1-based indexing.
                    # We directly map from the number to the corresponding ASCII byte.
                    write_buffer[k] = dna_letters[dna[i_start + k - 1] + 1]
                end
                
                # Write only the filled part of the buffer to the file.
                # `view` avoids creating another array for the slice.
                write(io, view(write_buffer, 1:chunk_len))
            end
            
            # FASTA format requires a newline at the end of each sequence
            println(io)
        end
    end
	return 1
end

function generate_reference(N::Int64; seed::Union{Integer, Nothing} = nothing)
    rng = Random.default_rng()
    if seed != nothing
        rng = Xoshiro(seed)
    end

    dna_ref = rand(rng, UInt8.(0:3), N)
	
    return dna_ref
end

function generate_reads(dna_ref::Vector{T}, k::Int64, n::Int64=1; err::Float64=0.15, seed::Union{Int64, Nothing}=nothing) where T
    # reads = zeros(T, k, n)  
	reads = [zeros(T, k) for i = 1:n]
	
	rng = Random.default_rng()
    if seed != nothing
        rng = Xoshiro(seed)
    end

	N = length(dna_ref)
	pos = rand(rng, 0:N-k, n)

	rngs = [Xoshiro(rand(rng, Int64)) for i=1:n] # generate rngs for each execution thread for reproducibility

	@showprogress @threads for i in 1:n
        @views dna = dna_ref[pos[i]+1:pos[i]+k]
        mut = mutate(dna, err, rng=rngs[i])
		# @views reads[:,i] = mut
		reads[i] .= mut
	end
	
	return reads, pos
end

function mutate(dna::AbstractVector{T}, err=0.1; rng=Random.default_rng(), mll=1000) where T # dna = [2,1,0,3,1,0,...]
	n = length(dna)
	mut = TreeArray(dna, max_len_leaf=mll)	
	n_err = n*err |> ceil |> Int
	ins_rate = 1/3
	n_ins = n_err*ins_rate |> floor |> Int
	n_del = n_ins
	n_sub = n_err - n_ins - n_del

	inds_s = Array{Int}(undef, n_sub)
	inds_d = Array{Int}(undef, n_del)
	inds_i = Array{Int}(undef, n_ins)

	vals_s = Array{T}(undef, n_sub)
	vals_i = Array{T}(undef, n_ins)
	
	rngs = [Xoshiro(rand(rng, Int64)) for i=1:4] # generate rngs for each execution thread for reproducibility
	@sync begin
		@spawn begin 
			for t=1:n_sub # gen indices for substitutions
				inds_s[t] = n*rand(rngs[1]) |> ceil |> Int
			end
			sort!(inds_s)

            for i in 1:n_sub # gen values for substitutions
				rl = rand(rngs[1], [T(k) for k=1:3])
				l = (dna[inds_s[i]] + rl)%4 # mutated letter
				vals_s[i] = l
			end
		end
		@spawn begin # gen indices for deletions
			for t=0:n_del-1
				inds_d[t+1] = (n-t)*rand(rngs[2]) |> ceil |> Int 
				# inds_d[t+1] = (n-t)*rand(rngs[2])/2 |> ceil |> Int 
			end
			sort!(inds_d)

			# make inds_d unique (this code is equivalent to that)
			for i = 2:n_del
				inds_d[i] = max(inds_d[i], inds_d[i-1]+1)
			end
		end 
		@spawn begin 
			for t=0:n_ins-1 # gen indices for insertions
				inds_i[t+1] = (n-n_del)*rand(rngs[3]) |> ceil |> Int 
				# inds_i[t+1] = (n-n_del) - (n-n_del)*rand(rngs[3])/2 |> ceil |> Int 
			end
			sort!(inds_i)
        end
        @spawn begin
			for i in 1:n_ins # gen values for insertions
				rl = rand(rngs[4], [T(k) for k=0:3])
				vals_i[i] = rl
			end		
		end
	end
	
	subsat!(mut, inds_s, vals_s)
	deleteat!(mut, inds_d)
	insertat!(mut, inds_i, vals_i)

	ret = Vector(mut)
	return ret
end

function mutate_shift(dna::AbstractVector{T}, err=0.1; shift=nothing, rng=Random.default_rng()) where T # dna = [2,1,0,3,1,0,...]
	n = length(dna)
    if shift==nothing
	    shift = n*err/2 |> floor |> Int
    end
	tail = rand(rng, T.(0:3), shift) # vendored fix: typed tail so vcat stays Vector{T}
	                                 # (legacy's rand(rng, 0:3, shift) promotes to Int)

	ret = vcat(view(dna, 1+shift:n), tail)
	return ret
end
