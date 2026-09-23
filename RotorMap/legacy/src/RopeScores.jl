# __precompile__(true)

module RopeScores

export cutting_threshold, cutting_threshold_new
export points, curve, predict_mutrate, scores

using Base.Threads
using LinearAlgebra
using StringDistances
using Random
using CUDA
using NPZ
using ProgressMeter
using ..RopeEncoders
using ..Utils

function cutting_threshold_new(re::RopeEncoder, err, err_shift = 0.05; n = 10000, normalize=1, double=false, fixed=false, euc=false)
    # @time   
    # dnas = rand(UInt8.(0:3), re.k*n)
    # muts = copy(dnas)
    if fixed == false
        dnas = rand(UInt8.(0:3), re.k*n)
    elseif fixed == true
        dna1 = rand(UInt8.(0:3), re.k)
        dnas = vcat([dna1 for i=1:n]...)
    else # fixed==dna1
        dnas = vcat([fixed for i=1:n]...)
    end
    muts = copy(dnas)

    starts = 1:re.k:length(dnas) |> collect
    stops = re.k:re.k:length(dnas) |> collect
    # if length(stops) < length(starts) 
    #     push!(stops, length(reads))
    # end
    # starts = starts .|> UInt32 |> CuArray
    # stops = stops .|> UInt32 |> CuArray

    # @time 
    @threads for i=1:n
        # @views mut = mutate(dnas[starts[i]:stops[i]], err)		
        # mut_shift = mutate_shift(mut, err_shift)		
        # @views muts[starts[i]:stops[i]] .= mut_shift
        @views mut_shift = mutate_shift(dnas[starts[i]:stops[i]], err_shift)		
        mut = mutate(mut_shift, err)		
        @views muts[starts[i]:stops[i]] .= mut
    end

    # # e = re.e |> CuArray
    # # CUDA.@time 
    # ednas = encode_batch_m1_cuda(re, CuArray(dnas), 1:re.k, e=e) 
    # # CUDA.@time 
    # emuts = encode_batch_m1_cuda(re, CuArray(muts), 1:re.k, e=e)

    CUDA.@time ednas, _ = encode_batch_cuda_all_best(re, CuArray(dnas), starts = CuArray(starts), stops = CuArray(stops), normalize=normalize, double=double)
    CUDA.@time emuts, _ = encode_batch_cuda_all_best(re, CuArray(muts), starts = CuArray(starts), stops = CuArray(stops), normalize=normalize, double=double)

    # @show size(ednas)

    # throw("cut")

    # CUDA.@time 
    # fids = sum(conj.(ednas) .* emuts, dims=2) .|> abs2 |> Array
    # fids = sum(fids, dims=1) / size(fids, 1)
    # fids = fids |> vec

    # ednas = reshape(ednas, :, size(ednas,3))
    # emuts = reshape(emuts, :, size(emuts,3))
    # we use default version in the mapping stage 1, so this code is the correct one
    if euc==false
        fids = sum(conj.(ednas) .* emuts, dims=(1,2)) .|> abs2 |> Array |> vec # no need to divide by m
    else
        fids = sum(conj.(ednas) .* emuts, dims=(1,2)) .|> real |> Array |> vec
    end
    # fids = sum(conj.(ednas) .* emuts, dims=1) .|> abs2 |> Array |> vec # no need to divide by m
    # fids = sum(fids, dims=1) / size(fids, 1)  
    # fids = fids |> vec 

    return (minimum(fids), maximum(fids), sum(fids)/n)
end

function cutting_threshold(re::RopeEncoder, err, err_shift = 0.05; n = 1000, e)
    # @time 
    dnas = rand(UInt8.(0:3), re.k, n)
    muts = copy(dnas)

    # @time 
    @threads for i=1:n
        @views mut = mutate(dnas[:,i], err)		
        mut_shift = mutate_shift(mut, err_shift)		
        @views muts[:,i] = mut_shift
    end

    # e = re.e |> CuArray
    # CUDA.@time 
    ednas = encode_batch_m1_cuda(re, CuArray(dnas), 1:re.k, e=e) 
    # CUDA.@time 
    emuts = encode_batch_m1_cuda(re, CuArray(muts), 1:re.k, e=e)

    # CUDA.@time 
    fids = sum(conj.(ednas) .* emuts, dims=1) .|> abs2 |> Array |> vec
    
    return (minimum(fids), maximum(fids), sum(fids)/n)
end

function points(re::RopeEncoder, max_err=0.5, sparsity=1; n = 1000, shift_err = 0.0, ld=true, default=true, shift=false, normalize=1, mode=:best, fixed=false, cache=nothing)
    # e = exp.(2π*im/re.k .* (0:re.k+re.s)) |> cu
    # @time 

    if cache != nothing 
        dnas, muts, yaxe = cache
        n = size(dnas, 2)
        @goto skip_gen         
    end

    # dnas = rand(UInt8.(0:3), re.k, n)
    # muts = deepcopy(dnas)
    if fixed == false
        dnas = [rand(UInt8.(0:3), re.k) for i=1:n]
    elseif fixed == true
        dna1 = rand(UInt8.(0:3), re.k)
        dnas = [dna1 for i=1:n]
    else # fixed==dna1
        dnas = [fixed for i=1:n]
    end
    muts = deepcopy(dnas)

    # if ld 
    #     yaxe = Array{Int64}(undef, n)
    # else
        # yaxe = Array{Float64}(undef, n)
    # end
    yaxe = zeros(Float64, n)
    # @time 
    @showprogress @threads for i=1:n 			
		err_i = max_err*i^sparsity/n^sparsity
        
        if !shift
            # mut_shift = mutate_shift(dnas[i], 0.1)
            if shift_err > 0.0
                mut_shift = mutate_shift(dnas[i], shift_err)
            else 
                mut_shift = dnas[i] 
            end
            # mut = mutate(dnas[i], err_i)
            mut = mutate(mut_shift, err_i)
            # @views mut = mutate_shift(mut, 0.05)		
        else
            mut = mutate_shift(dnas[i], err_i)
        end
        muts[i] = mut

        if ld 
            yaxe[i] += Levenshtein()(dnas[i], mut)
        else
            yaxe[i] = err_i
        end
    end

    @label skip_gen

    # # CUDA.@time 
    # ednas, _ = encode_batch_cuda_all_best(re, CuArray(dnas), e=e) 
    # # CUDA.@time 
    # emuts, _ = encode_batch_cuda_all_best(re, CuArray(muts), e=e)

    jdnas = jagged_array(dnas) 
    ednas, _ = encode_batch_cuda_all_best(re, jdnas.vect, starts=cu(jdnas.starts), stops=cu(jdnas.stops), normalize=normalize, mode=mode)
    jmuts = jagged_array(muts) 
    emuts, _ = encode_batch_cuda_all_best(re, jmuts.vect, starts=cu(jmuts.starts), stops=cu(jmuts.stops), normalize=normalize, mode=mode)

    # display(dnas)
    # println(emuts)
    # @show size(ednas)
    # @show CUDA.@allowscalar [norm(ednas[i,:,1]) for i=1:re.m]

    # CUDA.@time 
    ips = sum(conj.(ednas) .* emuts, dims=2) 
    # @show size(ips)

    if default 
        if normalize == 1
            ips = sum(ips, dims=1) / size(ips,1)
        elseif normalize == 0
            ips = sum(ips, dims=1) #/ size(ips,1)
        end
        ips = reshape(ips, size(ips,3))
        fids = abs2.(ips)  
        # @show fids |> sort
    else  # shift invariant              
        fids = abs2.(ips)
        fids = sum(fids, dims=1) / size(fids,1)
        fids = reshape(fids, size(fids,3))
        # fids = abs.(ips)
        # fids = sum(fids, dims=1).^2 #/ size(fids,1)
        # fids = reshape(fids, size(fids,3))
    end

    npzwrite("points.npy", hcat(fids |> Array, yaxe))

    # euclids = sum(abs2.(ednas.-emuts), dims=(1,2))
    # euclids = reshape(euclids, size(euclids,3))    
    
    # npzwrite("points.npy", hcat(euclids |> Array, yaxe))

    return dnas, muts, yaxe, fids
end

function curve(re::RopeEncoder, n=100; me=0.5, ne=100, rng=Random.default_rng(), default=true, shift=false, normalize=0, mode=:default) 
    # e = exp.(2π*im/re.k .* (0:re.k+re.s)) |> cu

    yaxe = zeros(Float64, n+1)
    xaxe = zeros(Float64, n+1)

    yaxe[1] = 0.0
    xaxe[1] = 1.0 

	@showprogress @threads for i in 1:n 
		err = me*i/n
        yaxe[i+1] = err
		begin 			
            # dnas = rand(rng, UInt8.(0:3), re.k, ne)
            # muts = copy(dnas)
            dnas = [rand(UInt8.(0:3), re.k) for i=1:ne]
            muts = deepcopy(dnas)

			@views for j=1:ne
				if !shift
					# muts[:,j] .= mutate(dnas[:,j], err, rng=rng)		
                    muts[j] .= mutate(dnas[j], err, rng=rng)		
				else 
					# muts[:,j] .= mutate_shift(dnas[:,j], err, rng=rng)		
                    muts[j] .= mutate_shift(dnas[j], err, rng=rng)		
				end 	
            end 

            # ednas, _ = encode_batch_cuda_all_best(re, CuArray(dnas), e=e) 
            # emuts, _ = encode_batch_cuda_all_best(re, CuArray(muts), e=e)

            jdnas = jagged_array(dnas) 
            ednas, _ = encode_batch_cuda_all_best(re, jdnas.vect, starts=cu(jdnas.starts), stops=cu(jdnas.stops), normalize=normalize, mode=mode)
            jmuts = jagged_array(muts) 
            emuts, _ = encode_batch_cuda_all_best(re, jmuts.vect, starts=cu(jmuts.starts), stops=cu(jmuts.stops), normalize=normalize, mode=mode)

            ips = sum(conj.(ednas) .* emuts, dims=2) 
            # @show size(ips)

            if default 
                if normalize == 1
                    ips = sum(ips, dims=1) / size(ips,1)
                elseif normalize == 0
                    ips = sum(ips, dims=1) #/ size(ips,1)
                end
                ips = reshape(ips, size(ips,3))
                fids = abs2.(ips)  
                # @show fids |> sort
            else  # shift invariant              
                fids = abs2.(ips)
                fids = sum(fids, dims=1) / size(fids,1)
                fids = reshape(fids, size(fids,3))
                # fids = abs.(ips)
                # fids = sum(fids, dims=1).^2 #/ size(fids,1)
                # fids = reshape(fids, size(fids,3))
            end

            fids_avg = sum(fids)/length(fids)
			xaxe[i+1] = fids_avg
		end		
	end

    npzwrite("curve.npy", hcat(xaxe, yaxe))
	return hcat(xaxe, yaxe)
end 

function predict_mutrate(curve, fid)
    @views xaxe = curve[:,1]#[2:end]
    @views yaxe = curve[:,2]#[2:end]

    n = length(xaxe)
    me = curve[:,2][end]

	# i = searchsortedfirst(xaxe, fid, rev=true)
    i = searchsortedfirst(xaxe, fid, rev=true) # i >= 2 if fid < 1.0

	if i == 1
		return 0.0
	end

	if i > n
		# return me*i/n # todo better value
        return min(1.0, yaxe[n] + (fid-xaxe[n])*((yaxe[n-1]-yaxe[n]))/(xaxe[n-1]-xaxe[n]))
	end

	# @show i, xaxe[i], xaxe[i-1]

	# y = me/n * ( (xaxe[i-1]-fid)*i + (fid-xaxe[i])*(i+1) ) / (xaxe[i-1]-xaxe[i])
    y = ( (xaxe[i-1]-fid)*yaxe[i] + (fid-xaxe[i])*yaxe[i-1] ) / (xaxe[i-1]-xaxe[i])
	return y
end 

function scores(re::RopeEncoder, curve, n=100; me=0.5, ne=100, rng=Random.default_rng(), default=true, shift=false) 
    e = exp.(2π*im/re.k .* (0:re.k+re.s)) |> cu

    yaxe = zeros(Float64, n+1)
    xaxe = zeros(Float64, n+1)

    yaxe[1] = 0.0
    xaxe[1] = 0.0 

	@showprogress @threads for i in 1:n 
		err = me*i/n
        yaxe[i+1] = err
		begin 			
            dnas = rand(rng, UInt8.(0:3), re.k, ne)
            muts = copy(dnas)
			@views for j=1:ne
				if !shift
					muts[:,j] .= mutate(dnas[:,j], err, rng=rng)		
				else 
					muts[:,j] .= mutate_shift(dnas[:,j], err, rng=rng)		
				end 	
            end 

            ednas, _ = encode_batch_cuda_all_best(re, CuArray(dnas), e=e) 
            emuts, _ = encode_batch_cuda_all_best(re, CuArray(muts), e=e)

            ips = sum(conj.(ednas) .* emuts, dims=2) 
            # @show size(ips)

            if default 
                ips = sum(ips, dims=1) / size(ips,1)
                ips = reshape(ips, size(ips,3))
                fids = abs2.(ips)
                # @show fids |> sort
            else  # shift invariant              
                fids = abs2.(ips)
                fids = sum(fids, dims=1)/ size(fids,1)
                fids = reshape(fids, size(fids,3))
            end

            pred_mrates = map(x -> predict_mutrate(curve, x), Array(fids))
            mse = sum(abs2.(pred_mrates .- err)) / ne
			xaxe[i+1] = mse |> sqrt
		end		
	end

    @show xaxe
    # yaxe is the real mut rate - plot on x
    # xaxe is the prediction error - plot on y
    npzwrite("scores.npy", hcat(yaxe, xaxe))
	return hcat(yaxe, xaxe)
end 

# function fitting_curve(re::RopeEncoder; me=0.5, ne=100, rng=Random.default_rng(), shift=false) # we use pure mode in re here
# 	efidall = Array{Float64}(undef, n)

# 	@sync for i in 1:n 
# 		err = me*i/n
# 		@spawn begin 
# 			efid = 0.0
# 			for j=1:ne
# 				dna = rand(rng, 0:3, re.N)
# 				if !shift
# 					mut = mutate(dna, err, rng=rng)		
# 				else 
# 					mut = mutate_shift(dna, err, rng=rng)		
# 				end 	
# 				edna = re(dna)
# 				emut = re(mut)
# 				if re.c==0 
# 					@views fid = sum([abs(edna[t,:]'*emut[t,:]) for t=1:re.m])^2
# 				else 
# 					@views fid = sum([abs(edna[t,:]'*emut[t,:]) for t=1:re.m])^2
# 					# @views fid = abs2(edna'*emut)
# 				end
# 				efid += fid
# 			end 
# 			efidall[i] = efid/ne
# 		end		
# 	end

# 	return efidall
# end 



end