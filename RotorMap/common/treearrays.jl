# =============================================================================
# treearrays.jl -- vendored verbatim from the legacy package (legacy/src/TreeArrays.jl):
# rope data structure for efficient DNA mutations; used by common/dna.jl's mutate().
# Consumers must include this file BEFORE common/dna.jl (it does `using .TreeArrays`).
# =============================================================================

# __precompile__(true)

# Similar to https://en.wikipedia.org/wiki/Rope_(data_structure)
# Needed for efficient mutations of DNA
module TreeArrays
export TreeArray, length, size, insert!, popat!, subsat!, insertat!, Vector

using Base.Threads
import Base: length, size, getindex, setindex!, insert!, popat!, deleteat!, Vector

const THRESHOLD = 10_000_000

mutable struct TreeArray{T} <: AbstractVector{T} 
	length::Int 
	const left::Union{TreeArray{T}, Nothing} 
	const right::Union{TreeArray{T}, Nothing}
	const leaf::Bool
	const data::AbstractVector{T}

    # note: should be length==length(data) if leaf and length==length(left)+length(right) otherwise
    # a rebalancing may be required from time to time

    function TreeArray(data::AbstractVector{T}; max_len_leaf=1000) where T
        n = length(data)
        if n <= max_len_leaf
            return new{T}(n,nothing,nothing,true,Vector(data))
        else 
            n2 = n÷2			
			left = TreeArray(view(data, 1:n2), max_len_leaf=max_len_leaf)
			right = TreeArray(view(data, n2+1:n), max_len_leaf=max_len_leaf)							
            return new{T}(n,left,right,false, Array{T}(undef, 0))
        end
    end
end

function length(A::TreeArray)
	return A.length
end

function size(A::TreeArray)
	return (A.length,)
end

function getindex(A::TreeArray, i::Int)
	if A.leaf 
		return A.data[i]
	end

	nleft = A.left.length 

	if i <= nleft
		return A.left[i]
	else
		return A.right[i-nleft]
	end
end 
# getindex(A::TreeArray, I::Vararg{Int, N}) where N = error("vararg not implemented")

function setindex!(A::TreeArray, v, i::Int)
	if A.leaf 
		A.data[i] = v
        return 
	end

	nleft = A.left.length 

	if i <= nleft
		A.left[i] = v
	else
		A.right[i-nleft] = v
	end
end

function insert!(A::TreeArray, i::Int, item)
	if A.leaf 
		insert!(A.data, i, item)        

		A.length += 1
        return 
	end

	nleft = A.left.length 

	if i <= nleft
		insert!(A.left, i, item)
	else
        insert!(A.right, i-nleft, item)
	end	

	A.length += 1
	return 
end

function subsat!(A::TreeArray, inds::AbstractVector, vals::AbstractVector) # substitutions
	n = length(inds)
	
	if A.leaf  
		for i =1:n 
			A.data[inds[i]] = vals[i]
		end
		return 
	end

	nleft = A.left.length 

	mids = searchsorted(inds, nleft) # inds[mid]<=nleft

	# m1 = mids.start
	m2 = mids.stop 
	
	if m2 == n # nleft >= inds
		subsat!(A.left, view(inds, 1:m2), view(vals, 1:m2))		
	elseif m2 == 0 # nleft < inds
		inds .-= nleft
		subsat!(A.right, inds, vals)		
	else
		# the order doesn't matter
		inds[m2+1:end] .-= nleft
		if m2 <= THRESHOLD 
			subsat!(A.left, view(inds, 1:m2), view(vals, 1:m2))	
			subsat!(A.right, view(inds, m2+1:n), view(vals, m2+1:n))	
		else 
			@sync begin 
				@spawn subsat!(A.left, view(inds, 1:m2), view(vals, 1:m2))	
				@spawn subsat!(A.right, view(inds, m2+1:n), view(vals, m2+1:n))	
			end
		end 
	end
end

function deleteat!(A::TreeArray, inds::AbstractVector) 	
	n = length(inds)
	A.length -= n

	if A.leaf  
		return deleteat!(A.data, inds)
	end

	nleft = A.left.length 

	mids = searchsorted(inds, nleft) # inds[mid]<=nleft

	# m1 = mids.start
	m2 = mids.stop 
	
	if m2 == n # nleft >= inds
		deleteat!(A.right, inds)		
	elseif m2 == 0 # nleft < inds
		inds .-= nleft
		deleteat!(A.right, inds)		
	else
		# the order doesn't matter
		# view(inds, m2+1:A.length).-=nleft
		@views inds[m2+1:n] .-= nleft
		if m2 <= THRESHOLD
			deleteat!(A.right, view(inds, m2+1:n))
			deleteat!(A.left, view(inds, 1:m2))
		else 
			@sync begin 
				@spawn deleteat!(A.right, view(inds, m2+1:n))
				@spawn deleteat!(A.left, view(inds, 1:m2))
			end
		end 		
	end
end

function insertat!(A::TreeArray, inds::AbstractVector, vals::AbstractVector)
	n = length(inds)
	A.length += n 

	if A.leaf  
		for i = n:-1:1 
			insert!(A.data, inds[i], vals[i])
		end
		return 
	end

	nleft = A.left.length 

	mids = searchsorted(inds, nleft) # inds[mid]<=nleft

	# m1 = mids.start
	m2 = mids.stop 
	
	if m2 == n # nleft >= inds
		insertat!(A.left, view(inds, 1:m2), view(vals, 1:m2))		
	elseif m2 == 0 # nleft < inds
		inds .-= nleft
		insertat!(A.right, inds, vals)		
	else
		# the order doesn't matter		
		inds[m2+1:end] .-= nleft
		if m2 <= THRESHOLD 
			insertat!(A.left, view(inds, 1:m2), view(vals, 1:m2))	
			insertat!(A.right, view(inds, m2+1:n), view(vals, m2+1:n))	
		else 
			@sync begin 
				@spawn insertat!(A.left, view(inds, 1:m2), view(vals, 1:m2))	
				@spawn insertat!(A.right, view(inds, m2+1:n), view(vals, m2+1:n))	
			end
		end 
	end
end

function checktree(A::TreeArray)
	if A.leaf
		return A.length == length(A.data)
	end

	cl = checktree(A.left)
	cr = checktree(A.right)
	ch = (A.length == A.left.length+A.right.length) && cl && cr
	return ch
end

function popat!(A::TreeArray, i::Int) 
    # todo: treat dead leaves? shouldn't happen	

	if A.leaf 
		v = popat!(A.data, i)
        A.length -= 1
		return v
	end

	nleft = A.left.length 

	if i <= nleft
		v = popat!(A.left, i)
        A.length -= 1
		return v
	else
		v = popat!(A.right, i-nleft)
        A.length -= 1
        return v
	end	
end

function flatten(A::TreeArray, dataview::SubArray)
	# @assert A.length==length(dataview)
	
	n = A.length
	if A.leaf 
		dataview[1:n] = A.data
        return 
    end

	n1 = A.left.length 

	flatten(A.left, view(dataview, 1:n1))
	flatten(A.right, view(dataview, n1+1:n))
    return  
end

function Vector(A::TreeArray{T}) where T
	n = length(A)
	data = Array{T}(undef, n)
	flatten(A, view(data, 1:n))
	return data
end

end