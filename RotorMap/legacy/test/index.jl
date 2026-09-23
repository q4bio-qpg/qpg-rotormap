using Pkg

Pkg.activate(".")

using RotorMap
using RotorMap.RopeEncoders
using RotorMap.RopeIndexers
using LinearAlgebra
using BenchmarkTools

N = 10^7
dna_ref = rand(UInt8.(0:3),N)
re21 = RopeEncoder(N=10^5, s=4, m=1, c=64) 
ri21 = RopeIndexer(N=10^7, re=re21, step=10^4)


struct IndexTester 
    ri::RopeIndexer
    dna_ref::Vector{Int8}
    # index::Matrix{ComplexF64} # ri.re.m*4^ri.re.s x ri.n
    # compact::Matrix{ComplexF64}

    function IndexTester(ri::RopeIndexer)
        dna_ref = rand(0:3, ri.N)
        ri(dna_ref) 
        # index = ri.index
        return new(ri, dna_ref)
    end

    function (self::IndexTester)(num=100) # use for testing that the indexer output matches the rope encodings       
        err = 0.0

        ls = rand(1:self.ri.n, num-2)
        push!(ls, 1)
        push!(ls, self.ri.n)
        @views for i = 1:num 
            l = ls[i]       
            dna = self.dna_ref[1 + self.ri.step*(l-1) : self.ri.k + self.ri.step*(l-1)]
            r, r_c = self.ri.re(dna)
            erri = (r - self.ri.index[:,l]) |> norm
            erri_c = (r_c - self.ri.compact[:,l]) |> norm
            # @show erri
            err += erri+erri_c
        end 
        
        return err
    end
end

it21 = IndexTester(ri21)

@show it21()
