# ==============================================================================
# probe_true_locus_scores.jl -- the TRUE-LOCUS rope inner product of sampled
# reads, on CPU (float64 golden-reference encoder), against their Levenshtein
# distance to the true window.
#
# WHY
#   probe_top1_scores on maize_1X_15_shuf vs the s5m1c5 index reports mean
#   top-1 = 0.771 -- "too high" next to the synthetic random-DNA sweep
#   (rope_mutation_sweep: err = 0.15 -> fidelity ~0.53), even though the reads
#   were generated with the SAME mutate() op model (1/3 sub + 1/3 ins +
#   1/3 del, indexsample.jl) at err = 0.15.  Since top1-correct is 99.7%,
#   mean top-1 ~= mean true-locus score; this probe measures that quantity
#   DIRECTLY (fresh encode of the read's leading K bases vs the window at its
#   provenance location) alongside the Levenshtein %, i.e. the EMPIRICAL
#   edit% -> fidelity curve on real maize, to be compared with the synthetic
#   random-DNA curve.
#
# NOTES
#   * forward-strand reads only (rc reads are excluded from the sample; their
#     file-first-K trim maps to the window's far end).
#   * Levenshtein via StringDistances (a RotorMap project dep) on the K-length
#     strings; threads over reads.
#
# USAGE
#   PTL_REF=<primary12.fna> PTL_READS=<reads.fasta> \
#     julia --project=RotorMap RotorMap/experiments/probe_true_locus_scores.jl
#   PTL_N (100), PTL_SEED (42), PTL_K (20000)
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("encode", "reference.jl") # _ref_rope_frag_real + RopeEncoder (pure Base)

using Printf
using Mmap
using LinearAlgebra
using Random
using Statistics
using StringDistances

const PTL_REF = get(ENV, "PTL_REF", "")
const PTL_READS = get(ENV, "PTL_READS", "")
const PTL_N = parse(Int, get(ENV, "PTL_N", "100"))
const PTL_SEED = parse(Int, get(ENV, "PTL_SEED", "42"))
const PTL_K = parse(Int, get(ENV, "PTL_K", "20000"))
const PTL_S, PTL_M, PTL_C = 5, 1, 5 # the s5m1c5 production config

const NL = UInt8('\n')
const CR = UInt8('\r')
const GT = UInt8('>')

# fasta record scan (probe_levenshtein's): call f(header, lo, hi) per record
function _each_record(f, raw::AbstractVector{UInt8})
    n = length(raw)
    s = 1
    while s <= n && raw[s] != GT
        s += 1
    end
    while s <= n
        h0 = s + 1
        h1 = h0
        while h1 <= n && raw[h1] != NL
            h1 += 1
        end
        head = rstrip(String(raw[h0:h1-1]), '\r')
        e = h1 + 1
        while e <= n && raw[e] != GT
            e += 1
        end
        f(head, h1 + 1, e - 1)
        s = e
    end
    return nothing
end

"""ACGT bytes -> 0:3 codes; anything else -> 0, counted (returned)."""
function _codes!(dst::Vector{UInt8}, src::AbstractVector{UInt8})
    n = 0
    @inbounds for i in eachindex(src)
        b = src[i]
        c = b == UInt8('A') ? UInt8(0) : b == UInt8('C') ? UInt8(1) :
            b == UInt8('G') ? UInt8(2) : b == UInt8('T') ? UInt8(3) : UInt8(0)
        c == 0 && b != UInt8('A') && (n += 1)
        dst[i] = c
    end
    return n
end

"""Split [Re; Im] unit-energy ropes -> (Re<z0,z1>, |<z0,z1>|)."""
function rope_ip(e0::AbstractVector{<:Real}, e1::AbstractVector{<:Real})
    D = length(e0) >> 1
    ipre = dot(e0, e1)
    ipim = dot(view(e0, 1:D), view(e1, D+1:2D)) -
           dot(view(e0, D+1:2D), view(e1, 1:D))
    return ipre, abs(ComplexF64(ipre, ipim))
end

function main()
    ref = open(PTL_REF) do io
        Mmap.mmap(io)
    end
    reads = open(PTL_READS) do io
        Mmap.mmap(io)
    end

    # reference: accession -> contiguous uppercase sequence
    refseqs = Dict{String,Vector{UInt8}}()
    _each_record(ref) do head, lo, hi
        acc = String(split(head)[1])
        v = Vector{UInt8}(uppercase(String(ref[lo:hi])))
        filter!(b -> b != NL && b != CR, v)
        refseqs[acc] = v
    end
    @info "reference records" n = length(refseqs)

    # reads: provenance + byte span (sequences pulled lazily per sample)
    raccs = String[]
    rstarts = Int[]
    rrcs = Bool[]
    rspans = UnitRange{Int}[]
    _each_record(reads) do head, lo, hi
        f = split(head, ':')
        length(f) == 3 || error("unreadable provenance header: $head")
        push!(raccs, String(f[1]))
        push!(rstarts, parse(Int, f[2]))
        push!(rrcs, split(f[3], '#')[1] == "true")
        push!(rspans, lo:hi)
    end
    nreads = length(raccs)
    nfwd = count(!, rrcs)
    @info "read records" nreads nfwd

    # sample N forward reads
    rng = MersenneTwister(PTL_SEED)
    fwd = findall(!, rrcs)
    pick = sort!(fwd[randperm(rng, length(fwd))[1:PTL_N]])

    re = RopeEncoder(k = PTL_K, s = PTL_S, m = PTL_M, c = PTL_C)
    res_re = Vector{Float64}(undef, PTL_N)
    res_abs = Vector{Float64}(undef, PTL_N)
    res_lev = Vector{Int}(undef, PTL_N)
    res_len = Vector{Int}(undef, PTL_N)

    Threads.@threads for i in 1:PTL_N
        r = pick[i]
        acc, start0 = raccs[r], rstarts[r]
        seq = Vector{UInt8}(@view reads[rspans[r]])
        filter!(b -> b != NL && b != CR, seq)
        length(seq) >= PTL_K || error("read $r: length $(length(seq)) < k")
        resize!(seq, PTL_K) # the flow encodes a read's leading K bases
        _upcase!(seq)
        rec = refseqs[acc]
        (start0 + PTL_K <= length(rec)) ||
            error("read $r: window at start0=$start0 overruns $acc")
        win = rec[(start0+1):(start0+PTL_K)]

        rcode = Vector{UInt8}(undef, PTL_K)
        _codes!(rcode, seq)
        er, _ = _ref_rope_frag_real(rcode, re; normalize = 0)
        wcode = Vector{UInt8}(undef, PTL_K)
        _codes!(wcode, win)
        ew, _ = _ref_rope_frag_real(wcode, re; normalize = 0)
        res_re[i], res_abs[i] = rope_ip(er, ew)
        res_lev[i] = evaluate(Levenshtein(), String(copy(win)), String(copy(seq)))
        res_len[i] = PTL_K
    end

    ord = sortperm(res_lev)
    @printf("%5s %-15s %11s %6s %6s %10s %10s\n", "i", "accession", "start0",
            "lev", "lev%", "Re<z,w>", "|<z,w>|")
    for (row, i) in enumerate(ord) # ascending Levenshtein
        row > 20 && row <= PTL_N - 3 && continue # head + tail of the sort
        r = pick[i]
        @printf("%5d %-15s %11d %6d %6.2f %10.6f %10.6f\n", i, raccs[r],
                rstarts[r], res_lev[i], 100 * res_lev[i] / PTL_K, res_re[i],
                res_abs[i])
    end

    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    @printf("\n== TRUE-LOCUS SCORES n=%d k=%d s5m1c5 (fresh float64 CPU encodes)\n",
            PTL_N, PTL_K)
    for (name, x) in (("Re<z_read,z_locus>", res_re), ("|<z_read,z_locus>|", res_abs))
        @printf("   %-20s mean %.6f std %.6f | min %.6f q05 %.6f q25 %.6f med %.6f q75 %.6f q95 %.6f max %.6f\n",
                name, sum(x) / length(x), std(x), quantile.(Ref(x), qs)...)
    end
    levpc = 100 .* res_lev ./ res_len
    @printf("   %-20s mean %.6f std %.6f | min %.6f q05 %.6f q25 %.6f med %.6f q75 %.6f q95 %.6f max %.6f\n",
            "levenshtein %", sum(levpc) / length(levpc), std(levpc),
            quantile.(Ref(levpc), qs)...)

    # empirical decay curve: mean true-locus Re per Levenshtein-% bin
    @printf("\n   lev%% bin     n   mean Re   mean |IP|   (synthetic sweep, op-err ~ 1.5x lev%%)\n")
    for blo in 0:2:20
        sel = findall(b -> blo <= b < blo + 2, levpc)
        isempty(sel) && continue
        mre = sum(res_re[sel]) / length(sel)
        mab = sum(res_abs[sel]) / length(sel)
        @printf("  [%4.1f,%4.1f) %5d %9.4f %11.4f\n", blo, blo + 2, length(sel),
                mre, mab)
    end
    flush(stdout)
end

_upcase!(s::Vector{UInt8}) = (for i in eachindex(s)
                                  b = s[i]
                                  UInt8('a') <= b <= UInt8('z') && (s[i] = b - 32)
                              end; s)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
