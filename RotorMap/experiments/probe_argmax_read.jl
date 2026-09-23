# ==============================================================================
# probe_argmax_read.jl -- forensic workup of ONE read whose mapping scores
# look impossible (vals ~ 1.0 despite err = 0.15 mutations).
#
# The subject: probe_top1_scores' rank-20 argmax read on the sample
# maize batch (read_128417, NC_050105.1:109977387, len 20000; all 20 of its
# top vals sit at 1.001..1.012).  For that read this probe computes, all on
# CPU with the float64 golden-reference encoder (s5m1c5, normalize = 0):
#   1. Levenshtein(read, true window)                  -- the real edit load
#   2. fresh Re<z_read,z_window> and |<z_read,z_window>| -- the honest score
#   3. synthetic control: 8x mutate(window, 0.15) (the sampler's exact model)
#      -> fidelity of each mutant vs the clean window's encoding, i.e. what
#      the encoding WOULD see if the edits were freshly drawn at this locus
#   4. same control on a RANDOM 20kb sequence (the rope_mutation_sweep
#      baseline, for contrast)
#   5. spectrum diagnostics: histogram energy concentration (top-bin share,
#      effective bin count 1/sum p_b^2, Shannon entropy) for the window, the
#      read, and the random sequence
#   6. local stride profile: fresh Re/|IP| vs the windows at start +/- 2000j
#      (the index's stride) out to +/- 20000
#   7. edit-load profile: Levenshtein per 1kb block (20 blocks) -- uniform or
#      clustered edits?
#
# USAGE
#   PRD_REF=<primary12.fna> PRD_READS=<reads.fasta> \
#     julia --project=RotorMap RotorMap/experiments/probe_argmax_read.jl
#   PRD_ID (read_128417), PRD_K (20000)
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("common", "treearrays.jl")
inc("common", "dna.jl") # mutate
inc("encode", "reference.jl")

using Random
using Printf
using Mmap
using LinearAlgebra
using Statistics
using StringDistances

const PRD_REF = get(ENV, "PRD_REF", "")
const PRD_READS = get(ENV, "PRD_READS", "")
const PRD_ID = get(ENV, "PRD_ID", "read_128417")
const PRD_K = parse(Int, get(ENV, "PRD_K", "20000"))
const PRD_ERR = parse(Float64, get(ENV, "PRD_ERR", "0.15"))
const PRD_SEEDS = 1:8

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

encode_codes(codes, re) = _ref_rope_frag_real(codes, re; normalize = 0)[1]

"""Histogram energy concentration of a split rope vector: (top-bin share,
effective bin count 1/sum p_b^2, Shannon entropy in bits, bins over p>1e-6).
The split layout is [Re(1:D); Im(D+1:2D)] -- bin b's complex value is
Re_b + i*Im_b."""
function concentration(e::AbstractVector{<:Real})
    D = length(e) >> 1
    m2 = abs2.(ComplexF64.(e[1:D], e[D+1:2D]))
    tot = sum(m2)
    p = m2 ./ tot
    eff = 1 / sum(abs2, p)
    ent = -sum(x -> x > 0 ? x * log2(x) : 0.0, p)
    return maximum(p), eff, ent, count(>(1e-6), p)
end

function main()
    # ---- inputs ---------------------------------------------------------------
    ref = open(PRD_REF) do io
        Mmap.mmap(io)
    end
    refseqs = Dict{String,Vector{UInt8}}() # accession -> contiguous sequence
    _each_record(ref) do head, lo, hi
        acc = String(split(head)[1])
        v = Vector{UInt8}(uppercase(String(ref[lo:hi])))
        filter!(b -> b != NL && b != CR, v)
        refseqs[acc] = v
    end

    rseq = UInt8[]
    meta = nothing
    reads = open(PRD_READS) do io
        Mmap.mmap(io)
    end
    _each_record(reads) do head, lo, hi
        meta === nothing && startswith(head, PRD_ID * " ") || return # head is '>'-stripped
        sp = findfirst(" src=", head)
        sp === nothing && return
        f = split(head[1:first(sp)-1])
        kv = Dict{String,String}()
        for x in f[2:end]
            eq = findfirst('=', x)
            eq === nothing && continue
            kv[String(x[1:first(eq)-1])] = String(x[first(eq)+1:end])
        end
        meta = (start = parse(Int, kv["start"]), len = parse(Int, kv["len"]),
                src = String(head[last(sp)+1:end])) # record header sans '>'
        append!(rseq, filter!(b -> b != NL && b != CR,
                              Vector{UInt8}(@view reads[lo:hi])))
    end
    meta === nothing && error("read $PRD_ID not found in $PRD_READS")
    length(rseq) == PRD_K || error("read length $(length(rseq)) != $PRD_K")
    acc = String(split(meta.src)[1])
    @info "subject" id = PRD_ID accession = acc start = meta.start len = meta.len

    rec = refseqs[acc]
    win = rec[meta.start:(meta.start + PRD_K - 1)] # start is 1-BASED (sample)

    re = RopeEncoder(k = PRD_K, s = 5, m = 1, c = 5)
    wcode = Vector{UInt8}(undef, PRD_K)
    _codes!(wcode, win)
    ew = encode_codes(wcode, re)
    rcode = Vector{UInt8}(undef, PRD_K)
    _codes!(rcode, rseq)
    er = encode_codes(rcode, re)

    # ---- 1-2. the honest numbers ---------------------------------------------
    lev = evaluate(Levenshtein(), String(copy(win)), String(copy(rseq)))
    ip_re, ip_abs = rope_ip(ew, er)
    @printf("\n== SUBJECT %s (%s:%d)\n", PRD_ID, acc, meta.start)
    @printf("   levenshtein(read, window)   = %d / %d = %.2f%%\n", lev, PRD_K,
            100 * lev / PRD_K)
    @printf("   fresh Re<z_read,z_window>   = %.6f   |<z_read,z_window>| = %.6f\n",
            ip_re, ip_abs)

    # ---- 3. synthetic-mutation control AT THIS LOCUS ---------------------------
    @printf("\n   synthetic control: 8x mutate(window, %.2f), fidelity vs clean window\n",
            PRD_ERR)
    fids = Float64[]
    for s in PRD_SEEDS
        m = mutate(wcode, PRD_ERR; rng = Xoshiro(s))
        em = encode_codes(m, re)
        f, _ = rope_ip(ew, em)
        push!(fids, f)
    end
    @printf("   fidelity: mean %.4f  min %.4f  max %.4f  (seeds %d-%d)\n",
            sum(fids) / length(fids), minimum(fids), maximum(fids), PRD_SEEDS[1],
            PRD_SEEDS[end])

    # ---- 4. random-sequence baseline -------------------------------------------
    rcode = UInt8.(rand(Xoshiro(777), 0:3, PRD_K))
    erand = encode_codes(rcode, re)
    rfids = Float64[]
    for s in PRD_SEEDS
        m = mutate(rcode, PRD_ERR; rng = Xoshiro(s))
        em = encode_codes(m, re)
        f, _ = rope_ip(erand, em)
        push!(rfids, f)
    end
    @printf("   random-sequence baseline: fidelity mean %.4f  min %.4f  max %.4f\n",
            sum(rfids) / length(rfids), minimum(rfids), maximum(rfids))

    # ---- 5. spectrum concentration ---------------------------------------------
    @printf("\n   %-16s %12s %14s %10s %8s\n", "sequence", "top-bin", "eff bins",
            "entropy", "bins>1e-6")
    for (nm, e) in (("window", ew), ("read", er), ("random", erand))
        top, eff, ent, nb = concentration(e)
        @printf("   %-16s %12.5f %14.1f %10.3f %8d\n", nm, top, eff, ent, nb)
    end

    # ---- 6. local stride profile ------------------------------------------------
    @printf("\n   local stride profile (offset : fresh Re : fresh |IP|)\n")
    for d in -20000:2000:20000
        s0 = meta.start + d
        (1 <= s0 && s0 + PRD_K - 1 <= length(rec)) || continue
        ccode = Vector{UInt8}(undef, PRD_K)
        _codes!(ccode, rec[s0:(s0 + PRD_K - 1)])
        ec = encode_codes(ccode, re)
        f, a = rope_ip(er, ec)
        @printf("   %+6d : %9.6f : %9.6f\n", d, f, a)
    end

    # ---- 7. edit-load profile per 1kb block --------------------------------------
    @printf("\n   edit%% per 1kb block:\n   ")
    for b0 in 1:1000:PRD_K
        b1 = min(b0 + 999, PRD_K)
        d = evaluate(Levenshtein(), String(win[b0:b1]), String(rseq[b0:b1]))
        @printf("%5.1f", 100 * d / (b1 - b0 + 1))
    end
    println()
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
