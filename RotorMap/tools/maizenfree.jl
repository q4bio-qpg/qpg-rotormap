# =============================================================================
# maizenfree.jl -- pool geometry of the N-avoiding read sampler
# (reads/sample.jl, nfree = true) over a reference fasta.
#
# The sampler's pool is the set of k-windows whose reference bytes contain no
# N (no non-ACGT byte -- sample.jl's _scan_nfree watermark rule).  This tool
# reports, per record and in total:
#
#   nchar    record length (seq chars, \n/\r skipped -- the indexreal
#            sequence-character space)
#   windows  N-free k-windows in the record (the sampler's pool; exactly the
#            w_j sample_reads reads, threaded per record)
#   clean    ACGT-only bases in the record
#   union    length of the record's clean runs of length >= k -- the genome
#            regions that can support at least one N-free k-window, i.e. the
#            union of the pool's parts (every base of such a run lies in
#            >= 1 window)
#
# and the headline: TOTAL LENGTH OF THE PARTS THE SAMPLER USES =
# (total N-free windows) * k.
#
# SELF-CONTAINED (Base + Mmap only, deliberately NOT including
# reads/sample.jl: that file `using StringDistances`, which keeps breaking on
# hosts with a stale env instantiation).  The scanner below is a verbatim
# copy of sample.jl's conventions: the fastareads_v3 record walk
# (_next_rec-style line-start '>' via memchr), the _LUT_ISBAD classification
# (bad byte = anything not ACGT/acgt) and the N-free watermark (a bad byte at
# char c poisons exactly the window starts c-k+1..c).
#
# Run (kau):  cd RotorMap && julia --project=. -t 16 tools/maizenfree.jl [fasta] [k]
#   defaults:  /share/q4bio/dandan/rotormap/data/
#              GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_primary12.fna, k = 20000
# =============================================================================
using Mmap
using Base.Threads

_mmap_fasta(file::String) =
    filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end

# the N-free pool's bad bytes: everything that is NOT a plain ACGT/acgt base
# (N/n, the other IUPAC codes, junk like a mid-line '>') -- all of it is
# silently substituted to G by sample.jl's code map, so a window touching any
# such letter is corrupted at the source.  \n/\r are skipped, never classified.
const _LUT_ISBAD = let
    lut = fill(true, 256)
    for c in "ACGTacgt"
        lut[Int(c) + 1] = false
    end
    lut
end

# Position of the first line-start '>' in [from, hi], or hi+1 (a '>' opens a
# record iff it is at position 1 or follows \n/\r).
function _next_rec(raw::AbstractVector{UInt8}, from::Int, hi::Int)
    n = length(raw)
    p = from
    p > hi && return hi + 1
    base = pointer(raw)
    NL = UInt8('\n'); CR = UInt8('\r')
    while p <= hi
        q = ccall(:memchr, Ptr{UInt8}, (Ptr{UInt8}, Cint, Csize_t),
                  base + (p - 1), Int('>'), Csize_t(hi - p + 1))
        q == C_NULL && break
        t = Int(q - base) + 1
        (t == 1 || raw[t - 1] == NL || raw[t - 1] == CR) && return t
        p = t + 1
    end
    return hi + 1
end

"""
    _ref_table(raw) -> (heads, lo, hi, nchar)

Walk the mmap'd fasta bytes record by record and return, per record: the
verbatim header line (with '>'), the inclusive byte span of its sequence and
its base count (every byte except \n/\r).
"""
function _ref_table(raw::Vector{UInt8})
    heads = String[]
    lo = Int[]
    hi = Int[]
    nchar = Int64[]
    NL = UInt8('\n'); CR = UInt8('\r')
    n = length(raw)
    s = _next_rec(raw, 1, n)
    while s <= n
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = _next_rec(raw, s + 1, n)
        cnt = 0
        @inbounds for p in (h+1):(e-1)
            b = raw[p]
            (b == NL || b == CR) || (cnt += 1)
        end
        push!(heads, String(raw[s:hend]))
        push!(lo, h + 1)
        push!(hi, e - 1)
        push!(nchar, cnt)
        s = e
    end
    return (heads, lo, hi, nchar)
end

# COUNT of N-free k-windows in [lo, hi] (sample.jl's _scan_nfree, count-only
# variant): at char c the window [c-k+1, c] closes and is N-free iff no bad
# byte sits in it, i.e. the last bad char <= c - k.
function count_nfree(raw::Vector{UInt8}, lo::Int, hi::Int, k::Int)
    NL = UInt8('\n'); CR = UInt8('\r')
    lut = _LUT_ISBAD
    cnt = 0
    lastbad = 0
    c = 0
    @inbounds for p in lo:hi
        b = raw[p]
        (b == NL || b == CR) && continue
        c += 1
        if lut[Int(b) + 1]
            lastbad = c
        elseif c >= k && lastbad <= c - k
            cnt += 1
        end
    end
    return cnt
end

# per-record clean-run scan: (total ACGT bases, sum of the lengths of the
# clean runs with length >= k).  A bad byte splits runs; \n/\r are not chars,
# so runs continue across line wraps.
function clean_runs(raw::Vector{UInt8}, lo::Int, hi::Int, k::Int)
    NL = UInt8('\n'); CR = UInt8('\r')
    lut = _LUT_ISBAD
    total = Int64(0)
    unionlen = Int64(0)
    run = 0
    @inbounds for p in lo:hi
        b = raw[p]
        if b == NL || b == CR
            continue
        end
        if lut[Int(b) + 1]
            if run >= k
                unionlen += run
            end
            run = 0
        else
            run += 1
            total += 1
        end
    end
    if run >= k
        unionlen += run
    end
    return (total, unionlen)
end

_commify(x::Integer) = replace(string(x), r"(?<=\d)(?=(\d{3})+(?!\d))" => ",")

function run_pool_stats(fasta::String, k::Int)
    isfile(fasta) || error("fasta not found: $fasta")
    raw = _mmap_fasta(fasta)
    (heads, lo, hi, nchar) = _ref_table(raw)
    R = length(nchar)
    G = sum(nchar)
    w = Vector{Int64}(undef, R)
    clean = Vector{Int64}(undef, R)
    unionlen = Vector{Int64}(undef, R)
    @threads for j in 1:R
        w[j] = count_nfree(raw, lo[j], hi[j], k)
        (clean[j], unionlen[j]) = clean_runs(raw, lo[j], hi[j], k)
    end
    W = sum(w)
    C = sum(clean)
    U = sum(unionlen)

    println("\n=== N-free k-window pool ($(basename(fasta)), k = $(_commify(k))) ===")
    println("records: $R, genome: $(_commify(G)) seq chars, " *
            "clean (ACGT) bases: $(_commify(C)) ($(round(100 * C / G; digits = 2))%)")
    println(rpad("record", 46) * lpad("nchar", 14) * lpad("N-free windows", 18) *
          lpad("clean", 14) * lpad("union >= k", 14))
    for j in 1:R
        nm = (t = split(heads[j])[1]; startswith(t, ">") ? String(t[2:end]) : String(t)) # the record NAME (first token, sans >)
        nm = length(nm) > 44 ? nm[1:44] : nm
        println(rpad(nm, 46) * lpad(_commify(nchar[j]), 14) *
              lpad(_commify(w[j]), 18) * lpad(_commify(clean[j]), 14) *
              lpad(_commify(unionlen[j]), 14))
    end
    println("--------------------------------------------------------------")
    println(rpad("TOTAL", 46) * lpad(_commify(G), 14) * lpad(_commify(W), 18) *
          lpad(_commify(C), 14) * lpad(_commify(U), 14))

    println("\n--- headline (k = $(_commify(k))) ---")
    T = W * k
    unit = T >= 1e12 ? ("Tb", T / 1e12) : ("Gb", T / 1e9)
    println("N-free windows the sampler can use:          $(_commify(W))")
    println("total length of those parts (W x k):         $(_commify(T)) bp = " *
            "$(round(unit[2]; digits = 3)) $(unit[1])")
    println("  (= $(round(100 * T / G; digits = 2))% of the $(_commify(G)) bp genome counted with overlap)")
    println("union of the regions they cover (runs >= k): $(_commify(U)) bp = " *
            "$(round(U / 1e9; digits = 3)) Gb ($(round(100 * U / G; digits = 2))% of the genome)")
    println("raw valid windows (nfree = false pool):      $(_commify(sum(max.(nchar .- k .+ 1, 0))))")
    return (G = G, W = W, C = C, U = U)
end

function main()
    fasta = length(ARGS) >= 1 ? ARGS[1] :
            get(ENV, "MAIZENFREE_FASTA",
                "/share/q4bio/dandan/rotormap/data/" *
                "GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_primary12.fna")
    k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : parse(Int, get(ENV, "MAIZENFREE_K", "20000"))
    run_pool_stats(fasta, k)
end

main()
