# ==============================================================================
# probe_levenshtein.jl -- TRUE-LOCATION Levenshtein check for externally
# generated eval reads: sample N reads uniformly at random from the reads
# fasta, extract the reference window at each read's header-recorded location
# (colon provenance >acc:start0:rc; revcomp the read when rc = true), and
# compute the Levenshtein distance between read and window (StringDistances.jl).
#
# GOTCHAS (both bitten):
#  * the .fna records are LINE-WRAPPED -- file-byte offsets are NOT sequence
#    offsets.  The record's sequence must be made contiguous (newlines
#    stripped) BEFORE windowing at start0, else the window lands ~start0/70
#    bases early and the distances come out ~4x too big.
#  * String(v::Vector{UInt8}) ADOPTS the vector's buffer and empties it --
#    build the strings last, nothing may read the byte vectors afterwards.
#
# Standalone: no RotorMap includes, no GPU.  Needs StringDistances in the
# active environment (e.g. a scratch env: julia -e 'using Pkg;
# Pkg.activate("/tmp/levenv"); Pkg.add("StringDistances")').
#
# USAGE
#   LEV_REF=<reference.fna> LEV_READS=<reads.fasta> \
#     julia --project=/tmp/levenv experiments/probe_levenshtein.jl
#   LEV_N (100), LEV_SEED (42), LEV_ORDS ("") -- comma-separated 1-based read
#   ordinals to check INSTEAD of the random sample (NOTE: probe_top1_scores'
#   stream ordinals need NOT match file records -- match by provenance
#   instead), or LEV_HEADS ("") -- comma-separated provenance keys
#   acc:start0:rc matched against the reads' headers (file order), and
#   LEV_K (0 = off) -- trim both read and window to their first LEV_K bases
#   (file orientation; the flow encodes a read's leading k bases).
# ==============================================================================

using Mmap
using Random
using Printf
using Statistics
using StringDistances

const LEV_REF = get(ENV, "LEV_REF", "")
const LEV_READS = get(ENV, "LEV_READS", "")
const LEV_N = parse(Int, get(ENV, "LEV_N", "100"))
const LEV_SEED = parse(Int, get(ENV, "LEV_SEED", "42"))
const LEV_ORDS = get(ENV, "LEV_ORDS", "")
const LEV_HEADS = get(ENV, "LEV_HEADS", "")
const LEV_K = parse(Int, get(ENV, "LEV_K", "0")) # 0 = use full read length

const NL = UInt8('\n')
const CR = UInt8('\r')
const GT = UInt8('>')

# scan a fasta byte buffer: call f(header, lo, hi) per record (hi may point
# before lo for an empty sequence; newlines included in the span)
function _each_record(f, raw::AbstractVector{UInt8})
    n = length(raw)
    s = 1
    while s <= n && raw[s] != GT # junk before the first record
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

_upcase!(s::Vector{UInt8}) = (for i in eachindex(s)
                                  b = s[i]
                                  UInt8('a') <= b <= UInt8('z') && (s[i] = b - 32)
                              end; s)

# complement ACGT (either case); anything else passes through
_comp(b::UInt8) = b == UInt8('A') ? UInt8('T') : b == UInt8('C') ? UInt8('G') :
                  b == UInt8('G') ? UInt8('C') : b == UInt8('T') ? UInt8('A') :
                  b == UInt8('a') ? UInt8('t') : b == UInt8('c') ? UInt8('g') :
                  b == UInt8('g') ? UInt8('c') : b == UInt8('t') ? UInt8('a') : b

function _revcomp!(s::Vector{UInt8})
    i, j = 1, length(s)
    @inbounds while i < j
        ci = _comp(s[i])
        s[i] = _comp(s[j])
        s[j] = ci
        i += 1
        j -= 1
    end
    @inbounds i == j && (s[i] = _comp(s[i]))
    return s
end

function main()
    isfile(LEV_REF) || error("reference fasta not found: $LEV_REF")
    isfile(LEV_READS) || error("reads fasta not found: $LEV_READS")
    @info "mmap-ing inputs" LEV_REF LEV_READS
    ref = open(LEV_REF, "r") do io
        Mmap.mmap(io)
    end
    rreads = open(LEV_READS, "r") do io
        Mmap.mmap(io)
    end

    # reference: accession -> (lo, hi) sequence span
    span = Dict{String,UnitRange{Int}}()
    _each_record(ref) do head, lo, hi
        acc = String(split(head)[1])
        haskey(span, acc) && error("duplicate reference accession: $acc")
        span[acc] = lo:hi
    end
    @info "reference records" n=length(span)

    # reads: (acc, start0, rc, seq)
    raccs = String[]
    rstarts = Int[]
    rrcs = Bool[]
    rkeys = String[] # provenance key acc:start0:rc (pre-#, exact head match)
    rseqs = Vector{Vector{UInt8}}()
    _each_record(rreads) do head, lo, hi
        f = split(head, ':')
        length(f) == 3 || error("unreadable provenance header: $head")
        rc = split(f[3], '#')[1] == "true"
        seq = Vector{UInt8}(@view rreads[lo:hi])
        filter!(b -> b != NL && b != CR, seq)
        push!(raccs, String(f[1]))
        push!(rstarts, parse(Int, f[2]))
        push!(rrcs, rc)
        push!(rkeys, string(f[1], ":", f[2], ":", split(f[3], '#')[1]))
        push!(rseqs, seq)
    end
    nreads = length(raccs)
    @info "read records" nreads readlen=length(rseqs[1])

    rng = MersenneTwister(LEV_SEED)
    pick = if !isempty(LEV_HEADS)
        targets = Set(split(LEV_HEADS, ","))
        found = [r for r in 1:nreads if rkeys[r] in targets]
        missing = setdiff(targets, Set(rkeys[r] for r in found))
        isempty(missing) || error("LEV_HEADS not found in the reads: $missing")
        found
    elseif !isempty(LEV_ORDS)
        sort!(parse.(Int, split(LEV_ORDS, ",")))
    else
        sort!(randperm(rng, nreads)[1:LEV_N])
    end
    all(r -> 1 <= r <= nreads, pick) || error("pick out of range 1..$nreads")

    # contiguous record sequences (newlines stripped), built lazily per accession
    rec_cache = Dict{String,Vector{UInt8}}()
    levs = Int[]
    clens = Int[] # COMPARED lengths (post-LEV_K-trim) for the summary line
    println("i  accession       start0     rc     len   levenshtein   lev%")
    for (i, r) in enumerate(pick)
        acc, start0, rc, seq = raccs[r], rstarts[r], rrcs[r], rseqs[r]
        haskey(span, acc) || error("read $i: accession $acc not in the reference")
        len = length(seq)
        if LEV_K > 0 && len > LEV_K # first-LEV_K trim, file orientation
            seq = seq[1:LEV_K]
            len = LEV_K
        end
        rec = get!(rec_cache, acc) do
            r = span[acc]          # UnitRange: (rlo, rhi = r) would destructure
            rlo, rhi = r.start, r.stop # its first two ELEMENTS (106, 107)!!
            v = Vector{UInt8}(uppercase(String(ref[rlo:rhi])))
            filter!(b -> (b != NL && b != CR), v) # contiguity: THE offset fix
            v
        end
        (start0 + len <= length(rec)) ||
            error("read $i: window at start0=$start0 (len $len) overruns $acc ($(length(rec)) bases)")
        win = rec[(start0 + 1):(start0 + len)] # range indexing copies
        _upcase!(seq)
        rc && _revcomp!(seq)
        # String() adopts (and empties) the byte vectors -- last use, by design
        d_lev = evaluate(Levenshtein(), String(win), String(seq))
        push!(levs, d_lev)
        push!(clens, len)
        @printf("%-2d %-15s %10d %-6s %6d %11d %6.2f%%\n",
                i, acc, start0, rc ? "true" : "false", len, d_lev, 100 * d_lev / len)
        i % 10 == 0 && flush(stdout)
    end
    println()
    @printf("SUMMARY n=%d | levenshtein mean %.1f med %d min %d max %d std %.1f | as %% of readlen: mean %.2f%% min %.2f%% max %.2f%%\n",
            length(levs), mean(levs), median(levs), minimum(levs), maximum(levs), std(levs),
            100 * sum(levs) / sum(clens),
            100 * minimum(levs .// clens), 100 * maximum(levs .// clens))
    flush(stdout)
end

isfile(LEV_REF) && isfile(LEV_READS) && main()
