# ==============================================================================
# fastasubset.jl -- keep the records of a fasta whose HEADER line matches any
# of the given regexes and copy them VERBATIM (raw bytes: header + wrapped
# sequence, no reformatting) into a new fasta file.
#
# Use case: GCF_000001405.40_GRCh38.p14_genomic.fna (the RefSeq GRCh38.p14
# analysis set, 705 records) carries 260 ALT loci + 254 patch scaffolds +
# 126 unplaced + 40 unlocalized contigs on top of the 24 primary chromosomes
# and the mitochondrion -- indexing everything would put near-duplicate ALT/
# patch copies into the mapping database (the maize top-1 tie coin-flip).
# This carves out the canonical 25-record mapping target:
#
#   julia --project=RotorMap -t 8 RotorMap/tools/fastasubset.jl \
#     /share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_genomic.fna \
#     /share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna \
#     '^>?NC_[0-9]+\.[0-9]+ Homo sapiens chromosome (1[0-9]|2[0-2]|[1-9]|X|Y), GRCh38\.p14 Primary Assembly$' \
#     'Homo sapiens mitochondrion, complete genome'
#
# The walker is the index builder's memchr line-start '>' record walk
# (_next_rec: a '>' opens a record iff at position 1 or after \n/\r; junk
# before the first record is discarded; every byte except \n/\r is a sequence
# character -- the fasta reader / packed loader convention), so the
# kept-record byte spans are exactly what the pipeline's own parsers see.
#
# Source: legacy/test/fastasubset.jl (body verbatim; standalone tool --
# stdlib only, no library-layer includes).
#
# USAGE
#   julia --project=RotorMap RotorMap/tools/fastasubset.jl <in.fasta> <out.fasta> <regex>...
#     keep every record whose header line (verbatim, with '>') matches ANY of
#     the <regex> patterns; zero matches is an error (typo guard).  The output
#     is a byte-for-byte concatenation of the kept records in file order.
# ==============================================================================

using Mmap
using Printf

# Position of the first line-start '>' in [from, hi], or hi+1 (sample.jl's
# _next_rec: a '>' opens a record iff it is position 1 or follows \n/\r).
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

# sequence characters (every byte except \n/\r) in [lo, hi]
_count_bases(raw::Vector{UInt8}, lo::Int, hi::Int) =
    count(i -> (c = raw[i]; c != UInt8('\n') && c != UInt8('\r')), lo:hi)

function main()
    ("--help" in ARGS || "-h" in ARGS) && begin
        println("usage: julia --project=RotorMap RotorMap/tools/fastasubset.jl <in.fasta> <out.fasta> <regex>...")
        println("  keep records whose header line matches ANY regex; verbatim byte copy")
        return
    end
    length(ARGS) >= 3 || error("usage: julia --project=RotorMap RotorMap/tools/fastasubset.jl <in.fasta> <out.fasta> <regex>...")
    (inn, out) = (ARGS[1], ARGS[2])
    regs = [Regex(r) for r in ARGS[3:end]]
    isfile(inn) || error("input fasta not found: $inn")
    @info "fastasubset" inn out regexes = String.(ARGS[3:end])

    raw = open(inn, "r") do io
        Mmap.mmap(io)
    end
    n = length(raw)
    kept = dropped = 0
    kept_bases = Int64(0)
    out_io = open(out, "w")
    try
        s = _next_rec(raw, 1, n)
        while s <= n
            h = s
            while h <= n && raw[h] != UInt8('\n')
                h += 1
            end
            e = _next_rec(raw, s + 1, n)
            head = String(raw[s:h-1])
            if any(r -> occursin(r, head), regs)
                nb = _count_bases(raw, h + 1, e - 1)
                GC.@preserve raw unsafe_write(out_io, pointer(raw, s), e - s)
                kept += 1
                kept_bases += nb
                @printf("%3d  %9d bases  %s\n", kept, nb, head)
            else
                dropped += 1
            end
            s = e
        end
    finally
        close(out_io)
    end
    if kept == 0
        rm(out; force = true)
        error("no record matched any regex -- removed the empty $out")
    end
    @info "fastasubset done" kept dropped kept_bases
    @info "wrote $out ($(round(filesize(out) / 2^20; digits = 1)) MiB)"
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
