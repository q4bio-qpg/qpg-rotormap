# ==============================================================================
# header_stats.jl -- Julia equivalent of header_stats.awk (same directory, same
# semantics, same output format): top1 / any-match accuracy statistics for
# minimap2 PAF output whose query names (column 1) carry ground truth:
#
#   seq:position:rc        e.g.  NC_000001.11:80433546:false
#
#   seq      reference sequence the read was drawn from;
#   position 1-based reference position the read starts at;
#   rc       true/false, whether the read is the reverse complement.
#
# Several PAF lines may share one query header (multiple alignments). For
# every unique header (first occurrence defines "first"):
#
#   top1  +1 iff the FIRST record with that header is a correct match;
#   any   +1 iff ANY record with that header is a correct match.
#
# A record is a correct match when it is consistent with its header
#
#   * header seq  == column 6  (target sequence name);
#   * header rc   == column 5  (strand): true -> "-", false -> "+";
#
# and the found position is close enough to the true one:
#
#   |header position - column 8| < THRESHOLD      (default 20000)
#
# (column 8 = target start = the "7th column" of the 0-based PAF layout;
# the literal 1-based column 7 is the target LENGTH, not a position).
# Inconsistent records count toward neither statistic.
#
# Run on kau (one report line per file, AWK-column-compatible):
#
#   julia --project=RotorMap RotorMap/tools/paf/header_stats.jl \
#     /share/q4bio/maksym/rotormap/mapping/minimap2/*.paf
#
# Config (env override): PAFHDR_THRESHOLD (default 20000).
#
# Source: Julia port of tools/paf/header_stats.awk (dev session 2026-09);
# standalone tool -- stdlib only.
# ==============================================================================

using Printf

const THRESHOLD = parse(Int, get(ENV, "PAFHDR_THRESHOLD", "20000"))

# process one PAF file -> (unique headers, top1 matches, any matches)
function _process(path::AbstractString, threshold::Int)
    seen = Set{String}()       # all headers encountered so far
    correct = Set{String}()    # headers with at least one correct match
    uniq = top1 = 0
    for line in eachline(path)
        cols = split(line, '\t')
        length(cols) >= 8 || continue
        name = cols[1]
        first = !(name in seen)
        push!(seen, name)
        first && (uniq += 1)

        parts = split(name, ':')           # seq:position:rc (seq may hold colons)
        length(parts) >= 3 || continue
        pos = tryparse(Int, parts[end - 1])
        (pos === nothing) && continue
        seq, rc = parts[1], parts[end]

        strand = cols[5]
        ((seq == cols[6]) &&
         ((rc == "true" && strand == "-") || (rc == "false" && strand == "+"))) || continue

        if abs(pos - parse(Int, cols[8])) < threshold
            push!(correct, name)
            first && (top1 += 1)
        end
    end
    return uniq, top1, length(correct)
end

function main()
    if isempty(ARGS)
        println(stderr, "usage: julia --project=RotorMap RotorMap/tools/paf/header_stats.jl <paf> [paf...]")
        return nothing
    end
    t = @elapsed for f in ARGS
        uniq, top1, anym = _process(f, THRESHOLD)
        if uniq == 0
            @printf("%-40s (no records)\n", basename(f))
            continue
        end
        @printf("%-40s unique=%-8d top1=%-8d (%.2f%%) | any=%-8d (%.2f%%)\n",
                basename(f), uniq, top1, 100 * top1 / uniq, anym, 100 * anym / uniq)
    end
    @printf("ALL DONE in %.2f s\n", t)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
