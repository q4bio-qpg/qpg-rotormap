# ==============================================================================
# header_stats.awk -- top1 / any-match accuracy statistics for minimap2 PAF
# output whose query names (column 1) carry ground truth in the header:
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
#   |header position - column 8| < 20000
#
# (column 8 = target start = the "7th column" of the 0-based PAF layout;
# columns 2-5 are query length/start/end/strand, so the literal 1-based
# column 7 is the target LENGTH, not a position). Inconsistent records
# count toward neither statistic.
#
# Run on kau (state resets per file, one report line per file):
#
#   awk -f RotorMap/tools/paf/header_stats.awk \
#     /share/q4bio/maksym/rotormap/mapping/minimap2/*.paf
#
# Output (stdout): per file
#   <file> unique=<N> top1=<N> (<pct>%) | any=<N> (<pct>%)
#
# Source: RotorMap dev session 2026-09 (ad hoc kau /tmp/analyze.awk script,
# consolidated; standalone tool -- POSIX awk, no deps).
# ==============================================================================

function report() {
    any = 0
    for (h in correct) any++
    printf "%-40s unique=%-8d top1=%-8d (%.2f%%) | any=%-8d (%.2f%%)\n", \
        fname, uniq, top1, 100 * top1 / uniq, any, 100 * any / uniq
    uniq = top1 = 0
    delete correct
    delete seen1
}

# report the previous file when a new one starts
FNR == 1 {
    if (NR > 1) report()
    fname = FILENAME
}

{
    # uniqueness + first-occurrence flag (file-scoped, reset in report())
    first = !($1 in seen1)
    seen1[$1] = 1
    if (first) uniq++

    # parse the header: seq:position:rc (seq may itself contain colons)
    n = split($1, a, ":")
    seq = a[1]
    pos = a[n - 1] + 0
    rc = a[n]

    # consistency with the alignment record
    ok = (seq == $6) && ((rc == "true" && $5 == "-") || (rc == "false" && $5 == "+"))
    if (!ok) next

    d = pos - $8
    if (d < 0) d = -d
    if (d < 20000) {
        correct[$1] = 1
        if (first) top1++
    }
}

END {
    if (NR > 0) report()
}
