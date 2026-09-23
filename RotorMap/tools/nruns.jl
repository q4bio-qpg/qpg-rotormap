# ==============================================================================
# nruns.jl -- collect the sizes of ALL continuous runs of N letters in a FASTA
# (built for and checked against the human reference genome GRCh38.p14 primary
# assembly, 68 records, ~3.1 Gb). Runs are MAXIMAL stretches of consecutive
# N/n bases within one record; the 60-column line wrapping is transparent (a
# run continues across newlines), headers break runs. Runs are never merged
# across records. The scan is a single raw-text pass over an mmap of the file
# (the to_packed_bin.jl verification approach -- the packed loaders substitute
# N/n -> G and cannot recover run positions). Run on kau:
#
#   julia --project=RotorMap RotorMap/tools/nruns.jl
#
# Outputs:
#   * stdout summary: per-record N statistics, run-size histogram, top runs;
#   * <fasta>.nruns.csv -- one line per run: record,start,length (1-based
#     start within the record), in file order.
#
# Config (env overrides): NRUNS_FASTA, NRUNS_OUT.
#
# Source: legacy/test/nruns.jl (body verbatim; standalone tool -- stdlib only,
# no library-layer includes; _comma kept local so it stays that way).
#
# RESULTS (kau, single thread, 2026): 3.31 s scan (0.95 GB/s of FASTA); 68
# records, 3,095,453,524 bases (asserted), 150,965,758 N/n (4.877%), 99 other
# IUPAC letters (both cross-checked against to_packed_bin.jl's known counts and the
# N total re-counted independently with awk). 942 maximal N-runs covering all
# 150,965,758 N bases. Run lengths: median 100, p95 60,000, max 30,000,000
# (chrY centromere at 26,673,215), mean ~160,261. 9 runs >= 1 Mb carry
# 89.8% of all N bases (per-chromosome centromere/first-scaffold gaps: chrY
# 30 Mb, chr1 18 Mb at 125,184,588, chr15/13/14 ~16-17 Mb at position 1, chr9
# 15 Mb, chr22 10.5 Mb, chr16 8.1 Mb, chr21 5 Mb); the long tail is mostly
# 100-base scaffold-junction runs (369 in 11-100) and 119 single-N records
# (369 in 11-100) and 119 single-N records (the mt genome J01415.2 carries
# exactly one N, per RefSeq convention). Full list:
# <fasta>.nruns.csv (942 rows: record,start,length).
# ==============================================================================

using Mmap
using Printf

const FASTA = get(ENV, "NRUNS_FASTA",
                  "/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna")
const OUT = get(ENV, "NRUNS_OUT", string(splitext(FASTA)[1], ".nruns.csv"))

const _NL = UInt8('\n')
const _CR = UInt8('\r')
const _GT = UInt8('>')

function _comma(n::Integer)
    s = string(n)
    out = IOBuffer()
    for (i, c) in enumerate(s)
        print(out, c)
        from_end = length(s) - i
        (from_end > 0 && from_end % 3 == 0) && print(out, ',')
    end
    return String(take!(out))
end

function main()
    @printf("== nruns.jl: %s ==\n", basename(FASTA))
    @printf("source: %s (%s bytes)\n", FASTA, _comma(filesize(FASTA)))
    @assert isfile(FASTA) "FASTA not found: $FASTA"

    raw = open(FASTA, "r") do io
        Mmap.mmap(io)
    end

    # ---- single-pass run scan ------------------------------------------------
    # state: current record (index + 1-based base position), the open run
    # (start position, length; 0 = none), per-record accumulators. Newlines are
    # transparent (wrapped lines), '>' opens a header (breaks runs).
    # plain locals only (no closures: they would box the loop state)
    rec = 0
    pos = 0
    run_start = 0
    run_len = 0
    nrec_bases = 0
    nrec_ns = 0
    nrec_runs = 0
    nrec_longest = 0
    nacgt_total = 0
    nother_total = 0 # non-ACGTN letters (IUPAC B/K/M/R/S/W/Y)
    heads = String[]
    rec_bases = Int[]
    rec_ns = Int[]
    rec_runs = Int[]
    rec_longest = Int[]
    run_rec = Int[]
    run_start_all = Int[]
    run_len_all = Int[]
    in_header = true
    hstart = 0
    ACGTN = (UInt8('A'), UInt8('C'), UInt8('G'), UInt8('T'),
             UInt8('a'), UInt8('c'), UInt8('g'), UInt8('t'))

    t = @elapsed @inbounds for i in eachindex(raw)
        c = raw[i]
        if in_header
            if c == _NL
                in_header = false
                rec += 1
                pos = 0
                push!(heads, String(raw[hstart:i-1]))
            elseif hstart == 0
                hstart = i # first '>' of the record
            end
        elseif c == _NL || c == _CR
            # line wrap: transparent, runs continue across it
        elseif c == _GT
            # record boundary: close the open run and the finished record
            if run_len > 0
                push!(run_rec, rec); push!(run_start_all, run_start)
                push!(run_len_all, run_len); nrec_runs += 1
                run_len > nrec_longest && (nrec_longest = run_len)
                run_start = 0; run_len = 0
            end
            if rec > 0
                push!(rec_bases, nrec_bases); push!(rec_ns, nrec_ns)
                push!(rec_runs, nrec_runs); push!(rec_longest, nrec_longest)
            end
            nrec_bases = 0; nrec_ns = 0; nrec_runs = 0; nrec_longest = 0
            in_header = true
            hstart = 0
        else
            pos += 1
            nrec_bases += 1
            if c == UInt8('N') || c == UInt8('n')
                run_len == 0 && (run_start = pos)
                run_len += 1
                nrec_ns += 1
            else
                if run_len > 0 # close the open run
                    push!(run_rec, rec); push!(run_start_all, run_start)
                    push!(run_len_all, run_len); nrec_runs += 1
                    run_len > nrec_longest && (nrec_longest = run_len)
                    run_start = 0; run_len = 0
                end
                c in ACGTN ? (nacgt_total += 1) : (nother_total += 1)
            end
        end
    end
    if run_len > 0 # close the final run + record at EOF
        push!(run_rec, rec); push!(run_start_all, run_start)
        push!(run_len_all, run_len); nrec_runs += 1
        run_len > nrec_longest && (nrec_longest = run_len)
    end
    if rec > 0
        push!(rec_bases, nrec_bases); push!(rec_ns, nrec_ns)
        push!(rec_runs, nrec_runs); push!(rec_longest, nrec_longest)
    end

    nrec = length(heads)
    nruns = length(run_len_all)
    nbase_total = sum(rec_bases)
    nN_total = sum(rec_ns)
    @assert nacgt_total + nN_total + nother_total == nbase_total "letter bookkeeping broken"
    @printf("scan: %.2f s (%.2f GB/s of FASTA)\n", t, filesize(FASTA) / t / 1e9)
    @printf("records: %s   bases: %s   ACGT: %s   N/n: %s (%.3f%%)   other IUPAC letters: %s\n",
            _comma(nrec), _comma(nbase_total), _comma(nacgt_total), _comma(nN_total),
            100 * nN_total / nbase_total, _comma(nother_total))
    @printf("N-runs: %s   bases in runs: %s\n", _comma(nruns), _comma(sum(run_len_all)))
    @assert nbase_total == 3_095_453_524 "unexpected total base count for GRCh38.p14: $nbase_total"

    # ---- run-size statistics -------------------------------------------------
    slens = sort(run_len_all)
    q(p) = slens[max(1, round(Int, p * nruns))]
    @printf("run length: min %s   p25 %s   median %s   p75 %s   p95 %s   max %s   mean %.1f\n",
            _comma(slens[1]), _comma(q(0.25)), _comma(q(0.5)), _comma(q(0.75)),
            _comma(q(0.95)), _comma(slens[end]), sum(run_len_all) / nruns)

    edges = [1, 2, 3, 11, 101, 1_001, 10_001, 100_001, 1_000_001, typemax(Int)]
    labels = ["1", "2", "3-10", "11-100", "101-1e3", "1e3-1e4", "1e4-1e5",
              "1e5-1e6", ">=1e6"]
    @printf("\nhistogram (run lengths):\n  %-10s %12s %16s %8s\n", "bucket", "runs", "bases in runs", "share")
    for b in eachindex(labels)
        lo, hi = edges[b], edges[b + 1] - (edges[b + 1] == typemax(Int) ? 0 : 1)
        idxs = filter(i -> lo <= run_len_all[i] <= hi, eachindex(run_len_all))
        nb = sum(view(run_len_all, idxs); init = 0)
        @printf("  %-10s %12s %16s %7.2f%%\n", labels[b], _comma(length(idxs)),
                _comma(nb), 100 * nb / max(nN_total, 1))
    end

    @printf("\ntop 20 largest runs:\n  %4s %4s %-58s %14s %12s\n",
            "rank", "rec", "head", "start", "length")
    top = sortperm(run_len_all; rev = true)[1:min(20, nruns)]
    for (rank, i) in enumerate(top)
        h = heads[run_rec[i]]
        @printf("  %4d %4d %-58s %14s %12s\n", rank, run_rec[i],
                length(h) > 58 ? h[1:55] * "..." : h,
                _comma(run_start_all[i]), _comma(run_len_all[i]))
    end

    @printf("\nper-record N statistics:\n  %4s %-56s %14s %12s %8s %10s %12s\n",
            "rec", "head", "length", "N/n", "N%", "runs", "longest run")
    for r in 1:nrec
        h = heads[r]
        @printf("  %4d %-56s %14s %12s %7.2f%% %10s %12s\n", r,
                length(h) > 56 ? h[1:53] * "..." : h,
                _comma(rec_bases[r]), _comma(rec_ns[r]), 100 * rec_ns[r] / rec_bases[r],
                _comma(rec_runs[r]), _comma(rec_longest[r]))
    end

    # ---- full run list -> CSV -------------------------------------------------
    open(OUT, "w") do io
        println(io, "record,start,length")
        for i in eachindex(run_len_all)
            println(io, run_rec[i], ',', run_start_all[i], ',', run_len_all[i])
        end
    end
    @printf("\nfull run list (%s runs): %s\n", _comma(nruns), OUT)
    @printf("ALL DONE in %.2f s\n", t)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
