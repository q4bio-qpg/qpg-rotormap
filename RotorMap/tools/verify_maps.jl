# =============================================================================
# verify_maps.jl -- score tools/map.jl's match table against the sample
# provenance (reads/sample.jl `gen` headers), the same statistics the e2e
# flows' map_and_count collects:
#
#   * top-1 accuracy: reads whose FIRST match line points at the header's
#     true locus (same record + k-long window overlap)
#   * top-k accuracy: reads whose top-of-table (any rank) match does
#     ("correctly mapped" in map_and_count's sense)
#   * first-hit rank histogram: rank_hist[j] = reads whose first intersecting
#     hit was rank j
#   * average overlap: over mapped reads, the mean (over reads) of the BEST
#     (maximum) k-long window overlap length over ALL that read's hits, and
#     the same as a share of the window length k
#
#   julia verify_maps.jl <reads.fasta> <out.maps> [K]
#
# K defaults to the first provenance header's `len=` (map.jl's match table
# carries no explicit k; the index window length is what the overlap test
# needs).  Reads absent from the provenance (plain ids, no key=val fields)
# are skipped quietly.  The line grammar is map.jl's:
#     <read header> <'+'|'-'> <score> <0-based position> <record name>
# with the read header possibly multi-token: the line is split around the
# BARE '+'/'-' direction token, everything after the position is the record
# name field (its leading '>' tolerated).  Lines of one read's block are
# ordered by score descending (map.jl's contract), so a hit's rank is the
# block-internal line index.
# =============================================================================

using Printf

function main()
    # ---- the provenance table: read id -> (src record name, 1-based start, len)
    prov = Dict{String, Tuple{String, Int, Int}}()
    for line in eachline(ARGS[1])
        startswith(line, ">") || continue
        toks = split(strip(line[2:end]), " "; keepempty = false)
        id = String(toks[1]) # the bare id token (sample format)
        src = ""; start = 0; len = 0
        for t in toks[2:end]
            kv = split(t, "="; limit = 2)
            length(kv) == 2 || continue
            kv[1] == "id" && (id = String(kv[2]))
            kv[1] == "src" && (src = String(kv[2]))
            kv[1] == "start" && (start = parse(Int, kv[2]))
            kv[1] == "len" && (len = parse(Int, kv[2]))
        end
        if start == 0 && src == "" && occursin(":", id)
            # the eval flow's colon provenance: >acc:0-based-start:rc[#dup]
            # (the reads/sample.jl parser's format 2: read length unknown --
            # the reader trims every record to exactly k, so the mapped
            # fragment is the db window at start; the optional trailing
            # #<n> duplicate-locus marker is ignored; start is 0-based)
            f = split(id, ":")
            length(f) == 3 || continue
            s0 = tryparse(Int, f[2])
            s0 === nothing && continue
            src = String(f[1])
            start = s0 + 1
        end
        (start > 0 || src != "") && (prov[id] = (src, start, len))
    end

    K = length(ARGS) > 2 ? parse(Int, ARGS[3]) :
        first(values(prov))[3] # a provenance header's len (0 for colon reads)
    K > 0 || error("colon-provenance reads carry no len; pass K=$K ... explicitly")

    # ---- scan the match table: blocks of lines sharing a read header --------
    top1 = correct = n = skipped = 0
    inter_sum = 0
    hitranks = Int[] # lazily grown: hitranks[j] = reads whose first hit was rank j
    cur = "\0"                 # the read header being accumulated
    cur_provenanced = false
    first_rank = 0             # that read's first intersecting hit's rank
    best_overlap = 0           # that read's best overlap over ALL hits
    rank_in_block = 0

    settle! = function ()
        # fold the settled read's stats in (a mapped read keeps the BEST
        # overlap over ALL its ktop hits -- exactly map_and_count's inter_sum)
        if cur_provenanced
            if first_rank > 0
                correct += 1
                top1 += first_rank == 1 ? 1 : 0 # rank 1 = top-1 accuracy
                inter_sum += best_overlap
                length(hitranks) < first_rank &&
                    append!(hitranks, zeros(Int, first_rank - length(hitranks)))
                hitranks[first_rank] += 1
            end
        end
    end

    for line in eachline(ARGS[2])
        f = split(line)
        i = findfirst(t -> t == "+" || t == "-", f)
        i === nothing &&
            error("no bare '+'/'-' direction token in the match line: $line")
        readid = String(f[1])
        startswith(readid, ">") && (readid = String(readid[2:end])) # sans '>'
        if readid != cur # a new read's block: settle the previous one
            settle!()
            cur_provenanced = haskey(prov, readid)
            cur_provenanced || (skipped += 1)
            n += 1
            cur = readid
            first_rank = 0
            best_overlap = 0
            rank_in_block = 0
        end
        cur_provenanced || continue # unprovenanced: skip quietly
        rank_in_block += 1
        pos0 = parse(Int, f[i+2]) # 0-based position (map.jl's field 4)
        dbf = join(f[i+3:end], " ")
        recname = (t = split(dbf)[1];
                   startswith(t, ">") ? String(t[2:end]) : String(t)) # sans '>'
        (src, start, lk) = prov[readid]
        kk = lk > 0 ? lk : K
        (recname == src && pos0 + 1 <= start + kk - 1 && start <= pos0 + kk) ||
            continue
        # an intersecting hit: [pos0+1, pos0+kk] vs [start, start+kk-1] overlap
        ol = min(start, pos0 + 1) + kk - 1 - max(start, pos0 + 1) + 1
        best_overlap = max(best_overlap, ol)
        first_rank == 0 && (first_rank = rank_in_block)
    end
    settle!() # the last block
    isempty(prov) && error("no provenance headers parsed from $(ARGS[1])")
    n == 0 && error("no reads found in the match table $(ARGS[2])")
    @printf("%d provenance reads checked (%d unprovenanced skipped)\n", n, skipped)
    @printf("  top-1 accuracy:  %d/%d = %.2f%%\n", top1, n, 100top1 / n)
    @printf("  top-k accuracy: %d/%d = %.2f%%\n", correct, n, 100correct / n)
    @printf("  first-hit rank histogram (ranks 1..%d): %s\n", length(hitranks),
            join(hitranks, ", "))
    @printf("  best intersection: avg %d bases over %d mapped reads = %.2f%% of the %d-base window\n",
            round(Int, inter_sum / correct), correct, 100 * inter_sum / correct / K, K)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
