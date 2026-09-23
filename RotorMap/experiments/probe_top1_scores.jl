# ==============================================================================
# probe_top1_scores.jl -- distribution of the TOP-1 inner-product score and of
# the per-read MEAN TOP-20 score of the e2e_compact flow: stream the reads
# once through rope_topk_stream_compact (auto residency, the eval driver's
# path) and record, per read, the raw fp8 GEMM score of its top-1 hit
# (`TopKBatch.vals[r, 1]`), the mean over its whole top-20 score row, and
# whether the top-1 hit intersects the read's true window (map_and_count's
# provenance logic verbatim).  Reports mean / std / quantiles over ALL reads
# and over the top-1-correct subset, an ASCII histogram of the per-read
# top-20 means, and the RANK-SCORE PROFILE (mean score at each rank 1..ktop
# over all reads -- vals rows are per-read sorted, so this is the expected
# score of the j-th best match); the top1-correct count doubles as a
# cross-check against the eval's rank_hist[1] (must match bitwise: same
# flow, same order).
#
# USAGE
#   PROBE_INDEX=<index.bin> PROBE_READS=<reads.fasta> \
#     julia --project=. -t 8 experiments/probe_top1_scores.jl
# ==============================================================================

include(joinpath(@__DIR__, "e2e_compact.jl")) # the whole e2e_compact flow

using Printf
using Statistics

const PB_INDEX = get(ENV, "PROBE_INDEX", "")
const PB_READS = get(ENV, "PROBE_READS", "")
const PB_KTOP = parse(Int, get(ENV, "PB_KTOP", "20")) # top-k selection width

function probe_top1(binfile, readsfile; ktop = 20, w = 2^16, batch_size = 2^13,
                    segs = 8, engine = :lt)
    @show CUDA.name(device())
    @show nthreads()
    isfile(readsfile) || error("reads fasta not found: $readsfile")
    isfile(binfile) || error("index .bin not found: $binfile")
    @info "loading the index" binfile
    d = load_index(binfile)
    t_B = @elapsed Bh = index_database_host(d)
    heads, starts, dk = d.heads, d.starts, d.k
    dsmc = (d.s, d.m, d.c)
    d = nothing
    GC.gc()
    re = RopeEncoder(k = dk, s = dsmc[1], m = dsmc[2], c = dsmc[3])
    N = size(Bh, 2)
    # accession-keyed record ids -- map_and_count's bookkeeping verbatim
    rec_id_acc = Dict{String,Int32}()
    rec_of_col = Vector{Int32}(undef, N)
    for j in 1:N
        acc = String(split(heads[j])[1][2:end])
        id = get(rec_id_acc, acc, Int32(0))
        id == 0 && (id = Int32(length(rec_id_acc) + 1); rec_id_acc[acc] = id)
        rec_of_col[j] = id
    end

    s1 = Vector{Float32}()   # top-1 score, every read
    s1ok = Vector{Float32}() # top-1 score, top-1 hit intersects the truth
    s20 = Vector{Float32}()  # per-read MEAN of the top-20 scores
    ranksum = zeros(Float64, ktop)  # per-rank score sums over all reads
    ranksum2 = zeros(Float64, ktop) # per-rank squared-score sums
    rankmax = fill(-Inf, ktop) # per-rank maxima over all reads
    rankarg = zeros(Int, ktop) # ordinal of the read achieving rankmax[j]
    rankarghead = Vector{String}(undef, ktop)
    rankargrow = Vector{Vector{Float32}}(undef, ktop)
    total = 0
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in rope_topk_stream_compact(re, readsfile, Bh;
                                                     k = ktop, w, batch_size,
                                                     rows_cap = batch_size,
                                                     segs, engine, progress = true,
                                                     err_out = err, tasks_out = tks)
        for r in 1:size(bat.vals, 1)
            h = parse_read_head(bat.heads[r])
            h === nothing &&
                error("unreadable provenance header: $(bat.heads[r])")
            h.len === nothing || h.len == dk ||
                error("read $(h.id): len = $(h.len) != db window length $dk")
            # format-1 (sample) src has no '>' -- do NOT chop its first letter
            rid = h.len === nothing ?
                  get(rec_id_acc, h.src, Int32(0)) :
                  get(rec_id_acc, String(split(h.src)[1]), Int32(0))
            c1 = Int(bat.locs[r, 1])
            v1 = bat.vals[r, 1]
            push!(s1, v1)
            push!(s20, sum(@view(bat.vals[r, :])) / size(bat.vals, 2))
            ordinal = total + 1
            @inbounds for j in 1:ktop
                vj = Float64(bat.vals[r, j])
                ranksum[j] += vj
                ranksum2[j] += vj * vj
                if vj > rankmax[j]
                    rankmax[j] = vj
                    rankarg[j] = ordinal
                    rankarghead[j] = String(bat.heads[r])
                    rankargrow[j] = Vector{Float32}(@view(bat.vals[r, :]))
                end
            end
            total += 1
            # the top-1 hit intersects the true window?  (map_and_count's test)
            if rec_of_col[c1] == rid &&
               h.start <= starts[c1] + dk - 1 && starts[c1] <= h.start + dk - 1
                push!(s1ok, v1)
            end
        end
    end
    foreach(wait, tks[])
    err[] === nothing || error("flow failed: $(err[])")

    # rank-score profile: the mean score of the j-th best match, j = 1..ktop
    @printf("\n   rank-score profile (mean/max score of the j-th best match, all %d reads)\n", total)
    @printf("   %4s %10s %10s %10s %9s\n", "rank", "mean", "std", "max", "vs rank1")
    m1 = ranksum[1] / total
    for j in 1:ktop
        m = ranksum[j] / total
        sd = sqrt(max(ranksum2[j] / total - m * m, 0.0))
        @printf("   %4d %10.6f %10.6f %10.6f %8.2f%%\n", j, m, sd, rankmax[j],
                100 * m / m1)
    end

    # the read holding the max at the last rank + its full score row
    j = ktop
    h20 = parse_read_head(rankarghead[j])
    @printf("\n   argmax read at rank %d: ordinal %d, head `%s`\n", j,
            rankarg[j], rankarghead[j])
    h20 === nothing || @printf("   parsed provenance: src = %s, start = %s, len = %s\n",
                               h20.src, repr(h20.start), repr(h20.len))
    @printf("   its rank scores (mean %.6f):\n   ",
            sum(rankargrow[j]) / length(rankargrow[j]))
    for (jj, v) in enumerate(rankargrow[j])
        if jj == ktop
            @printf("r%-2d %.6f\n", jj, v)
        else
            @printf("r%-2d %.6f  ", jj, v)
        end
    end

    # ASCII histogram of the per-read top-20 mean scores
    lo, hi = extrema(s20)
    nbins = 40
    counts = zeros(Int, nbins)
    for v in s20
        counts[clamp(Int(floor((v - lo) / (hi - lo) * nbins)) + 1, 1, nbins)] += 1
    end
    @printf("\n   histogram of per-read mean top-20 score (n = %d, mean %.6f, std %.6f, %d bins over [%.6f, %.6f])\n",
            length(s20), mean(s20), std(s20), nbins, lo, hi)
    for b in 1:nbins
        blo = lo + (b - 1) * (hi - lo) / nbins
        bhi = lo + b * (hi - lo) / nbins
        @printf("   [%7.4f,%7.4f) %7d %5.2f%% %s\n", blo, bhi, counts[b],
                100 * counts[b] / length(s20), '█'^max(1, round(Int, 60 * counts[b] / maximum(counts))))
    end

    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    @printf("\n== TOP-1 SCORE PROBE %s\n", basename(binfile))
    @printf("   reads %s | k = %d, ktop = %d, db %d cols x rdim %d (host db build %.1f s)\n",
            readsfile, dk, ktop, N, size(Bh, 1), t_B)
    @printf("   flow wall %.1f s, %d reads, top1-correct %d (%.2f%%)\n",
            t, total, length(s1ok), 100 * length(s1ok) / total)
    for (name, x) in (("all reads", s1), ("top1-correct", s1ok))
        isempty(x) && continue
        @printf("   %-13s n=%d mean %.6f std %.6f | min %.6f q05 %.6f q25 %.6f med %.6f q75 %.6f q95 %.6f max %.6f\n",
                name, length(x), mean(x), std(x),
                (quantile(x, p) for p in qs)...)
    end
    flush(stdout)
    return nothing
end

isfile(PB_INDEX) && isfile(PB_READS) && probe_top1(PB_INDEX, PB_READS; ktop = PB_KTOP)
