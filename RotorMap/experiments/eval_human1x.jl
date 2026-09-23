# ==============================================================================
# eval_human1x.jl -- instrumented EVAL driver for maksym's human chrX eval
# batch (colon provenance headers, >acc:start:rc) through the e2e_compact
# flow.  Per index it reports:
#   * top20   -- reads whose true window intersects ANY of the top-20 hits
#   * top1    -- reads whose FIRST hit intersects (rank_hist[1])
#   * avg20   -- the average BEST top-20 overlap (bases, and % of the k
#                fragment) over the mapped reads
#   * t_load / t_build -- host index load + fp8-database build
#   * map_s per rep    -- map_and_count wall time (rep1 pays the process's
#                one-time pipeline JIT, rep2 is warm)
#   * e2e_warm_s = t_load + t_build + warm map_s
#   * index .bin size (GB / GiB)
#   * PEAK VRAM -- 20 ms CUDA.free_memory poller, driver-level used =
#                total - min free (our process owns the GPU)
# GPU busy/compute time comes from the nvidia-smi sampler the calling wrapper
# runs alongside: PHASE markers on stderr carry wall-clock stamps for the
# per-rep windows (parse them against the smi log's timestamp column).
#
# Both GRCh38 primary25 indexes (s5m1c5 ~5.9 GiB, s4m1c4 ~1.5 GiB) fit the
# RTX 5090's 32 GiB, so the compact flow auto-delegates to the resident
# engine (one upload per map call, no per-batch PCIe sweep); the log states
# the residency decision either way.
#
# USAGE (conventions of e2e_compact.jl; -t 8 = the eval's thread count)
#   julia --project=. -t 8 experiments/eval_human1x.jl [binfile ...]
#   (no args = the two GRCh38 primary25 fp8 indexes; reads: E1X_READS env
#   override, default maksym's human_1X_5.fasta)
# ==============================================================================

include(joinpath(@__DIR__, "e2e_compact.jl")) # the whole e2e_compact flow

using Dates

const E1X_READS = get(ENV, "E1X_READS",
                      "/share/q4bio/maksym/rotormap/generated_data/eval/human_1X_5.fasta")
const E1X_BINS = [
    "/share/q4bio/dandan/rotormap/data/" *
    "GCF_000001405.40_GRCh38.p14_primary25.indexreal_fp8_s5m1c5.bin",
    "/share/q4bio/dandan/rotormap/data/" *
    "GCF_000001405.40_GRCh38.p14_primary25.indexreal_fp8_s4m1c4.bin",
]

_now() = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS.sss")
phase(msg...) = println(stderr, "PHASE ", join(msg, " "), " ", _now())

# the eval's short index tag: the trailing _s<m>c<c> of the file name
_e1x_tag(binfile) =
    (m = match(r"_s(\d)m(\d)c(\d)\.bin$", basename(binfile))) === nothing ?
    basename(binfile) : "s$(m[1])m$(m[2])c$(m[3])"

"""
    eval_index(binfile; ktop = 20, w = 2^16, batch_size = 2^13, segs = 8,
               engine = :lt, reps = 2)

One index through the e2e_compact flow (auto residency): load -> host fp8
database -> `reps` x map_and_count over E1X_READS with the summary metrics
above, then a machine-readable EVALRESULT row (the warm rep's numbers).
"""
function eval_index(binfile; ktop = 20, w = 2^16, batch_size = 2^13, segs = 8,
                    engine = :lt, reps = 2)
    isfile(E1X_READS) || error("reads fasta not found: $E1X_READS")
    isfile(binfile) || error("index .bin not found: $binfile")
    tag = _e1x_tag(binfile)
    @show CUDA.name(device())
    @show nthreads()

    phase("load_start idx=", tag)
    t_load = @elapsed d = load_index(binfile)
    t_B = @elapsed Bh = index_database_host(d)
    heads, starts, dk, kstep, dsmc = d.heads, d.starts, d.k, d.kstep,
                                     (d.s, d.m, d.c)
    d = nothing # keep heads/starts/Bh
    GC.gc()
    re = RopeEncoder(k = dk, s = dsmc[1], m = dsmc[2], c = dsmc[3])
    go_resident = resident_fits(Bh; rows_cap = batch_size, rdim = size(Bh, 1), w)
    vtotal = CUDA.totalmem(device())
    vfree0 = CUDA.free_memory()
    phase("map_start idx=", tag, " resident=", go_resident,
          " free0_gib=", round(vfree0 / 2^30; digits = 2))

    vmin = Ref(vfree0) # peak-VRAM poller: driver used = vtotal - min free
    stop = Ref(false)
    sampler = Threads.@spawn begin
        while !stop[]
            f = CUDA.free_memory()
            f < vmin[] && (vmin[] = f)
            sleep(0.02)
        end
    end

    # The RESIDENT engine is built ONCE and all reps stream through it: a
    # per-rep rebuild would re-upload B -- and re-check residency against its
    # own still-live copy, which cannot fit for any index over ~half of VRAM
    # (maize k4000 s5m1c5: need 22.5 GiB vs the 31.4 - 22.5 = 8.3 GiB left).
    # The upload lands under the VRAM poller; rep1 pays it + JIT, rep2 is a
    # true warm pass.  Ring path (go_resident = false) keeps the per-rep
    # rope_topk_stream_compact call unchanged.
    eng = nothing
    _resident_stream = nothing
    if go_resident
        need = _resident_need(Bh; rows_cap = batch_size, rdim = size(Bh, 1), w)
        GC.gc(true)
        free = (CUDA.reclaim(); CUDA.free_memory())
        need <= free ||
            error("resident engine: need $(round(need / 2^30; digits = 1)) GiB, " *
                  "only $(round(free / 2^30; digits = 1)) GiB free")
        eng = TopKEngine(CuMatrix{F8}(Bh); k = ktop, w, rows_cap = batch_size,
                         segs, engine, cout = :f16)
        _resident_stream = function (re_, f, B; k = 20, w = 2^16,
                                     batch_size = 2^13, rows_cap = batch_size,
                                     normalize = 0, parts = Threads.nthreads(),
                                     in_cap = 2, out_cap = 2, topk_out_cap = 2,
                                     segs = 8, engine = :lt, cout = :f16,
                                     depth = 3, resident = nothing,
                                     progress = false,
                                     err_out = Ref{Any}(nothing),
                                     tasks_out = Ref{Vector{Task}}(Task[]))
            # rope_topk_stream_compact's resident branch, verbatim, but with
            # the SHARED engine (no per-call upload / residency re-check)
            rope_ch = rope_encode_real_stream(re_, f; k = re_.k, batch_size,
                                              normalize, fp16 = true, parts,
                                              in_cap, out_cap, progress,
                                              err_out, tasks_out)
            return topk_flow(rope_ch, eng; out_cap = topk_out_cap,
                             err_out, tasks_out)
        end
    end

    rs = []
    for rep in 1:reps
        rep > 1 && (GC.gc(true); CUDA.reclaim()) # hand rep-1's pool blocks back
        phase("rep$(rep)_start idx=", tag)
        t = @elapsed res = map_and_count(re, E1X_READS, Bh, heads, starts, dk;
                                         ktop, w, batch_size, segs, engine,
                                         progress = true,
                                         stream = eng === nothing ?
                                             (re_, f, B; kw...) ->
                                                 rope_topk_stream_compact(re_, f, B;
                                                     kw..., resident = go_resident) :
                                             _resident_stream)
        phase("rep$(rep)_end idx=", tag, " map_s=", round(t; digits = 1))
        push!(rs, (t, res))
    end
    stop[] = true
    wait(sampler)
    vpeak = vtotal - vmin[]

    fgb = filesize(binfile)
    println()
    @printf("== EVAL %s: file %.2f GB (%.2f GiB), db %d cols x rdim %d, k = %d, kstep = %d, encoder s = %d m = %d c = %d\n",
            tag, fgb / 1e9, fgb / 2^30, size(Bh, 2), size(Bh, 1), dk, kstep, dsmc...)
    @printf("   reads %s | load %.1f s, host db build %.1f s | resident = %s\n",
            E1X_READS, t_load, t_B, go_resident)
    for (rep, (t, r)) in enumerate(rs)
        @printf("   rep%d: map %.1f s | top20 %d/%d = %.2f%% | top1 %d = %.2f%% | avg20 %.0f bases = %.2f%% of k\n",
                rep, t, r.correct, r.total, 100 * r.correct / r.total,
                r.rank_hist[1], 100 * r.rank_hist[1] / r.total,
                r.inter_sum / r.inter_cnt, 100 * (r.inter_sum / r.inter_cnt) / dk)
    end
    @printf("   peak VRAM %.2f GiB of %.2f GiB total (20 ms free-memory poll)\n",
            vpeak / 2^30, vtotal / 2^30)
    total_e2e = t_load + t_B + rs[end][1]
    @printf("   total e2e (load + build + warm map) = %.1f s\n", total_e2e)
    println()

    r = rs[end][2] # the warm rep -> the machine-readable row
    # (one literal format string: @printf parses it at macro time, so a "a" * "b"
    # concatenation would be misread as the (io, fmt) form and throw)
    @printf("EVALRESULT idx=%s file_gb=%.3f gib=%.3f k=%d kstep=%d cols=%d rdim=%d reads=%d load_s=%.1f build_s=%.1f resident=%d map1_s=%.1f map2_s=%.1f top20_pct=%.3f top1_pct=%.3f avg20_bases=%.1f avg20_pct=%.3f vram_peak_gib=%.2f vram_total_gib=%.2f e2e_warm_s=%.1f\n",
            tag, fgb / 1e9, fgb / 2^30, dk, kstep, size(Bh, 2), size(Bh, 1),
            r.total, t_load, t_B, go_resident, rs[1][1], rs[end][1],
            100 * r.correct / r.total, 100 * r.rank_hist[1] / r.total,
            r.inter_sum / r.inter_cnt, 100 * (r.inter_sum / r.inter_cnt) / dk,
            vpeak / 2^30, vtotal / 2^30, total_e2e)
    flush(stdout)
    Bh = nothing
    heads = starts = nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

"Run the eval: ARGS bin paths, or both GRCh38 defaults."
function run_eval()
    bins = filter(a -> !startswith(a, "--"), ARGS)
    isempty(bins) && (bins = E1X_BINS)
    for b in bins
        eval_index(b)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_eval()
end
