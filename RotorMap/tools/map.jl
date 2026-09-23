# =============================================================================
# map.jl -- the plain mapping CLI: env INDEX + READS -> a whitespace table of
# raw top-k matches + scores, NO provenance scoring (the reads MAY lack
# provenance headers -- nothing here parses one).
#
#   julia --project=RotorMap -t 8 RotorMap/tools/map.jl
#
#   INDEX       index .bin path (required)
#   READS       reads fasta path (required; records shorter than the index
#               window k are skipped by the reader, provenance optional)
#   MAPS        output table path (default READS + ".maps"; overwritten)
#   -topk=20    top-k matches kept (and emitted) per read
#   --w=65536   database column chunk width = ring slot width (fp8 optimum;
#               fp16 mode defaults to 32768)
#   --batch=8192  reads per encode/GEMM batch
#   --segs=8    column segments per row in the fused top-k kernel
#   -v          verbose: stage timings (index load, host database build, flow
#               split at the first batch's JIT warm-up), batches/s + reads/s,
#               the ring sweep's PCIe GB/s or the resident upload GB, and
#               GPU memory occupancy around the pipeline (the e2e budget)
#
# Numeric path: picked from the GPU's compute capability BEFORE any include
# (the one-variant rule) exactly like selftest.jl/e2e_auto.jl -- cc >= 8.9
# includes experiments/e2e_compact.jl wholesale (fp8 e4m3, cuBLASLt :lt
# engine -- the only engine this script runs), anything older
# experiments/e2e_compact16.jl (fp16 cuBLAS).  Residency is AUTO
# (`resident_fits`/`resident_fits16`: the engine uploads the database once
# when it fits free VRAM and runs the resident engine, else the
# chunk-streaming ring) -- the same decision e2e_auto makes, logged either
# way.
#
# OUTPUT FORMAT (one line per (read, rank), topk lines per read, whitespace
# separated; matches are sorted by score DESCENDING within each read):
#
#   <read header> <'+'|'-'> <score> <0-based position> <reference record header>
#
#   read header     the read fasta's verbatim header line ('>' included)
#   direction       the indexed column's strand: '+' = forward window,
#                   '-' = reverse-complement window
#   score           the inner product dot(read embedding e4m3-quantized (fp8
#                   mode), B[:, c]) -- the engines' descending D_val
#   position        the record-relative 1-based window start minus 1, in the
#                   ORIGINAL record coordinates (the index stores it that way
#                   for both strands)
#   record header   the index record header's FIRST TOKEN (before the first
#                   whitespace), sans leading '>'; multi-token record headers
#                   would break the whitespace-separated table
#
# The engine's per-batch flow (rope encode -> fp8/fp16 GEMM + fused seg
# top-k, event-choreographed) and the auto-residency ring are e2e_compact
# verbatim; this script only swaps the provenance scorer for a table writer.
# =============================================================================

using CUDA # stage 0 needs the device; the layered stack re-`using`s in util.jl

# ------------------------------------------------------------------------------
# stage 0: hardware detection -> numeric path (BEFORE the includes; e4m3 fp8
# tensor cores exist on sm_89, sm_90, sm_120+, everything older fp16 cuBLAS)
# ------------------------------------------------------------------------------
const _MAP_FP8 = let
    CUDA.functional() ||
        error("no functional CUDA device -- mapping is GPU-only")
    cap = CUDA.capability(device())
    cap.major > 8 || (cap.major == 8 && cap.minor >= 9)
end

include(joinpath(@__DIR__, "..",
                 _MAP_FP8 ? "experiments/e2e_compact.jl" :   # full fp8 stack
                            "experiments/e2e_compact16.jl")) # full fp16 stack

using Random
using Printf
using Base.Threads

@info "map.jl: numeric stack picked from the GPU (before the includes)" stack =
    _MAP_FP8 ? "fp8 (e2e_compact flow, :lt engine)" : "fp16 (e2e_compact16 flow)"

# ------------------------------------------------------------------------------
# Configuration (env); INDEX/READS are the CLI's primary inputs
# ------------------------------------------------------------------------------
const MAP_INDEX = get(ENV, "INDEX", "")
const MAP_READS = get(ENV, "READS", "")
const MAP_OUT = get(ENV, "MAPS", "") # default: READS + ".maps"

_no_gt(s::AbstractString) = startswith(s, ">") ? String(s[2:end]) : String(s)

"""
    run_map(; index, reads, out, topk, w, batch_size, segs)

The full flow with auto residency (rope_topk_stream_compact['16'] with
`resident = nothing` -- its auto policy, logged) and NO provenance parsing:
every batch's top-k rows are written to `out` in the documented table
format.  `verbose` (the -v flag) collects the e2e flows' stage budget:
index load, host database build, flow wall time split at the first batch
(pipeline JIT/first-call warm-up vs steady state), batches/s and reads/s,
the ring sweep's PCIe economics or the resident engine's one-time upload
GB, and GPU memory occupancy around the pipeline.  Returns the
processed-read count.
"""
function run_map(; index::String = MAP_INDEX, reads::String = MAP_READS,
                 out::String = isempty(MAP_OUT) ? string(reads, ".maps") : MAP_OUT,
                 topk::Int = 20, w::Int = _MAP_FP8 ? 2^16 : 2^15,
                 batch_size::Int = 2^13, segs::Int = 8, verbose::Bool = false)
    isempty(index) && error("set INDEX=<index .bin path>")
    isempty(reads) && error("set READS=<reads fasta path>")
    isfile(index) || error("index .bin not found: $index (generate with " *
                           "tools/build_index.jl save)")
    isfile(reads) || error("reads fasta not found: $reads")
    @show CUDA.name(device())
    @show nthreads()
    t_load = @elapsed d = load_index(index)
    @info "building the host database"
    t_B = @elapsed Bh = _MAP_FP8 ? index_database_host(d) : index_database_host_f16(d)
    verbose && @printf("  stage: index load %.2f s, host database build %.2f s\n",
                       t_load, t_B)
    db_gb = length(Bh) * (_MAP_FP8 ? 1 : 2) / 1e9 # fp8: 1 B/elem, fp16: 2 (GB, decimal)
    verbose && @info "host database ready" gb = round(db_gb; digits = 2) dims = size(Bh)
    heads_out = d.heads        # verbatim record headers ('>' included)
    starts_out = d.starts      # 1-based record-relative window starts (both strands)
    strand = d.strand          # 0 = fwd, 1 = rc
    # the record headers' FIRST TOKENS (before the first whitespace, sans
    # '>') -- the output table's last field, so a multi-token record header
    # (e.g. ">chrA synthetic") cannot break the whitespace-separated format:
    recname = [_no_gt(split(heads_out[j])[1]) for j in eachindex(heads_out)]
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    w = min(w, prevpow(2, size(Bh, 2))) # small test indexes admit no production width
    d = nothing # keep heads/starts/strand/Bh; the loader NamedTuple can go
    stream = _MAP_FP8 ? rope_topk_stream_compact : rope_topk_stream_compact16
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    nreads = nbat = 0
    t_jit = 0.0 # the first batch's wall time (JIT compile + first-call cost)
    tbatsum = 0.0
    verbose && (gpu0 = CUDA.total_memory() - CUDA.free_memory())
    # the residency decision the stream will apply under `resident = nothing`
    # (its exact auto policy, restated here for the verbose report wording)
    go_resident = verbose && (_MAP_FP8 ?
        resident_fits(Bh; rows_cap = batch_size, rdim = size(Bh, 1), w) :
        resident_fits16(Bh; rows_cap = batch_size, w))
    t = @elapsed open(out, "w") do io
        for bat in stream(re, reads, Bh; k = topk, w, batch_size, segs,
                          err_out = err, tasks_out = tks)
            tbat = @elapsed begin
                vals, locs, heads = bat.vals, bat.locs, bat.heads
                nb = size(vals, 1)
                for r in 1:nb, j in 1:topk
                    c = Int(locs[r, j])
                    @printf(io, "%s %c %s %d %s\n", _no_gt(heads[r]),
                            strand[c] == 0 ? '+' : '-', vals[r, j],
                            starts_out[c] - 1, recname[c])
                end
                nreads += nb
            end
            nbat += 1
            tbatsum += tbat
            nbat == 1 && (t_jit = tbat)
            @printf("  batch %d: +%d reads (%d total) in %.2f s\n",
                    nbat, nb, nreads, tbat)
        end
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    @printf("MAPPED %d reads -> %s (%d match lines) in %.1f s\n",
            nreads, out, nreads * topk, t)
    verbose && begin
            steady = t - tbatsum
        @printf("  flow: %.2f s over %d batches (%.2f batch/s, %.0f reads/s); the batches' write-out took %.2f s (first: %.2f s), the rest %.2f s = the pipeline's own work (JIT warm-up, encoding, GEMM + top-k) overlapping it\n",
                t, nbat, nbat / t, nreads / t, tbatsum, t_jit, steady)
        if go_resident
            @printf("  resident path: %.2f GB uploaded to VRAM once -- no per-batch sweep\n",
                    db_gb)
        else
            swept = nbat * db_gb # every read batch sweeps the whole database
            @printf("  ring sweep: %d batches x %.2f GB = %.1f GB over PCIe in %.2f s (%.1f GB/s effective)\n",
                    nbat, db_gb, swept, t, swept / t / 1e9)
        end
        gpu1 = CUDA.total_memory() - CUDA.free_memory()
        @printf("  GPU: %.2f GB used at the pipeline's start, %.2f GB after teardown\n",
                gpu0 / 1e9, gpu1 / 1e9)
    end
    Bh = nothing
    GC.gc(); CUDA.reclaim()
    return nreads
end

# ==============================================================================
# Mode dispatch -- env inputs + the flag spellings -topk/--batch/--segs/--w
# (a leading - or -- both work)
# ==============================================================================
function main()
    function getflag(name::String, default::String)
        for a in ARGS
            for p in ("--" * name * "=", name * "=")
                startswith(a, p) && return String(a[(length(p) + 1):end])
            end
        end
        return default
    end
    verbose = "-v" in ARGS || "--v" in ARGS || "--verbose" in ARGS ||
              "-verbose" in ARGS
    run_map(; topk = parse(Int, getflag("topk", "20")),
            w = parse(Int, getflag("w", string(_MAP_FP8 ? 2^16 : 2^15))),
            batch_size = parse(Int, getflag("batch", string(2^13))),
            segs = parse(Int, getflag("segs", "8")), verbose)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
