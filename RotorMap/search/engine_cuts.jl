# ==============================================================================
# e2ecuts.jl -- END-TO-END HUMAN-GENOME MAPPING with per-COLUMN CUTS: the
# e2ehuman flow (the indexreal fp8 index as the database, sample
# reads streamed through ropeflowreal -> flowtopkfp8) with the per-row top-k
# reduction REPLACED by a per-column threshold ("cut") gather: every inner
# product ABOVE its column's cut saves its value and location.
#
# PROBLEM
#   top-k answers "which k database windows match this read best"; a cut
#   answers "which windows match this read AT ALL" -- a per-column
#   significance bar instead of a per-read quota.  The cut values are COLUMN
#   DEPENDENT: for a database B (rdim x N) the flow takes a cut vector of
#   length N (2^22 values for a 2^11 x 2^22 B -- one per reference window;
#   column j's energy/repetitiveness set its own similarity scale, so a
#   single global threshold would over/under-call per column).  A read's
#   result is a VARIABLE-LENGTH hit list (value, global column id): 0 hits
#   to -- in pathological repetitive regions -- N hits.  Ground truth is
#   unchanged (a read is CORRECTLY MAPPED iff any of its hits' windows
#   intersects its provenance window); the rank histogram is meaningless
#   here (no ranks) and is replaced by hit-count statistics.
#
# DESIGN
#   1. Everything upstream of the reduction is e2ehuman verbatim (included
#     here: flowtopkfp8's fp8 GEMM engine + streams, the index .bin loader,
#     the provenance scorer's parts, the tiny fixture builders -- the
#     same-process one-variant rule still applies, do NOT include e2ehuman
#     or this script together with flowtopk.jl).  The engine keeps its
#     double-buffered chunk loop, quantize+upload and cuBLASLt GEMM; ONLY
#     the per-chunk consumer kernel changes.
#   2. The cut gather kernel (`cuts_gather_kernel!`): one thread block per C
#     chunk column, 256 threads striding the rows -- element (r, j) with
#     v = Float32(C[r, j]) (fp16 chunk storage widened EXACTLY like the
#     top-k's D_val) is a hit iff v > cuts[offset + j] (STRICT >; cuts
#     indexed by global 1-based column id into B).  Hits are appended to
#     three flat device lists (value, global column, batch row) with ONE
#     global Int64 atomic counter (Int64: rows_cap * N can exceed 2^31).
#     The counter runs past the lists' capacity with the extra writes
#     dropped, so overflow is detected EXACTLY, never truncated silently.
#   3. Batch contract: the engine computes the full padded rows_cap height
#     (flowtopkfp8's DESIGN 3); pad rows are exact zeros, so for the
#     intended cuts >= 0 they never produce hits (negative cuts DO sweep the
#     pad rows in as zeros -- the scorer filters rows > nb, and the raw
#     kernel self-test accounts for them exactly).  Results per batch are
#     `CutBatch(vals, locs, rows, heads, first, total)` -- fresh HOST
#     vectors in GPU emission order (NOT sorted; group by `rows`).
#   4. Cut vector (`cut_vector`): a constant (--cut=0.5) or a per-column
#     file (--cutsfile / E2CUTS_FILE env: raw little-endian Float32 of
#     exactly 4N bytes, or whitespace-separated text with exactly N
#     numbers); length must equal N; uploaded to the GPU once per stream.
#   5. Scoring: e2ehuman's provenance logic per HIT -- interned record ids
#     and window intersection; a read maps iff ANY of its hits intersects
#     its true window.  The report adds the hit statistics the top-k flow
#     has no analogue for: hits total and rate per GEMM comparison, per-read
#     hit-count buckets, max, and zero-hit reads.  The gather kernel also
#     accumulates the run's inner-product VALUE histogram (34 bins of 1/16
#     over [-1, 1], block-aggregated in shared memory, reported cumulative):
#     the calibration data for choosing cut levels, attached to the summary
#     AND to the capacity-overflow error (see RESULTS for the human index).
#
# SELF-TEST (mode `test`; production files not needed; shares e2ehuman's
#   cached 1M-base two-record fixture and err = 0 reads):
#   A. raw kernel extremes through the REAL engine (no pipeline): cut
#     -1e30 -> total == rows_cap * N EXACTLY (every element counted,
#     including the zeroed pad rows), cut +1e30 -> 0 on the SAME counter
#     (per-batch reset witnessed); global column ids in 1:N.
#   B. full pipeline vs an exact CPU reference (production-kernel read
#     encoding -> e4m3 quantize -> float64 GEMM against the DEQUANTIZED B,
#     flowtopkfp8's reference model): the per-batch hit SET must match the
#     reference set within the fp16-storage band (atol 5e-3 around the cut;
#     near-boundary elements may go either way), over multiple batches with
#     a ragged tail; no rows beyond nb (pad leak), no overflow -- at BOTH
#     cut levels (bar 0.90, the production operating point, and cut 0.5).
#   C. map_and_count_cuts: ALL 1000 err = 0 reads correctly mapped at the
#     0.5 cut (an err = 0 read shares >= 95% of its bases with an index
#     window, so inner products only reach down to ~0.9 -- an all-mapped
#     assert needs the lower cut), and the mapping count at the bar
#     REPORTED (a fixture data property, not pass/fail).
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16; index = the production
# indexreal save of GRCh38.p14: 1,547,081 windows x2 strands = 3,094,162
# columns x rdim 2048 e4m3, 67 unique records; reads = sample n=131072
# k=20000 err=0.05 s=42; w = 2^16, batch = 2^13 -> 16 batches, engine :lt,
# maxhits = 2^26):
#
#   THE SIMILARITY LANDSCAPE (the cumulative value histogram over the whole
#   run; 131072 x 3,094,162 = 4.06e11 comparisons): same-genome window pairs
#   are NOT near-orthogonal -- a background mode at 0.375-0.5 (2.21e11 +
#   1.07e11 pairs) dominates, so cuts below ~0.5 drown: cut 0.5 = 358.8M hits
#   in ONE batch (1.4% of all comparisons -- the exact overflow error IS the
#   measurement; count-only).  A NEAR-IDENTICAL spike at [0.9375, 1] (4.35e8
#   pairs/run ~ 3.3k windows per read: segmental duplicates, alt loci,
#   tandem arrays -- plus 2.2e7 quantization overshoots > 1) sits on top.
#
#   PRODUCTION CUT SWEEP (e2e incl. hit-list D2H + host scoring; ~20 s each):
#     cut 0.95: 441M hits (3364/read mean, max 73,138) -- 6,424/131,072 =
#       4.90% mapped; 124,338 reads with ZERO hits (their mutated true window
#       scores < 0.95) vs 6,033 repeat-lander reads with >1000 hits each:
#       the 0.95 bar maps ONLY the repeat crowd
#     cut 0.75: 461M hits (3517/read) -- 122,568/131,072 = 93.51% mapped;
#       8,314 zero-hit reads; 110,030 reads with EXACTLY 1 hit (their locus)
#     cut 0.60: 554M hits (4226/read) -- 131,072/131,072 = 100.00% mapped;
#       no zero-hit reads (every mutated true window clears 0.6)
#   vs e2ehuman's top-20 (94.98% in 4.29 s, exactly 20 hits/read): the bar's
#   recall is FIXED by the value distribution -- it needs calibration (0.6
#   here) to match the quota, and its hit BUDGET is value-driven (3.4k-4.2k
#   reads at these cuts); the 20 s (vs 4.29 s) is the hit lists' D2H + host
#   scoring, not the GEMM.
#
#   TINY SELF-TEST (1M fixture, 1000 err = 0 reads, N = 976): kernel
#   extremes exact (983,808 = rows_cap x N all-match, 0 no-match); hit sets
#   match the float64 CPU reference at BOTH levels; 1000/1000 mapped at cut
#   0.5; 826/1000 at the 0.90 bar and 537/1000 at 0.95 -- the fixture's
#   err = 0 reads share >= 95% of bases with their index window, so inner
#   products reach down to ~0.9: an all-mapped assert needs the low cut,
#   the bar level is witnessed by the CPU-reference hit sets.
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (kernel extremes + CPU-reference hit sets + all mapped)
#   run   production: human index + n131072_e0.05 reads -> cut mapping score
#   all   test + run (default)
#   flags: --cut=0.95 (run mode's constant cut level; --cut does not reach
#          the self-test, which pins its own levels: all-mapped at 0.5,
#          correctness + mapping report at the 0.90 bar), --cutsfile=
#          (per-column cut file, beats --cut; E2CUTS_FILE env),
#          --maxhits=67108864 (per-batch hit capacity, 3 x 4 B x maxhits of
#          device memory), --w=65536 (B-column chunk), --batch=8192,
#          --engine=lt (the top-k's segs/k knobs are GONE -- no top-k kernel
#          here; mma still needs w | N, and the human N = 3,094,162 is not a
#          power of two)
#
# USAGE
#   julia --project=. -t 16 test/e2ecuts.jl test
#   julia --project=. -t 16 test/e2ecuts.jl run --cut=0.95
#   julia --project=. -t 16 test/e2ecuts.jl run --cutsfile=cuts_3M.f32
#   julia --project=. -t 16 test/e2ecuts.jl            # test + run
# ------------------------------------------------------------------------------
# NEW TREE NOTES (reorganization): this file is now a LIBRARY layer
# (search/engine_cuts.jl).  The legacy include of e2ehuman.jl is replaced by
# the entry script's canonical include order; requires:
#   common/dna.jl, common/util.jl, common/testref.jl, common/harness.jl
#   (_ensure_e2h_test_ref / _e2h_* fixtures used by the run modes),
#   fasta/*, encode/*, gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl,
#   search/engine_fp8.jl (TopKEngine, upload_fp16_as_f8!, rope_topk_stream),
#   reads/provenance.jl (parse_read_head), index/{build,load}.jl.
# run_e2c_test / run_e2c_human moved to experiments/e2e_cuts.jl (the
# dispatch below was dropped here -- include this file from the experiment).
# REQUIRES engine_fp8.jl -> NOT co-includable with engine_fp16.jl /
# engine_complex.jl.
# ------------------------------------------------------------------------------

using Random
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
const E2C_MAXHITS = 2^26        # per-batch hit-list capacity (hits; 805 MB of
# device lists + a transient host copy per batch -- sized for the production
# index at the default cut 0.95, whose batch 1 carries 28.1M hits)
const E2C_CUTS_FILE_ENV = "E2CUTS_FILE" # per-column cut file (see cut_vector)

"""
    cut_vector(N; cut = 0.5, path = nothing) -> Vector{Float32}

The cut parameter: one Float32 value per column of B (N = the number of
database windows).  A constant vector `fill(cut, N)`, or -- when `path` is
given -- per-column values from a file: raw little-endian Float32 of exactly
`4N` bytes, or whitespace-separated text with exactly N numbers (e.g. one
calibrated cut per reference window from an offline pass).  The flow's
comparison is `innerproduct > cuts[column]` (STRICT).
"""
function cut_vector(N::Integer; cut::Real = 0.5,
                    path::Union{String,Nothing} = nothing)
    path === nothing && return fill(Float32(cut), N)
    isfile(path) || error("cut file not found: $path")
    raw = read(path)
    if length(raw) == 4 * N
        return collect(reinterpret(Float32, raw))
    end
    toks = split(String(raw); keepempty = false)
    length(toks) == N ||
        error("cut file $path: $(length(toks)) values ($(length(raw)) bytes) " *
              "for N = $N columns")
    out = Vector{Float32}(undef, N)
    for (i, s) in enumerate(toks)
        x = tryparse(Float32, s)
        x === nothing && error("cut file $path: not a number: \"$s\"")
        out[i] = x
    end
    return out
end

# ------------------------------------------------------------------------------
# The cut gather kernel: seg_rowtopk_merge_kernel!'s replacement -- a
# threshold filter + flat compaction instead of a per-row top-k reservoir.
# ------------------------------------------------------------------------------

"""
    cuts_gather_kernel!(hits_val, hits_loc, hits_row, cnt, C, cuts, offset)

Scan one C chunk ((rows_cap, wi); fp16 or fp32 -- the fp8 GEMM's
double-buffered output, possibly a ragged strided view) and append EVERY
element above its column's cut to the flat hit lists: element (r, j) with
v = Float32(C[r, j]) (fp16 storage widened exactly like the top-k's D_val)
is a hit iff v > cuts[offset + j] (offset = the chunk's 0-based global
column offset, so the stored locations are GLOBAL 1-based column ids into
B).  Appends go through one global Int64 atomic counter; the counter is
allowed to run past the lists' capacity -- overflowing writes are dropped
but COUNTED, so the host sees the exact total and can detect overflow.
One thread block per column, 256 threads striding the rows: consecutive
threads read consecutive row addresses (coalesced), cuts[offset + j] is one
broadcast load per block.  Row/column ids are Int32; the counter is Int64
(rows_cap * N can exceed 2^31).
"""
function cuts_gather_kernel!(hits_val::CuDeviceVector{Float32},
                             hits_loc::CuDeviceVector{Int32},
                             hits_row::CuDeviceVector{Int32},
                             cnt::CuDeviceVector{Int64},
                             C, cuts::CuDeviceVector{Float32}, offset::Int32,
                             hist::CuDeviceVector{Int64})
    j = blockIdx().x
    g = offset + j                 # global 1-based column id into B
    cut = cuts[g]
    M = Int32(size(C, 1))
    tid = threadIdx().x
    B_ = Int32(blockDim().x)
    cap = Int64(length(hits_val))
    # block-local similarity histogram (34 bins: v < -1, 32 bins of 1/16 over
    # [-1, 1], v > +1), flushed to the global run accumulator once per block
    shist = CuDynamicSharedArray(Int64, (34,))
    if tid <= Int32(34)
        @inbounds shist[tid] = Int64(0)
    end
    sync_threads()
    @inbounds for r in tid:B_:M
        v = Float32(C[r, j])
        v == v || continue         # NaN: a broken GEMM, not a similarity
        b = unsafe_trunc(Int32, (v + 1f0) * 16f0) + Int32(2)
        b = clamp(b, Int32(1), Int32(34))
        CUDA.atomic_add!(pointer(shist, b), Int64(1))
        v > cut || continue
        p = CUDA.atomic_add!(pointer(cnt, 1), Int64(1)) + Int64(1) # old + 1 = my slot
        p <= cap || continue       # capacity overflow: counted, not written
        hits_val[p] = v
        hits_loc[p] = g
        hits_row[p] = r
    end
    sync_threads()
    if tid <= Int32(34)
        CUDA.atomic_add!(pointer(hist, tid), shist[tid])
    end
    return nothing
end

function launch_cuts_gather!(hits_val::CuVector{Float32}, hits_loc::CuVector{Int32},
                             hits_row::CuVector{Int32}, cnt::CuVector{Int64},
                             C::CuMatrix, cuts_d::CuVector{Float32}, offset::Integer,
                             hist_d::CuVector{Int64}; threads::Int = 256)
    @cuda threads = threads shmem = 34 * sizeof(Int64) blocks = (size(C, 2), 1) cuts_gather_kernel!(
        hits_val, hits_loc, hits_row, cnt, C, cuts_d, Int32(offset), hist_d)
    return nothing
end

"""
    batch_gemm_cuts!(hits_val, hits_loc, hits_row, cnt, eng, cuts_d, hist_d;
                     overlap = true) -> total::Int

batch_gemm_topk!'s double-buffered chunk loop with the per-chunk top-k
REPLACED by the cut gather: the quantized batch must already be staged in
`eng.a8` (`upload_fp16_as_f8!` first, plus zeroing rows nb+1:rows_cap when
the batch shrinks -- cuts_flow does both); the FULL padded rows_cap height
is computed and the pad rows are exact zeros (they pass only cuts < 0).
`cnt` is reset here (batches are independent query sets); `hist_d` is NOT
(it accumulates the run's similarity histogram).  Returns the EXACT total
number of matches found -- which may exceed `length(hits_val)`: the caller
detects capacity overflow from it.  Returns with the device synchronized.
"""
function batch_gemm_cuts!(hits_val::CuVector{Float32}, hits_loc::CuVector{Int32},
                          hits_row::CuVector{Int32}, cnt::CuVector{Int64},
                          eng::TopKEngine, cuts_d::CuVector{Float32},
                          hist_d::CuVector{Int64};
                          overlap::Bool = true)
    rdim, N = size(eng.B)
    @assert size(eng.a8) == (eng.rows_cap, rdim) "engine staging must be (rows_cap, rdim)"
    fill!(cnt, Int64(0))
    CUDA.device_synchronize()

    sg = overlap ? eng.sg : CUDA.default_stream()
    st = overlap ? eng.st : sg
    for i in 1:cld(N, eng.w)
        b = mod1(i, 2)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        buf = b == 1 ? eng.buf1 : eng.buf2
        # base pointer of B column lo; the column stride stays the parent rdim
        Bp = byteptr(pointer(eng.B), (lo - 1) * rdim)

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(eng.evt[b])   # buffer b's previous gather done
            _gemm_chunk!(eng, buf, Bp, wi)   # launches on the current stream (= sg)
            CUDA.record(eng.evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b])            # C chunk is ready
            launch_cuts_gather!(hits_val, hits_loc, hits_row, cnt,
                                wi == eng.w ? buf : @view(buf[:, 1:wi]),
                                cuts_d, lo - 1, hist_d)
            CUDA.record(eng.evt[b])
        end
    end

    CUDA.device_synchronize()
    return Int(Array(cnt)[1]) # exact match count (allowscalar(false): bulk D2H)
end

# ------------------------------------------------------------------------------
# The streamed cut flow: rope_topk_stream with the reduction swapped
# ------------------------------------------------------------------------------

"""
One streamed batch of cut results.  The hit lists are in GPU EMISSION order
(NOT sorted; group by `rows`): `vals[p]` is an inner product that exceeded
its column's cut, `locs[p]` the matching GLOBAL 1-based column id into B,
`rows[p]` the 1-based batch row (= which read; <= length(heads)).  `vals`
are the fp16-stored GEMM values widened to Float32 -- the same numbers the
top-k flow reports in TopKBatch.vals.  `heads`/`first` are the rope batch's
passthrough; `total` is the exact match count (== length(vals): a batch
with more matches than the capacity errors out instead); `hist` is the
CUMULATIVE similarity histogram over the stream so far (34 Int64 bins:
v < -1, 32 bins of 1/16 over [-1, 1], v > +1).  Fresh HOST
vectors -- retain freely.
"""
struct CutBatch
    vals::Vector{Float32}      # hit inner products
    locs::Vector{Int32}        # matching global column ids into B
    rows::Vector{Int32}        # matching batch rows (the reads)
    heads::Vector{String}
    first::Int
    total::Int
    hist::Vector{Int64}        # cumulative similarity histogram (34 bins)
end

"""
    cuts_flow(source, eng, cuts; maxhits, out_cap, err_out, tasks_out)
        -> Channel{CutBatch}

Consume `source` (fp16 rope batches, topk_flow's contract) and, per batch,
quantize the embeddings to e4m3 on the GPU and run
`batch_gemm_cuts!` against the engine's resident fp8 database with `cuts`
(one Float32 cut per column of B, uploaded once), emitting
`CutBatch(vals, locs, rows, heads, first, total)` with fresh host vectors.
A batch with more matches than `maxhits` errors the stage (exact count in
the message -- raise maxhits or the cuts).  Error/teardown contract =
topk_flow's (err_out, tasks_out, early close unwinds the rope pipeline).
"""
function cuts_flow(source, eng::TopKEngine, cuts::Vector{Float32};
                   maxhits::Int = E2C_MAXHITS, out_cap::Int = 2,
                   err_out::Ref{Any} = Ref{Any}(nothing),
                   tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    N = size(eng.B, 2)
    length(cuts) == N ||
        throw(ArgumentError("cuts: $(length(cuts)) values for N = $N columns"))
    cuts_d = CuArray(cuts)             # uploaded once for the whole stream
    hits_val = CuVector{Float32}(undef, maxhits)
    hits_loc = CuVector{Int32}(undef, maxhits)
    hits_row = CuVector{Int32}(undef, maxhits)
    cnt = CuVector{Int64}(undef, 1)
    hist_d = CuVector{Int64}(undef, 34) # run-cumulative similarity histogram
    out = Channel{CutBatch}(out_cap)
    nb_done = 0
    t_cut = Threads.@spawn begin
        try
            for bat in source
                emb = bat.embeds
                eltype(emb) == Float16 ||
                    throw(ArgumentError("batch eltype $(eltype(emb)) != Float16 (build the " *
                                        "rope stream with fp16 = true; this stage quantizes " *
                                        "the embeddings to e4m3 on the GPU)"))
                nb = size(emb, 1)
                nb == 0 && continue
                nb <= eng.rows_cap ||
                    throw(ArgumentError("batch height $nb > engine rows_cap $(eng.rows_cap)"))
                # upload/quantize on the GEMM stream (topk_flow's pattern: keep
                # multi-ms kernel queues OFF the default stream -- CUDACore's
                # legacy-sync crash, see flowtopkfp8's header)
                CUDA.stream!(eng.sg) do
                    upload_fp16_as_f8!(eng, emb)
                    if nb < eng.rows_cap
                        # unlike topk_flow (which slices the pad rows off), the
                        # gather LOOKS at every computed row: on a reused engine
                        # rows nb+1:rows_cap would otherwise score the PREVIOUS
                        # batches' stale embeddings -- zero them (both the e4m3
                        # staging and its fp32 scratch, which the next batch's
                        # converter re-reads 1-D over the full height) so the
                        # pad rows are exact zeros again
                        @views eng.a8[nb+1:end, :] .= F8(0)
                        @views eng.a32[nb+1:end, :] .= 0f0
                    end
                end
                total = batch_gemm_cuts!(hits_val, hits_loc, hits_row, cnt,
                                         eng, cuts_d, hist_d)
                nb_done += 1
                hist = Array(hist_d)     # cumulative snapshot (272 B)
                total > maxhits &&
                    throw(ErrorException("cut overflow: $total matches in one batch " *
                                         "exceed maxhits = $maxhits ($(total - maxhits) " *
                                         "more slots needed; raise --maxhits or the cuts). " *
                                         "Similarity histogram over $nb_done batches (bins of " *
                                         "1/16 over [-1,1]; first <-1, last >+1): $hist"))
                vals = Vector{Float32}(undef, total)
                locs = Vector{Int32}(undef, total)
                rows = Vector{Int32}(undef, total)
                copyto!(vals, 1, hits_val, 1, total)
                copyto!(locs, 1, hits_loc, 1, total)
                copyto!(rows, 1, hits_row, 1, total)
                put!(out, CutBatch(vals, locs, rows, bat.heads, bat.first, total, hist))
            end
        catch err
            # close the rope pipeline on BOTH paths (the same fix topk_flow
            # got): on the benign teardown it propagates the unwind upstream;
            # on a real error the guarded variant would leave the rope stage
            # pumping into this dead task and the caller's task join would hang
            _topkfp8_flow_error(err, err_out, out; who = "cutsgather")
            source isa AbstractChannel && close(source)
        finally
            close(out) # also the early-close path (close on closed is a no-op)
        end
    end
    append!(tasks_out[], [t_cut])
    return out
end

"""
    rope_cuts_stream(re, file, B, cuts; maxhits, w, batch_size, ...) ->
        Channel{CutBatch}

rope_topk_stream with the top-k replaced by the cut gather: stream-rope-
encode the fragments (`rope_encode_real_stream`, fp16) and, per batch,
quantize to e4m3 and run the double-buffered fp8 GEMM over B's column
chunks, gathering every inner product above the column's cut into the flat
hit lists (CutBatch).  `cuts` must hold one Float32 value per column of B;
the comparison is STRICT `>` on the fp16-stored GEMM value widened to
Float32 (the same number the top-k flow would report).  `maxhits` is the
per-batch hit capacity (3 x 4 B x maxhits of device memory + a transient
host copy per batch).  All other parameters and the error/teardown contract
are rope_topk_stream's.
"""
function rope_cuts_stream(re::RopeEncoder, file::String, B::CuMatrix{F8},
                          cuts::Vector{Float32};
                          maxhits::Int = E2C_MAXHITS, w::Int = 2^16,
                          batch_size::Int = 2^13, rows_cap::Int = batch_size,
                          normalize::Int = 0, parts::Int = Threads.nthreads(),
                          in_cap::Int = 2, out_cap::Int = 2, gather_out_cap::Int = 2,
                          engine::Symbol = :lt, cout::Symbol = :f16,
                          progress::Bool = false,
                          err_out::Ref{Any} = Ref{Any}(nothing),
                          tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(B, 1) == rdim ||
        throw(ArgumentError("B is $(size(B, 1)) x $(size(B, 2)); the encoder's real " *
                            "embedding dim is 2*m*4^c = $rdim"))
    length(cuts) == size(B, 2) ||
        throw(ArgumentError("cuts: $(length(cuts)) values for N = $(size(B, 2)) columns"))
    eng = TopKEngine(B; k = 1, w, rows_cap, segs = 8, engine, cout) # k is unused
    # here (the top-k kernel is replaced by the gather); k = 1 keeps the
    # engine's assert happy and its D-buffers unallocated
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return cuts_flow(rope_ch, eng, cuts; maxhits, out_cap = gather_out_cap,
                     err_out, tasks_out)
end

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, gather hits, score them against the
# provenance (e2ehuman's map_and_count, hit-list edition)
# ------------------------------------------------------------------------------

"""
    map_and_count_cuts(re, reads_file, B, db_heads, db_starts, dbk, cuts;
                       maxhits, w, batch_size, engine, normalize, progress)
        -> (total, correct, hits, hitmax, zerohits, buckets, seconds)

Run the cut flow (`rope_cuts_stream`) over `reads_file` and score every read
against its provenance header: CORRECTLY MAPPED iff ANY of its hits' windows
(db_heads[c], db_starts[c], length dbk) intersects the true window
[start, start+len-1] in record src (e2ehuman's intersection test, applied
per hit instead of per ranked top-k entry).  Alongside the processed-read
count and the correctly-mapped count this returns the hit statistics the
top-k flow has no analogue for: `hits` (total hits above the cuts),
`hitmax` (worst per-read hit count), `zerohits` (reads with no hit at all)
and `buckets` (per-read hit counts in [0, 1, 2-10, 11-100, 101-1000,
>1000]).
"""
function map_and_count_cuts(re::RopeEncoder, reads_file::String, B::CuMatrix{F8},
                            db_heads::Vector{String}, db_starts::Vector{Int},
                            dbk::Int, cuts::Vector{Float32};
                            maxhits::Int = E2C_MAXHITS, w::Int = 2^16,
                            batch_size::Int = 2^13, engine::Symbol = :lt,
                            normalize::Int = 0, progress::Bool = false)
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(B, 2)
    length(cuts) == N ||
        throw(ArgumentError("cuts: $(length(cuts)) values for N = $N columns"))
    # per-column record id: the .bin's heads repeat a handful of unique
    # record headers over millions of columns -- intern them once
    rec_id = Dict{String,Int32}()
    rec_id_acc = Dict{String,Int32}() # provenance src names the record by accession
    rec_of_col = Vector{Int32}(undef, N)
    for j in 1:N
        rec_of_col[j] = get!(rec_id, db_heads[j]) do
            Int32(length(rec_id) + 1)
        end
        acc = String(split(db_heads[j])[1][2:end]) # the header's first token, sans '>'
        haskey(rec_id_acc, acc) || (rec_id_acc[acc] = rec_of_col[j])
    end
    total = correct = nb_bat = hitsum = 0
    hitmax = zerohits = 0
    buckets = zeros(Int, 6) # per-read hits: 0 | 1 | 2-10 | 11-100 | 101-1000 | >1000
    hist = zeros(Int64, 34) # cumulative similarity histogram (CutBatch snapshots)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in rope_cuts_stream(re, reads_file, B, cuts; maxhits, w,
                                             batch_size, rows_cap = batch_size,
                                             normalize, engine, progress,
                                             err_out = err, tasks_out = tks)
        nb_bat += 1
        nb = length(bat.heads)
        @assert bat.total == length(bat.vals) "overflow leaked into a batch"
        # provenance of every read in the batch (e2ehuman's parse)
        rstart = Vector{Int}(undef, nb)
        rid = Vector{Int32}(undef, nb)
        for r in 1:nb
            h = parse_read_head(bat.heads[r])
            h === nothing &&
                error("unreadable provenance header: $(bat.heads[r])")
            h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rstart[r] = h.start
            rid[r] = get(rec_id_acc, h.src, Int32(0)) # provenance names by accession
        end
        mapped = falses(nb)
        hpc = zeros(Int, nb)
        for p in eachindex(bat.vals)
            r = Int(bat.rows[p])
            1 <= r <= nb || continue # pad rows (only cuts < 0 sweep them in)
            hpc[r] += 1
            if !mapped[r]
                c = Int(bat.locs[p])
                # read window [rstart, rstart+dbk-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid[r] &&
                   rstart[r] <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= rstart[r] + dbk - 1
                    mapped[r] = true
                end
            end
        end
        nh = sum(hpc)
        hitsum += nh
        hitmax = max(hitmax, nh > 0 ? maximum(hpc) : 0)
        zerohits += count(iszero, hpc)
        for h in hpc
            buckets[h == 0 ? 1 : h == 1 ? 2 : h <= 10 ? 3 : h <= 100 ? 4 :
                        h <= 1000 ? 5 : 6] += 1
        end
        total += nb
        correct += count(mapped)
        hist .= bat.hist # cumulative: the last snapshot is the whole stream
        @printf("  batch %2d: +%d reads, %d hits (%.1f/read), %d/%d correctly mapped so far\n",
                nb_bat, nb, nh, nh / nb, correct, total)
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    return (total = total, correct = correct, hits = hitsum, hitmax = hitmax,
            zerohits = zerohits, buckets = buckets, hist = hist, seconds = t)
end

# ------------------------------------------------------------------------------
# Self-test helpers
# ------------------------------------------------------------------------------

# Host reference encoding of a batch of reads (packed words -> the PRODUCTION
# kernel -> fp16 rows), for the exact CPU-reference hit-set check.
function _e2c_encode_batch(re::RopeEncoder, words::Vector{Vector{UInt32}};
                           normalize::Int = 0)
    nb = length(words)
    W = cld(re.k, 16)
    rdim = 2 * re.m * 4^re.c
    fw = Vector{UInt32}(undef, W * nb)
    for i in 1:nb
        @assert length(words[i]) == W "packed fragment width mismatch"
        copyto!(fw, (i - 1) * W + 1, words[i], 1, W)
    end
    dest = CUDA.zeros(Float16, nb, rdim)
    dn = CUDA.zeros(Float32, re.m, nb)
    encode_frag_real_batch!(dest, dn, re, cu(fw); normalize)
    emb = Array(dest)
    dest = dn = fw = nothing
    GC.gc(); CUDA.reclaim()
    return emb
end

