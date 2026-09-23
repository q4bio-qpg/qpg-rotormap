# ==============================================================================
# e2ecompact.jl -- END-TO-END MAPPING with a RAM-RESIDENT index: the e2ehuman
# flow for databases TOO BIG FOR VRAM.  The whole fp8 database B stays in HOST
# memory (plain Julia Matrix{F8}, no pinning -- see below); every read batch
# sweeps it through the PCIe link once, `w` columns at a time, into a
# TRIPLE-BUFFERED device ring whose slots feed the same double-buffered fp8
# GEMM + seg-top-k pipeline as the resident engine (flowtopkfp8's
# batch_gemm_topk! with one extra stage bolted in front).
#
# PIPELINE (per chunk i of a batch sweep)
#   h2d stream : wait ring slot b free (gemm i-depth read it) -> cudaMemcpyAsync
#                the contiguous host column range into slot b -> record ev_h2d[b]
#   sg stream  : wait ev_h2d[b] (+ C buffer b2's previous top-k) -> fp8 GEMM
#                D = A * B_chunk -> record evg[b2] AND ev_dev[b] (slot consumed)
#   st stream  : wait evg[b2] -> seg_rowtopk_merge into the running top-k ->
#                record evt[b2]
# All waits are GPU-side (events); the host loop only enqueues, so the three
# stages overlap exactly like the resident flow's gemm || topk, plus the H2D
# under both.  Ring depth 3 (not 2) keeps the pipe full when the H2D and the
# compute are near-balanced and jitter would otherwise bubble a depth-2 ring.
# B's leading dimension is rdim both for the resident parent and for each
# (rdim, w) ring slot, so the engine dispatch (_gemm_chunk! -> fp8_gemm_lt! /
# fp8_gemm_mma!) is untouched: same chunk base pointer, same ldb, ragged final
# chunk reads wi < w columns from the slot base (production engine :lt; :mma
# keeps the resident engine's w | N restriction).
#
# THE INNER TopKEngine is constructed with ring slot 1 as its "B": a real
# (rdim, w) CuMatrix that satisfies every constructor assert and only ever
# contributes size(eng.B, 1) = rdim to the chunk dispatch -- the actual B
# bytes per chunk come from the ring pointer passed to _gemm_chunk!.
#
# WHY NO PINNED MEMORY (measured on kau, RTX 5090 + Gen5 x16, CUDA 6.3.jl):
# RLIMIT_MEMLOCK is 8 MiB soft AND hard here (container inherited from the
# host, not raisable without root), so cudaHostRegister/cudaHostAlloc beyond
# a few MiB fails.  Pageable H2D of 128 MiB chunks (w = 2^16, rdim = 2048)
# measures 6.7 ms/chunk ~ 20 GB/s (w = 2^17: 16 ms ~ 16 GB/s; two overlapped
# streams do NOT beat one -- the driver serializes staging).  Against the
# compute per chunk (fp8 GEMM 3.0 ms + seg top-k ~1.9 ms at M = 2^13) that
# means: at the default batch the sweep is mildly H2D-bound (6.7 vs ~5 ms);
# at --batch=2^15 the compute (~20 ms/chunk) dwarfs the H2D and the pipeline
# is compute-bound -- WITH OR WITHOUT pinning, i.e. the 8 MiB memlock cap
# costs nothing that matters.  Results are bitwise what the resident engine
# would produce (same A, same B bytes, same chunking); only the wall time
# carries the extra H2D.  Total H2D per read batch = the whole database once
# (33 GiB for the maize kstep-250 index), so larger --batch amortizes it
# linearly -- the compact flow's main tuning lever, VRAM is not the limit
# (C buffers 2 x rows_cap x w x 2 B: 8 GiB at batch 2^15 of the 32 available).
#
# AUTO RESIDENCY (design): the ring exists for databases TOO BIG FOR VRAM --
# when the host database plus the engine's working set DOES fit in the device
# memory free right now, `rope_topk_stream_compact` (resident = nothing, the
# default) uploads B ONCE and delegates to the resident engine (`topk_flow`,
# engine_fp8.jl) instead of sweeping: bitwise the same results (same A, same
# B bytes, same chunking) at exactly the resident flow's speed -- a fitting
# index must not pay the per-batch PCIe sweep.  `resident_fits` is the
# policy; e2e_compact.jl's --resident=on|off forces either side.
#
# SCORING is e2ehuman's map_and_count verbatim (same provenance parsing, same
# best-of-top-k intersection metric) with the batch producer swapped: the
# `stream` keyword it grew makes rope_topk_stream replaceable.
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (e2ehuman's 1M reference fixture -> index -> err = 0
#         reads -> the COMPACT chunked path -> all mapped)
#   run   production: the maize B73 index (default) + n131072_k4000_e0.3
#         reads (default) -> mapping score
#   all   test + run (default)
#   flags: --k=20 (top-k size), --w=65536 (B-column chunk width = ring slot
#          width), --batch=8192, --segs=8, --engine=lt (:mma needs w | N)
#   env:   E2COMPACT_INDEX (default: the maize kstep-250 s5m1c5 fp8 index,
#          33 GiB -- does not fit 32 GiB VRAM, the reason this script exists)
#          E2COMPACT_READS (default: the sample n131072_k4000_e0.3_s42
#          batch; point it at e.g. maksym's maize_1X_15.fasta for the eval)
#
# USAGE
#   julia --project=. -t 16 test/e2ecompact.jl test
#   julia --project=. -t 16 test/e2ecompact.jl run --engine=lt
#   julia --project=. -t 16 test/e2ecompact.jl run --batch=32768
#
# RESULTS (kau, RTX 5090, julia -t 16; maize B73 primary12 index k = 4,000,
# kstep = 250, encoder (5,1,5), 17,060,078 cols x rdim 2048 = 32.54 GiB in
# host RAM, load 17.8 s; production engine :lt, w = 2^16):
#
#   sample n131072_k4000_e0.3_s42 reads (131,072 reads, err = 0.3):
#     127,827/131,072 = 97.52% correctly mapped
#     rank hist: 116647, 4118, 1695, 1048, ... (rank-1 = 89.4% of all reads)
#     best top-20 intersection: avg 3,937 bases = 98.43% of the 4,000-base
#       fragment -- exactly the k - kstep/4 = 3,937.5 grid-geometry prediction
#     --batch=8192: 52.7 s (16 sweeps x 32.54 GiB, 10.6 GB/s effective; the
#       pageable staging serializes with the DMA -- see the header note above)
#     --batch=32768: 33.6 s (4 sweeps, 130.2 GiB total; compute-bound -- the
#       same 97.52% / 98.43%, bitwise-stable across batch sizes)
#
#   RESIDENT vs STREAMED head-to-head (PRE-auto-residency design, kept for
#   the record; same kstep-400 index -- 20.34 GiB,
#   FITS in VRAM, so both flows run it; same reads, :lt, w = 2^16, ktop 20):
#     e2ehuman resident, batch 8192      20.6 s   (B resident ~23 GiB VRAM)
#     e2ecompact streamed, batch 8192    33.6 s   (1.63x: H2D-bound, 325 GiB
#       swept at 10.4 GB/s effective -- the pageable staging serializes)
#     e2ecompact streamed, batch 32768   22.5 s   (1.09x: compute-bound,
#       81.3 GiB swept; ~8.5 GiB VRAM vs ~23 resident)
#     All three: bitwise 125,378/131,072 = 95.66%, intersection 97.52%.
#     Reading: when the index fits, resident wins by ~9% at the compact
#     flow's tuned batch (plus the resident flow's one-time ~2 s B upload
#     vs zero for compact); the compact flow exists for the ones that
#     don't fit -- there resident cannot run at all.
#
#   AUTO RESIDENCY head-to-head (the CURRENT design -- rope_topk_stream_
#   compact decides by itself; kau, RTX 5090, one day after the block above,
#   same 20.34 GiB kstep-400 index + same n131072_k4000_e0.3 reads, :lt,
#   w = 2^16, batch 8192, julia -t 16):
#     e2e_fp8 resident (the baseline)                 19.1 s
#     e2ecompact --resident=auto (-> RESIDENT)        20.3 s   PARITY -- the
#       1.2 s delta is exactly the one-time 20.34 GiB upload, which the
#       compact flow pays INSIDE res.seconds while the resident flow pays
#       it in index_database BEFORE its timer (measured 1.3 s); steady
#       state is identical (no per-batch sweep to amortize anymore)
#     e2ecompact --resident=off (the ring, old path)  32.7 s   (1.6x)
#     All three: bitwise 125,378/131,072 = 95.66%, intersection 97.52%
#     (identical rank histograms).  The too-big default (32.54 GiB kstep-250
#     index) still auto-streams: 49.9 s, 520.6 GiB swept at 11.2 GB/s,
#     bitwise the recorded 127,827/131,072 = 97.52% / 98.43%.  The tiny
#     `test` fixture now runs BOTH paths and asserts identical rank hists.
#
#   SMALL INDEX head-to-head (s4m1c4, k = 20,000, kstep = 2000, rdim 512,
#   2,132,328 cols = 1.02 GiB; eval reads maize_1X_15.fasta, 86,834 x ~23 kb
#   -> trimmed to k; readlen < 2k so the revcomp provenance scores exactly):
#     resident 7.4 s | streamed 7.5 s (batch 8192, 11 sweeps) | 7.4 s (32768,
#     3 sweeps) -- bitwise 86,722/86,834 = 99.87% mapped (112 unmapped),
#     best top-20 intersection 18,243 bases = 91.22% (below the k - kstep/4
#     plateau: the external generator's harsher error model + revcomp
#     trimmed-fragment geometry, same signature as the human eval batch).
#     On a ~1 GiB database the sweep is 0.3-1.1 GiB total and the flows are
#     indistinguishable -- the compact overhead only appears (and only
#     matters) when B approaches VRAM size.
#
#   eval reads maize_1X_15.fasta (86,834 reads, ~23 kb, ~50% revcomp):
#     43,545/86,834 = 50.15% -- BUT this is a PROVENANCE-GEOMETRY ARTIFACT,
#     not a mapping failure: the reader trims every read to its first k =
#     4,000 bases, and for a revcomp read the true source window sits at the
#     END of the read, so the scored fragment lies ~readlen - k ~ 19 kb away
#     from the header's start whenever readlen > 2k (the human eval held
#     readlen < 2k: 29,718 < 40,000; maize violates it: 23,316 >= 8,000, and
#     the colon header carries no read length to recover the true window).
#     Evidence: unmapped 43,289 ~ the revcomp count 43,426; mapped 43,545 ~
#     the forward count 43,408 (+137 short revcomp reads); mapped first-hits
#     are 97.4% rank-1.  Effective placement accuracy ~ 99.8%+ of the
#     scoreable (forward) half.  To score such batches honestly the header
#     format needs a length (or the trim must take the provenance end).
# ==============================================================================

# (legacy include of e2ehuman.jl replaced by the entry script's canonical
# include order: search/engine_fp8.jl provides TopKEngine, upload_fp16_as_f8!,
# rope_topk_stream; reads/provenance.jl provides parse_read_head; index/load.jl
# provides load_index/index_database; encode/stream.jl the rope stream)

# ------------------------------------------------------------------------------
# Configuration (env overrides): a database too big for VRAM by default
# ------------------------------------------------------------------------------
# NEW TREE NOTES (reorganization): this file is now a LIBRARY layer
# (search/engine_compact.jl).  The legacy include of e2ehuman.jl is replaced
# by the entry script's canonical include order; requires:
#   common/dna.jl, common/util.jl, common/testref.jl, common/harness.jl,
#   fasta/*, encode/*, gemm/{fp8_convert,fp8_ptx,fp8_lt,topk_kernels}.jl,
#   search/engine_fp8.jl, reads/provenance.jl, index/{build,load}.jl.
# The experiment ENV consts (E2C_INDEX / E2C_READS) and the run entry points
# (run_e2e_compact_test / run_e2e_compact + dispatch) moved to
# experiments/e2e_compact.jl.  REQUIRES engine_fp8.jl -> NOT co-includable
# with engine_fp16.jl / engine_complex.jl.
# ------------------------------------------------------------------------------

using CUDA
using LinearAlgebra

function index_database_host(d)
    if _is_e2h_fp8_format(get(d, :format, nothing)) # new + legacy indexflowreal tags
        return d.embeds8
    end
    return F8.(Float32.(permutedims(d.embeds)))
end

# ------------------------------------------------------------------------------
# Auto residency: when the whole host database fits in VRAM (the database
# plus the engine's working set, against the device memory free RIGHT NOW),
# the compact flow uploads B once and delegates to the resident engine --
# bitwise identical results, and the speed parity with the e2e_fp8 flow that
# is the compact flow's contract for fitting indices.
# ------------------------------------------------------------------------------

# Resident-path device working set beyond B itself: the engine's e4m3 batch
# staging (rows_cap x rdim, 1 B/elem) + its fp32 scratch (4 B/elem), the two
# C chunk buffers (rows_cap x w, 2/4 B/elem), per-batch temps (the fp16
# upload staging, D_val/D_loc, the D2H gather temp -- a few tens of MiB) and
# context/pool/allocator slack.
_resident_extra(rows_cap::Int, rdim::Int, w::Int, cout::Symbol) =
    rows_cap * rdim * 5 +                          # a8 + a32
    2 * rows_cap * w * (cout === :f16 ? 2 : 4) +   # buf1 + buf2
    96 * 2^20                                      # temps + context + pool slack

_resident_need(host::Matrix{F8}; rows_cap::Int, rdim::Int, w::Int,
               cout::Symbol = :f16) =
    sizeof(host) + _resident_extra(rows_cap, rdim, w, cout)

"""
    resident_fits(host; rows_cap, rdim, w, cout = :f16) -> Bool

The auto-residency policy of `rope_topk_stream_compact` (`resident = nothing`):
true when `host` plus the resident engine's working set fits in the device
memory free right now -- i.e. when the compact flow will upload the database
ONCE and run the resident engine (e2e_fp8's speed; no per-batch PCIe sweep)
instead of the chunk ring.

THE RULE (conservative by construction -- an underestimate would OOM mid-run,
while falling back to the ring is always safe):

    need = sizeof(host)                            # B verbatim, as uploaded
         + rows_cap * rdim * 5                     # a8 (e4m3, 1 B) + a32 (fp32, 4 B)
         + 2 * rows_cap * w * (cout == :f16 ? 2 : 4)  # buf1 + buf2 (C chunks)
         + 96 MiB                                  # small stuff + slack
    need <= CUDA.free_memory()   # after CUDA.reclaim()

  * the 96 MiB covers the per-batch temps (the ~32 MiB fp16 upload staging,
    D_val/D_loc, the D2H gather temp), the cuBLASLt workspace, and
    context/pool/allocator slack;
  * `CUDA.reclaim()` first returns the pool's cached blocks to the driver, so
    a long-lived process does not misjudge its own freed memory;
  * `CUDA.free_memory()` is the driver's free-VRAM reading AT THE CALL --
    anything else holding VRAM (another process, a second engine) is
    respected automatically;
  * `rows_cap` is compared pre-rounding (the engine rounds it up to its
    pad multiple -- MiB-scale, never flips a verdict).

Worked example (kau, RTX 5090 = 32 GiB, batch 8192, w = 2^16, :f16): the
20.34 GiB kstep-400 maize index needs 20.34 + ~0.08 + ~2.0 + ~0.09 ~ 22.5 GiB
<= 30.9 GiB free -> RESIDENT; the 32.54 GiB kstep-250 default needs ~34.7 GiB
> 30.9 -> the ring.
"""
function resident_fits(host::Matrix{F8}; rows_cap::Int, rdim::Int = size(host, 1),
                       w::Int, cout::Symbol = :f16)
    CUDA.reclaim() # judge free VRAM, not the pool's cached blocks
    return _resident_need(host; rows_cap, rdim, w, cout) <= CUDA.free_memory()
end

# ------------------------------------------------------------------------------
# The compact engine: host database + device chunk ring + streamed sweep
# ------------------------------------------------------------------------------

"""
Reusable per-batch fp8 GEMM+top-k state for a HOST database: the resident
`TopKEngine` (its B field is ring slot 1 -- shape only, see the header) plus
the host matrix, a depth-slot device ring of (rdim, w) e4m3 chunk buffers, a
dedicated non-blocking H2D stream and per-slot event pairs (h2d into the slot
done / gemm done reading the slot).  `depth >= 2`; 3 keeps the pipe full when
H2D and compute are near-balanced.
"""
struct CompactEngine
    eng::TopKEngine               # a8/a32 staging, C bufs, sg/st, evg/evt (cout :f16)
    host::Matrix{F8}               # (rdim, N) the whole database, host RAM
    depth::Int                     # device ring depth (slots)
    dev::Vector{CuMatrix{F8}}      # depth x (rdim, w) chunk ring
    h2d::CuStream                  # chunk H2D stream
    ev_h2d::Vector{CuEvent}        # ev_h2d[b]: the copy into slot b is done
    ev_dev::Vector{CuEvent}        # ev_dev[b]: the GEMM reading slot b is done
end

function CompactEngine(host::Matrix{F8}; k::Int = 20, w::Int = 2^16,
                       rows_cap::Int = 2^13, segs::Int = 8,
                       engine::Symbol = :lt, cout::Symbol = :f16, depth::Int = 3)
    rdim, N = size(host)
    1 <= k <= w <= N ||
        throw(ArgumentError("need 1 <= k <= w <= N (got k = $k, w = $w, N = $N)"))
    depth >= 2 || throw(ArgumentError("ring depth must be >= 2 (got $depth)"))
    engine in (:mma, :lt) || throw(ArgumentError("engine must be :mma or :lt"))
    if engine === :mma # resident restriction carried over: no ragged mma chunks
        N % w == 0 ||
            throw(ArgumentError("engine :mma needs w to divide N (no ragged final chunk)"))
    end
    dev = [CuMatrix{F8}(undef, rdim, w) for _ in 1:depth]
    # the inner engine's B is ring slot 1: a real (rdim, w) e4m3 matrix that
    # satisfies every constructor assert; only size(eng.B, 1) is ever read
    eng = TopKEngine(dev[1]; k, w, rows_cap, segs, engine, cout)
    h2d = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    CompactEngine(eng, host, depth, dev, h2d,
                  [CuEvent() for _ in 1:depth], [CuEvent() for _ in 1:depth])
end

"""
    batch_gemm_topk_compact!(D_val, D_loc, ceng) -> (D_val, D_loc)

`batch_gemm_topk!` with the database streamed from host RAM: same contract
(running top-k reset first, the quantized batch already staged in
ceng.eng.a8, full rows_cap height computed, returns device-synchronized) --
plus the per-chunk H2D stage on ceng.h2d and the device ring's event
bookkeeping (slot b is refilled only after the GEMM `depth` chunks back has
consumed it; every wait is GPU-side, the host loop only enqueues).
"""
function batch_gemm_topk_compact!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                                  ceng::CompactEngine)
    eng = ceng.eng
    M, KA = size(eng.a8)
    rdim, N = size(ceng.host)
    @assert KA == rdim "A's inner dim must match B's rows"
    @assert M == eng.rows_cap "engine staging must be (rows_cap, rdim)"
    @assert size(D_val) == (M, eng.k) && size(D_loc) == (M, eng.k)

    fill!(D_val, typemin(Float32)) # reset the running top-k; the kernel merges into it
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    sg, st, h2d = eng.sg, eng.st, ceng.h2d
    for i in 1:cld(N, eng.w)
        b = mod1(i, ceng.depth)          # device ring slot
        b2 = mod1(i, 2)                  # C buffer (the resident flow's pairing)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        cbuf = b2 == 1 ? eng.buf1 : eng.buf2
        # stage the host chunk into ring slot b: contiguous (lo:hi) column
        # range -- column-major makes the slice one flat memcpy (measured
        # 20 GB/s pageable; see the header for the pinning non-option)
        CUDA.stream!(h2d) do
            i > ceng.depth && CUDA.wait(ceng.ev_dev[b]) # slot b free (gemm i-depth)
            copyto!(ceng.dev[b], 1, ceng.host, (lo - 1) * rdim + 1, rdim * wi)
            CUDA.record(ceng.ev_h2d[b])
        end
        CUDA.stream!(sg) do
            CUDA.wait(ceng.ev_h2d[b])        # chunk resident
            i > 2 && CUDA.wait(eng.evt[b2])  # buffer b2's previous top-k finished
            _gemm_chunk!(eng, cbuf, byteptr(pointer(ceng.dev[b])), wi)
            CUDA.record(eng.evg[b2])
            CUDA.record(ceng.ev_dev[b])      # slot b consumed
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b2])           # C chunk is ready
            launch_seg_rowtopk_merge!(D_val, D_loc,
                                      wi == eng.w ? cbuf : @view(cbuf[:, 1:wi]),
                                      lo - 1, eng.k; segs = eng.segs)
            CUDA.record(eng.evt[b2])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

"""
    topk_flow_compact(source, ceng::CompactEngine; out_cap = 2, err_out, tasks_out)
        -> Channel{TopKBatch{Float32}}

`topk_flow` against a CompactEngine: identical contract (fp16 rope batches in,
fresh host TopKBatch matrices out, pad rows sliced off, early-close teardown,
error routing) -- only the per-batch GEMM+top-k call is the chunk-streaming
`batch_gemm_topk_compact!`.
"""
function topk_flow_compact(source, ceng::CompactEngine;
                           out_cap::Int = 2,
                           err_out::Ref{Any} = Ref{Any}(nothing),
                           tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k = ceng.eng.k
    cap = ceng.eng.rows_cap
    out = Channel{TopKBatch{Float32}}(out_cap)
    t_topk = Threads.@spawn begin
        try
            for bat in source
                emb = bat.embeds
                eltype(emb) == Float16 ||
                    throw(ArgumentError("batch eltype $(eltype(emb)) != Float16 (build the " *
                                        "rope stream with fp16 = true; this stage quantizes " *
                                        "the embeddings to e4m3 on the GPU)"))
                nb = size(emb, 1)
                nb == 0 && continue
                nb <= cap ||
                    throw(ArgumentError("batch height $nb > engine rows_cap $cap"))
                # upload/quantize on the GEMM stream (keeps multi-ms kernel
                # queues OFF the default stream -- flowtopkfp8's CUDACore note)
                CUDA.stream!(ceng.eng.sg) do
                    upload_fp16_as_f8!(ceng.eng, emb)
                end
                D_val = CuMatrix{Float32}(undef, cap, k)
                D_loc = CuMatrix{Int32}(undef, cap, k)
                batch_gemm_topk_compact!(D_val, D_loc, ceng)
                vals = _gather_rows(@view(D_val[1:nb, :])) # fresh host matrices,
                locs = _gather_rows(@view(D_loc[1:nb, :])) # pad rows sliced off
                D_val = D_loc = nothing # device pool reuses them next batch
                put!(out, TopKBatch(vals, locs, bat.norms, bat.heads, bat.first))
            end
        catch err
            _topkfp8_flow_error(err, err_out, out; who = "topkfp8-compact")
            source isa AbstractChannel && close(source)
        finally
            close(out) # also the early-close path (close on closed is a no-op)
        end
    end
    append!(tasks_out[], [t_topk])
    return out
end

"""
    rope_topk_stream_compact(re, file, Bh; <rope_topk_stream's kwargs>,
                             depth = 3, resident = nothing)
        -> Channel{TopKBatch{Float32}}

The full fp8 flow for a HOST-resident database `Bh` (rdim x N e4m3, rdim
must equal the encoder's 2*m*4^c).  `resident` picks the engine:
  `nothing` (default) -- AUTO: when `Bh` + the engine's working set fits in
      the free VRAM (`resident_fits`), upload `Bh` once and run the RESIDENT
      engine (`topk_flow`): bitwise the same results and no per-batch PCIe
      sweep -- e2e_fp8's speed.  Otherwise the chunk-streaming ring below.
  `false` -- always the ring: rope-encode the fragments and sweep all N
      columns per read batch through the depth-slot device ring (`--batch`
      fragments share each sweep -- the lever that amortizes the H2D).
  `true` -- force the resident upload (fails fast when it cannot fit).
Same kwargs (plus `resident`/`depth`), output contract and teardown as
`rope_topk_stream`.
"""
function rope_topk_stream_compact(re::RopeEncoder, file::String, Bh::Matrix{F8};
                                  k::Int = 20, w::Int = 2^16,
                                  batch_size::Int = 2^13, rows_cap::Int = batch_size,
                                  normalize::Int = 0, parts::Int = Threads.nthreads(),
                                  in_cap::Int = 2, out_cap::Int = 2, topk_out_cap::Int = 2,
                                  segs::Int = 8, engine::Symbol = :lt, cout::Symbol = :f16,
                                  depth::Int = 3,
                                  resident::Union{Nothing,Bool} = nothing,
                                  progress::Bool = false,
                                  err_out::Ref{Any} = Ref{Any}(nothing),
                                  tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(Bh, 1) == rdim ||
        throw(ArgumentError("Bh is $(size(Bh, 1)) x $(size(Bh, 2)); the encoder's real " *
                            "embedding dim is 2*m*4^c = $rdim"))
    if resident === nothing
        resident = resident_fits(Bh; rows_cap, rdim, w, cout)
    end
    if resident # the whole database admits VRAM residency: NO per-batch sweep
        need = _resident_need(Bh; rows_cap, rdim, w, cout)
        free = (CUDA.reclaim(); CUDA.free_memory())
        need <= free ||
            throw(ArgumentError("resident = true but the database + engine working set " *
                                "$(round(need / 2^30; digits = 1)) GiB exceeds the " *
                                "$(round(free / 2^30; digits = 1)) GiB free " *
                                "(use resident = nothing/false)"))
        @info "compact flow: the index fits in VRAM -> RESIDENT engine " *
              "(one-time upload, no per-batch sweep)" db_gib =
            round(sizeof(Bh) / 2^30; digits = 2) free_gib = round(free / 2^30; digits = 1)
        eng = TopKEngine(CuMatrix{F8}(Bh); k, w, rows_cap, segs, engine, cout)
        rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                          fp16 = true, parts, in_cap, out_cap,
                                          progress, err_out, tasks_out)
        return topk_flow(rope_ch, eng; out_cap = topk_out_cap, err_out, tasks_out)
    end
    @info "compact flow: the index exceeds free VRAM -> chunk-streaming ring" db_gib =
        round(sizeof(Bh) / 2^30; digits = 2)
    ceng = CompactEngine(Bh; k, w, rows_cap, segs, engine, cout, depth)
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return topk_flow_compact(rope_ch, ceng; out_cap = topk_out_cap, err_out, tasks_out)
end

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end through the COMPACT path: e2ehuman's 1M fixture (reference ->
index -> 1000 err = 0 reads), scored by map_and_count with the chunk-streaming
producer.  At err = 0 every read must map, every first hit at rank 1 -- the
ring/event bookkeeping pinned down exactly.
"""
