# ==============================================================================
# flowtopk.jl -- the full flow: ropeflowreal's streamed REAL rope encodings
# fed straight into gemmtopk's batched fp16 GEMM + per-row top-k against a
# large database matrix resident in VRAM.
#
# PROBLEM
#   The rope stream (ropeflowreal.jl) emits batches of real fragment
#   embeddings embeds_bat ∈ fp16^(2^13 x 2^11) -- batch_size = 2^13 fragments
#   x the [Re; Im] dimension 2*m*4^c = 2^11 at the workflow config
#   (s = 8, m = 4, c = 4, k = 20,000).  A database matrix
#       B ∈ fp16^(2^11 x 2^21)   (8 GiB; 2^21 normalized reference vectors)
#   is GIVEN as input and sits on the GPU.  Wanted: for every streamed
#   fragment the k = 20 best-matching database vectors, i.e. the top-k over
#   each row of
#       C_bat = embeds_bat * B ∈ (2^13 x 2^21)   (~32 GiB fp16 per batch),
#   with the matched database ids.  C_bat is as unmaterializable as
#   gemmtopk's full C (32 GiB vs the RTX 5090's 32 GiB VRAM, and the stream
#   is open-ended) -- so each batch is processed by exactly the gemmtopk
#   machinery: column chunks of B (w = 2^15), double-buffered C chunks on two
#   non-blocking streams, and the fused rowtopk_merge_kernel! merging every
#   chunk's per-row top-k into the batch's running top-k.  One streamed batch
#   = one gemmtopk problem of height 2^13 (there: 2^17 x 2^11 x 2^21 total =
#   here: 16 streamed batches of 2^13 x 2^11 x 2^21).
#
#   Orientation (matching both parents): a fragment's scores against the
#   database are the ROW of C_bat; database vector j is COLUMN j of B; the
#   emitted `locs` are global 1-based column ids into B (valid across the
#   whole stream, not just one batch).
#
# DESIGN
#   1. Stage topology.  flowtopk adds ONE stage on top of the rope pipeline:
#
#     fasta ─fasta_reads─▶ gather ─▶ encode ─▶ Channel{RopeRealBatch}
#                                                │  (out_cap batches in flight;
#                                                 producer keeps encoding ahead)
#              topk stage (this file):  H2D(A) on sg, then per B-chunk
#              [mul! on sg ‖ rowtopk_merge on st] x cld(N, w) chunks,
#              then D2H of the merged (nb x k) results
#                                                ▼
#                                     Channel{TopKBatch}
#
#     The rope pipeline overlaps the encoding of batch i+1 with the GEMM+top-k
#     of batch i through its own channel buffer (the encode stage costs
#     ~30 ms/batch against ~250 ms/batch of GEMM, so it hides completely).
#   2. TopKEngine.  gemmtopk's gemm_topk! allocates its two C buffers, two
#     streams and two event pairs per call; here the SAME per-batch loop runs
#     16+ times, so those are hoisted into a reusable `TopKEngine(B; ...)`
#     (two rows_cap x w fp16 C buffers, e.g. 2 x 512 MiB at w = 2^15, plus
#     streams, event pairs).  Per
#     batch only the small tensors are fresh (A_d 32 MiB, D_val/D_loc
#     nb x k) -- the CUDA pool recycles them.  The event choreography is
#     gemmtopk's verbatim: topk(i) waits gemm(i); gemm(i+2) waits topk(i).
#   3. Kernel.  The engine uses gemmtopk's `seg_rowtopk_merge_kernel!` -- the
#     PS-threads-per-row variant of `rowtopk_merge_kernel!` (same register-
#     reservoir top-k, software-pipelined scan and two-pointer merge; each
#     row is scanned by `segs` threads over W/segs-column segments, staged
#     through shared memory and merged by the block's first warp).  The
#     1-thread kernel's parallelism scales with M: at M = 2^13 (one streamed
#     batch) it launches only 64 blocks and reads C at ~160 GiB/s vs 747 at
#     gemmtopk's M = 2^17 -- the segmented variant decouples parallelism
#     from M (see the kernel's comment in gemmtopk.jl).  The running top-k
#     state is reset per batch with fill!(typemin), since batches are
#     independent sets of queries.  overlap = false runs the same loop
#     sequentially on the default stream (benchmark baseline; results are
#     bitwise identical -- cuBLAS and the kernel are deterministic, and a
#     row's top-k depends only on that row's C values).
#   4. Results contract.  Per batch the stage D2Hs and emits
#        TopKBatch(vals (nb,k) fp16 descending, locs (nb,k) Int32 global
#                  column ids into B, norms (m,nb) fp32 passthrough,
#                  heads, first)
#     -- fresh HOST matrices, retain freely (same contract as RopeRealBatch).
#     320 KiB + 640 KiB per batch at the default config: negligible D2H next
#     to the 32 MiB A upload.  A GPU-side sink is a possible follow-up.
#   5. Types.  B is fp16 (given); the rope stream is forced fp16 = true (the
#     fp16 rope option exists precisely for this consumer); eltype mismatches
#     are hard errors.  `k` is the TOP-K SIZE (gemmtopk's convention); the
#     fragment length is the encoder's re.k throughout (the rope stream
#     requires it and it is not a kwarg here).
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16, min of 3, big file =
# 2.6 GB / 2^17 fragments, B = 2^11 x 2^21 fp16 column-normalized N(0,1),
# k = 20, w = 2^15, batch = 2^13 x 2^11 -> 16 batches of 64 chunks):
#
#   rope stream only, no-op consumer fp16 (H2D+kernel+D2H)   0.541 s    4.86 GB/s
#   flow END-TO-END: fasta -> rope fp16 -> gemm+topk         5.288 s  212.9 TFLOPS*
#   flow COMPUTE-ONLY: gemm+topk (rope excluded, replayed)   5.096 s  220.9 TFLOPS*
#   compute-only, sequential (overlap=false)                 5.13 s    1.01x
#   per-chunk gemm (w=2^15, M=2^13)                          3.950 ms  278 TFLOPS
#   per-chunk seg-rowtopk (segs=8)                           0.780 ms  641 GiB/s of C
#   seg-rowtopk sweep (segs = 2 / 4 / 8):  1.518 / 0.945 / 0.780 ms per chunk
#   old 1-thread kernel, same shapes:      3.103 ms (161 GiB/s) -- 4.0x slower
#   w sweep: 2^12 -> 9.16 s (122.9) · 2^14 -> 5.61 s (200.6) · 2^15 -> 5.17 s (217.9)
#   batch sweep: 2^11 -> 5.93 s (189.9) · 2^12 -> 5.32 s (211.7) · 2^14 -> 5.10 s (220.6)
#   (batch_size = 2^13 kept: halving it loses GEMM/top-k efficiency and doubles
#   the batch boundaries; 2^14 is flat and doubles the engine's C buffers)
#
#   * TFLOPS = 2^50 FLOP / wall time over the whole big file (gemmtopk's
#   monolithic 2^17 x 2^11 x 2^21 run: 4.5 s = 250 TFLOPS).
#
#   The rope stage (~0.5-0.6 s) hides COMPLETELY behind the compute stage
#   (end-to-end 5.29 s vs compute-only 5.10 s).  The GEMM floor is 4.04 s
#   (16 x 64 x 3.950 ms at 278 TFLOPS -- skinnier GEMMs than gemmtopk's
#   315-TFLOPS ones), the seg top-k adds 0.80 s at 641 GiB/s, residual
#   overhead ~0.26 s.  History: with the verbatim 1-thread kernel the flow
#   ran 7.43 s / 151.5 TFLOPS -- that kernel's top-k read C at only ~160
#   GiB/s at M = 2^13 (64 blocks on 170 SMs vs gemmtopk's 1024); the
#   segmented kernel recovered 4x in the kernel, and the fatter default
#   chunk (w = 2^15; per-chunk launch/event overhead dominated at w <= 2^13)
#   most of the rest.
#
# CORRECTNESS (all on kau, julia -t 16): seg_rowtopk_merge_kernel! has its
# own exact CPU-multiset unit test (ragged / strided-parent / w == k /
# empty-segment / real-geometry shapes); engine vs an exact chunk-wise cuBLAS
# reference (top-k values AND locations bitwise) over ragged (w ∤ N), w == N,
# single-row and real-geometry (2^13 x 2^15) shapes, a tail batch on a reused
# engine, pipelined == sequential; end-to-end on the 164 MB file vs the
# float64 CPU reference (independent rope encoder + fp64 GEMM): batch layouts
# (tail / exact fit / single), normalize 0/1, encoder config (5,1,4) (rdim
# 512), sampled rows' values + locations + norms passthrough (fp16 pipeline
# value noise measured at ~3e-3 absolute on scores |.| <~ 0.2 -- locations
# exact: scores at the returned locations equal the reference top-k),
# heads/first bookkeeping, early-close teardown (all 3 tasks joined, err_out
# clean, pipeline reusable), empty file -> 0 batches.  ALL PASS.
# gemmtopk.jl --quick re-run after the kernel changes: ALL PASS (determinism
# pipelined == sequential, unchanged spot-check accuracy 0.90).
#
# KNOWN RISK / FOLLOW-UPS
#   * top-k parallelism at M = 2^13: RESOLVED by seg_rowtopk_merge_kernel!
#     (segs = 8: 0.780 ms/chunk at 641 GiB/s vs 3.103 ms at 161 GiB/s for the
#     1-thread kernel; see RESULTS).  segs > 8 (fp16, K = 20) needs > 48 KiB
#     shared memory, i.e. an opt-in cuFuncSetAttribute -- diminishing returns
#     anyway (segs 4 -> 8 buys only ~20%).  The 1-thread rowtopk_merge_kernel!
#     remains in gemmtopk.jl for the monolithic gemm_topk! path, where
#     M = 2^17 fills the GPU on its own.
#   * the remaining gap to gemmtopk's 4.5 s is mostly the GEMM itself
#     (278 vs 315 TFLOPS: M = 2^13 GEMMs are skinnier) + 0.8 s of top-k.
#   * async pinned D2H ring for TopKBatch (currently one synchronizing copy
#     of ~1 MiB per batch, already negligible).
#   * revc parity, normalize-mode plumbing -- inherit ropeflowreal's notes.
#
# Run modes (ARGV[1]):
#   gen   generate the test fasta files if missing (fastareads_v3)
#   test  correctness: exact engine check vs chunk-wise cuBLAS reference
#         (ragged/w==N/single-row shapes, tail batches, overlap equivalence),
#         end-to-end flow on the 164 MB file vs a float64 CPU reference
#         (batch layouts incl. single-batch, normalize 0/1, a second encoder
#         config, values + locations + sortedness), early-close teardown,
#         empty file
#   bench  end-to-end flow on the 2.6 GB file (fasta -> top-k results) and
#         the compute-only stage (rope excluded, replayed embeddings), plus
#         per-chunk kernel timings, thread-count and w sweeps
#   all   everything above (default)
#
# USAGE
#   julia --project=. test/flowtopk.jl            # gen + test + bench
#   julia --project=. test/flowtopk.jl test
# ------------------------------------------------------------------------------
# NEW TREE NOTES (reorganization): this file is now a LIBRARY layer
# (search/engine_fp16.jl).  The legacy include block (ropeflowreal.jl +
# gemmtopk.jl) is replaced by the entry script's canonical include order;
# requires:
#   common/dna.jl, common/util.jl (_timed_gpu, _reps), common/testref.jl
#   (ensure_data3), fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl,
#   gemm/fp16.jl (gemm_topk! AND the original fp16 kernels:
#   seg_rowtopk_merge_kernel!, launch_seg_rowtopk_merge!), reads/provenance.jl (parse_read_head).
# The provenance scorer `map_and_count` (legacy e2ehuman16.jl, fp16 variant)
# lives near the bottom of this file.  NOT co-includable with engine_fp8.jl /
# engine_complex.jl (same names: TopKEngine, topk_flow, rope_topk_stream,
# map_and_count, ...).  The PROGRAM_FILE dispatch is kept: running this file
# directly still offers gen|test|bench|all.
# ------------------------------------------------------------------------------

using CUDA
using LinearAlgebra
using Random
using Printf
using Base.Threads

# ==============================================================================
# The batched GEMM + top-k engine (gemmtopk's per-batch loop, state hoisted)
# ==============================================================================

"""
One streamed batch of top-k results.  `vals[r, :]` are fragment r's k best
scores against the database B, DESCENDING; `locs[r, :]` the matching GLOBAL
1-based column ids into B (scores[r, j] = dot(embeds_r, B[:, locs[r, j]])).
`norms`/`heads`/`first` are the rope batch's passthrough (per-copy norms
(m, nb) Float32, fasta headers, 1-based global first-fragment index).  Fresh
HOST matrices -- retain freely.
"""
struct TopKBatch{T<:Union{Float16,Float32}}
    vals::Matrix{T}            # (nb, k), descending
    locs::Matrix{Int32}        # (nb, k), global column ids into B
    norms::Matrix{Float32}     # (m, nb), passthrough
    heads::Vector{String}
    first::Int
end

"""
Reusable per-batch GEMM+top-k state for one database matrix: gemmtopk's
double-buffered chunk loop with the two fp16 C buffers (`rows_cap x w` each),
the two non-blocking streams and the two event pairs allocated ONCE instead
of per batch.  `k` is the top-k size, `w` the B-column chunk width,
`rows_cap` the batch-height capacity (every streamed batch must have
nb <= rows_cap).  Requires 1 <= k <= w <= N.
"""
struct TopKEngine{T<:Union{Float16,Float32}}
    B::CuMatrix{T}             # (rdim, N) resident database
    buf1::CuMatrix{T}          # the two (rows_cap, w) C chunk buffers,
    buf2::CuMatrix{T}          #   double-buffered: chunk i uses mod1(i, 2)
    k::Int                     # top-k size
    w::Int                     # B-column chunk width
    sg::CuStream               # GEMM stream
    st::CuStream               # top-k stream
    evg::NTuple{2,CuEvent}     # evg[b]: gemm of the chunk using buffer b done
    evt::NTuple{2,CuEvent}     # evt[b]: top-k of the chunk using buffer b done
    segs::Int                   # column segments per row for the top-k kernel
end

function TopKEngine(B::CuMatrix{T}; k::Int = 20, w::Int = 2^15,
                    rows_cap::Int = 2^13, segs::Int = 8) where {T<:Union{Float16,Float32}}
    rdim, N = size(B)
    1 <= k <= w <= N ||
        throw(ArgumentError("need 1 <= k <= w <= N (got k = $k, w = $w, N = $N)"))
    rows_cap >= 1 || throw(ArgumentError("rows_cap must be >= 1"))
    buf1 = CuMatrix{T}(undef, rows_cap, w)
    buf2 = CuMatrix{T}(undef, rows_cap, w)
    sg = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    st = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    TopKEngine(B, buf1, buf2, Int(k), Int(w), sg, st, (CuEvent(), CuEvent()),
               (CuEvent(), CuEvent()), Int(segs))
end

"""H2D-upload one batch of host embeddings on the engine's GEMM stream
(ordered before the first chunk's mul!), returning the fresh device matrix."""
function upload_batch(eng::TopKEngine{T}, embeds::Matrix{T}) where {T<:Union{Float16,Float32}}
    nb = size(embeds, 1)
    @assert size(embeds, 2) == size(eng.B, 1) "embeds dim $(size(embeds, 2)) != B rows $(size(eng.B, 1))"
    A_d = CuMatrix{T}(undef, nb, size(eng.B, 1))
    CUDA.stream!(eng.sg) do
        copyto!(A_d, embeds) # ~32 MiB at the default config, pageable src
    end
    return A_d
end

"""
One streamed batch: gemmtopk's `gemm_topk!` chunk loop against the engine's
resident B, with the running top-k RESET first (batches are independent query
sets).  `A` is the batch's device embeddings (nb, rdim); `D_val`/`D_loc`
(nb, k) receive the merged top-k (descending, global column ids into B).
`overlap = false` runs sequentially on the default stream (benchmark
baseline; bitwise identical results).  Returns with the device synchronized.
"""
function batch_gemm_topk!(D_val::CuMatrix{T}, D_loc::CuMatrix{Int32},
                          eng::TopKEngine{T}, A::CuMatrix{T};
                          overlap::Bool = true) where {T<:Union{Float16,Float32}}
    M, KA = size(A)
    rdim, N = size(eng.B)
    @assert KA == rdim "A's inner dim must match B's rows"
    @assert size(D_val) == (M, eng.k) && size(D_loc) == (M, eng.k)
    @assert M <= size(eng.buf1, 1) "batch height $M > engine rows_cap $(size(eng.buf1, 1))"

    fill!(D_val, typemin(T)) # reset the running top-k; the kernel merges into it
    fill!(D_loc, Int32(0))
    CUDA.device_synchronize()

    sg = overlap ? eng.sg : CUDA.default_stream()
    st = overlap ? eng.st : sg
    for i in 1:cld(N, eng.w)
        b = mod1(i, 2)
        lo = (i - 1) * eng.w + 1
        wi = min(N, i * eng.w) - lo + 1
        buf = b == 1 ? eng.buf1 : eng.buf2
        Bv = @view eng.B[:, lo:lo+wi-1]
        Cv = @view buf[1:M, 1:wi]

        CUDA.stream!(sg) do
            i > 2 && CUDA.wait(eng.evt[b])   # buffer b's previous top-k finished
            mul!(Cv, A, Bv)                  # cuBLAS fp16 GEMM (tensor cores)
            CUDA.record(eng.evg[b])
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b])            # C chunk is ready
            launch_seg_rowtopk_merge!(D_val, D_loc, Cv, lo - 1, eng.k; segs = eng.segs)
            CUDA.record(eng.evt[b])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

# ==============================================================================
# The streamed flow
# ==============================================================================

# Task-level error router: same contract as ropeflowreal's
# `_rope_real_stream_error` (InvalidStateException = benign teardown, never a
# data error; real errors print immediately, land in err_out, first error
# wins, and every channel is closed so all stages unwind).
function _topk_flow_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "flowtopk: $(who) failed: ",
            sprint(showerror, err, catch_backtrace()))
    flush(stderr)
    err_out[] === nothing && (err_out[] = err)
    for ch in chs
        try
            close(ch)
        catch
        end
    end
    return false
end

"""
    topk_flow(source, eng::TopKEngine; out_cap = 2, err_out, tasks_out)
        -> Channel{TopKBatch}

Consume `source` -- any iterable of rope batches (anything with
`.embeds/.norms/.heads/.first`, e.g. a `Channel{RopeRealBatch{Float16}}` or a
pre-materialized `Vector{RopeRealBatch}`) -- and, for every batch, run
`batch_gemm_topk!` against the engine's resident database, emitting
`TopKBatch(vals, locs, norms, heads, first)` with fresh HOST matrices.  The
stage runs on its own task; batches flow through a `Channel(out_cap)` so the
upstream rope pipeline keeps encoding ahead (backpressure: a slow consumer of
the results throttles the whole pipeline).  Batch eltype must match the
engine's (build the rope stream with `fp16 = true` for an fp16 database);
batch height must be <= the engine's rows_cap.

Consume to the end (the channel closes itself) or `close(ch)` early: the
stage closes `source` when it is a channel, which tears the rope pipeline
down quietly (ropeflowreal's contract).  Real failures land in `err_out[]`
after the stream ends short; `tasks_out` receives this stage's task (pass the
same Ref to the rope stream to collect its two tasks as well, so a caller
that stopped early can `wait` all of them).
"""
function topk_flow(source, eng::TopKEngine{T};
                   out_cap::Int = 2,
                   err_out::Ref{Any} = Ref{Any}(nothing),
                   tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[])) where {T}
    k = eng.k
    cap = size(eng.buf1, 1)
    out = Channel{TopKBatch{T}}(out_cap)
    t_topk = Threads.@spawn begin
        try
            for bat in source
                emb = bat.embeds
                eltype(emb) == T ||
                    throw(ArgumentError("batch eltype $(eltype(emb)) != engine eltype $T " *
                                        "(build the rope stream with fp16 = $(T == Float16))"))
                nb = size(emb, 1)
                nb == 0 && continue
                nb <= cap ||
                    throw(ArgumentError("batch height $nb > engine rows_cap $cap"))
                A_d = upload_batch(eng, emb)
                D_val = CuMatrix{T}(undef, nb, k)
                D_loc = CuMatrix{Int32}(undef, nb, k)
                batch_gemm_topk!(D_val, D_loc, eng, A_d)
                vals = Matrix{T}(undef, nb, k)
                locs = Matrix{Int32}(undef, nb, k)
                copyto!(vals, D_val) # synchronizing D2H, ~1 MiB at the default config
                copyto!(locs, D_loc)
                A_d = D_val = D_loc = nothing # device pool reuses them next batch
                put!(out, TopKBatch(vals, locs, bat.norms, bat.heads, bat.first))
            end
        catch err
            if _topk_flow_error(err, err_out, out; who = "topk")
                # benign teardown (the consumer closed `out`): stop the
                # upstream rope pipeline; it unwinds quietly on its own
                source isa AbstractChannel && close(source)
            end
        finally
            close(out) # also the early-close path (close on closed is a no-op)
        end
    end
    append!(tasks_out[], [t_topk])
    return out
end

"""
    rope_topk_stream(re::RopeEncoder, file::String, B::CuMatrix{Float16};
                     k = 20, w = 2^15, batch_size = 2^13, rows_cap = batch_size,
                     normalize = 0, parts = nthreads(), in_cap = 2, out_cap = 2,
                     topk_out_cap = 2, segs = 8, progress = false,
                     err_out = Ref{Any}(nothing),
                     tasks_out = Ref{Vector{Task}}(Task[]))
                     -> Channel{TopKBatch}

The full flow on a fasta `file`: stream-rope-encode the fragments
(`rope_encode_real_stream` with `fp16 = true` -- the fragment length is the
encoder's `re.k`) and process every batch with the engine's double-buffered
fp16 GEMM + fused per-row top-k against the RESIDENT database
`B ∈ fp16^(rdim x N)` (rdim must equal the encoder's 2*m*4^c; its columns are
the reference vectors, assumed normalized).  `k` is the TOP-K SIZE, `w` the
B-column chunk width, `batch_size` the fragments per streamed batch,
`rows_cap` the engine's batch-height capacity (C buffers: 2 * rows_cap * w
fp16); a batch taller than `rows_cap` is an error, so cap it at the file's
read count when `batch_size` is an over-estimate.  Emits
`TopKBatch(vals (nb,k) fp16, locs (nb,k) Int32 global column ids into B,
norms, heads, first)` with fresh host matrices; `nb <= batch_size` (only the
last batch is partial).  Fragment order within the stream is the parsers'
arbitrary interleaving (headers make every row self-describing); per-row
results are deterministic.

Consume to the end or `close(ch)` early (quiet teardown of the whole
pipeline); `err_out`/`tasks_out` wire both stages together (three tasks).
"""
function rope_topk_stream(re::RopeEncoder, file::String, B::CuMatrix{Float16};
                          k::Int = 20, w::Int = 2^15,
                          batch_size::Int = 2^13, rows_cap::Int = batch_size,
                          normalize::Int = 0, parts::Int = Threads.nthreads(),
                          in_cap::Int = 2, out_cap::Int = 2, topk_out_cap::Int = 2,
                          segs::Int = 8, progress::Bool = false,
                          err_out::Ref{Any} = Ref{Any}(nothing),
                          tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(B, 1) == rdim ||
        throw(ArgumentError("B is $(size(B, 1)) x $(size(B, 2)); the encoder's real " *
                            "embedding dim is 2*m*4^c = $rdim"))
    eng = TopKEngine(B; k, w, rows_cap, segs)
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return topk_flow(rope_ch, eng; out_cap = topk_out_cap, err_out, tasks_out)
end

# ==============================================================================
# Correctness.
# ==============================================================================

# The engine's chunk loop, exactly: fresh GEMM per chunk (same shapes AND
# leading dimensions as the engine's strided mul! views -> bitwise-identical
# C values on a deterministic cuBLAS), materialized on the host for an exact
# CPU-reference comparison of the merged top-k.
function _ref_topk_chunked(A_d::CuMatrix{T}, B_d::CuMatrix{T},
                           rows_cap::Int, w::Int) where {T<:Union{Float16,Float32}}
    M, N = size(A_d, 1), size(B_d, 2)
    Cbuf = CuMatrix{T}(undef, rows_cap, w)
    Ch = Matrix{T}(undef, M, N)
    for i in 1:cld(N, w)
        lo = (i - 1) * w + 1
        wi = min(N, i * w) - lo + 1
        Cv = @view Cbuf[1:M, 1:wi]
        mul!(Cv, A_d, @view B_d[:, lo:lo+wi-1])
        Ch[:, lo:lo+wi-1] = Array(Cv)
    end
    return Ch
end

function _assert_topk_rows(Ch::Matrix{T}, Dv::Matrix{T}, Dl::Matrix{Int32},
                           k::Int) where {T<:Union{Float16,Float32}}
    M, N = size(Ch)
    for r in 1:M
        ref = partialsort!(Vector{T}(Ch[r, :]), 1:k; rev = true)
        got = Dv[r, :]
        @assert issorted(got; rev = true) "row $r: output not sorted"
        @assert sort(got; rev = true) == ref "row $r: wrong top-$k values"
        @assert all(>(0), Dl[r, :]) && Ch[r, Dl[r, :]] == got "row $r: wrong locations"
    end
    return nothing
end

# Engine vs an exact chunk-wise reference over several shapes (incl. ragged
# w ∤ N, w == N, single row, the real batch geometry), plus a tail batch
# (nb < rows_cap) on a REUSED engine and the pipelined == sequential
# determinism check.
function _check_engine_gemm_topk(; k::Int = 20, seed = 123)
    CUDA.seed!(seed)
    for (rdim, M, N, w) in ((64, 64, 100, 37), (64, 257, 128, 128),
                            (32, 1000, 333, 111), (64, 1, 4096, 512),
                            (64, 511, 2048, 2048), (128, 8192, 2^15, 2^13))
        A_d = CuMatrix{Float16}(randn(Float32, M, rdim))
        B_d = CuMatrix{Float16}(randn(Float32, rdim, N))
        eng = TopKEngine(B_d; k, w, rows_cap = M)
        D_val = CuMatrix{Float16}(undef, M, k)
        D_loc = CuMatrix{Int32}(undef, M, k)
        batch_gemm_topk!(D_val, D_loc, eng, A_d)
        Ch = _ref_topk_chunked(A_d, B_d, M, w)
        _assert_topk_rows(Ch, Array(D_val), Array(D_loc), k)

        if M == 257 # tail batch on a reused engine: nb < rows_cap
            A2 = A_d[1:100, :]
            D_val2 = CuMatrix{Float16}(undef, 100, k)
            D_loc2 = CuMatrix{Int32}(undef, 100, k)
            batch_gemm_topk!(D_val2, D_loc2, eng, A2)
            Ch2 = _ref_topk_chunked(A2, B_d, size(eng.buf1, 1), w)
            _assert_topk_rows(Ch2, Array(D_val2), Array(D_loc2), k)

            # sequential (no overlap) must be bitwise identical
            D_val3 = similar(D_val2)
            D_loc3 = similar(D_loc2)
            batch_gemm_topk!(D_val3, D_loc3, eng, A2; overlap = false)
            @assert Array(D_val3) == Array(D_val2) && Array(D_loc3) == Array(D_loc2) "overlap=false diverged"
        end
        @info "  engine exact check OK (rdim=$rdim, M=$M, N=$N, w=$w, k=$k)"
    end
    return nothing
end

# End-to-end: stream the fasta file through the full flow and compare the
# top-k against a float64 CPU reference (embeddings from ropeflowreal's
# independent reference encoder, matched by header -- the parsers' batch
# composition is arbitrary; scores use the same B on the host in float64).
function _check_topk_stream(re, path, B_d::CuMatrix{Float16}, Bh::Matrix{Float64};
                            k::Int, w::Int, batch_size::Int, rows_cap::Int,
                            normalize::Int, kfrag::Int,
                            refw::Dict{String,Vector{UInt32}},
                            sample::Int, seed = 13)
    rs = MersenneTwister(seed)
    N = size(B_d, 2)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    nbatches = seen = 0
    heads_all = String[]
    for bat in rope_topk_stream(re, path, B_d; k, w, batch_size, rows_cap,
                                normalize, err_out = err, tasks_out = tks)
        nbatches += 1
        nb = length(bat.heads)
        @assert bat isa TopKBatch && eltype(bat.vals) == Float16 "batch type/eltype mismatch"
        @assert size(bat.vals) == (nb, k) && size(bat.locs) == (nb, k) "batch shape mismatch"
        @assert size(bat.norms) == (re.m, nb) "norms passthrough shape mismatch"
        @assert bat.first == seen + 1 "batch first-index mismatch"
        for r in 1:nb
            @assert issorted(@view(bat.vals[r, :]); rev = true) "row $r: not sorted"
            @assert all(lo -> lo in 1:N, @view(bat.locs[r, :])) "row $r: locations out of range"
        end
        if normalize == 0 # shared raw norm + partials: positive & finite
            @assert all(x -> isfinite(x) && x > 0, bat.norms) "norm passthrough not positive/finite"
        end
        # sampled rows against the float64 CPU reference
        # fp16 embedding quantization + fp16 tensor-core GEMM noise: measured
        # ~3e-3 absolute on scores |.| <~ 0.2 (locations are robust -- scores
        # at the returned locations match the reference top-k to ~1e-6 gaps)
        atol = 1e-2
        for r in rand(rs, 1:nb, min(sample, nb))
            head = bat.heads[r]
            href, nrmref = _ref_rope_frag_real(_frag_codes(refw[head], kfrag), re; normalize)
            scores = Bh' * href # (N,) float64 reference scores
            refv = partialsort!(copy(scores), 1:k; rev = true)
            gotv = bat.vals[r, :]
            gotl = bat.locs[r, :]
            # norms passthrough vs the CPU reference (mode 0: only row 1 is the norm)
            for idm in (normalize == 0 ? (1:1) : (1:re.m))
                @assert isapprox(bat.norms[idm, r], nrmref[idm]; rtol = 1e-3) "norms passthrough mismatch ($(head), copy $idm)"
            end
            @assert isapprox(gotv, refv; atol, rtol = 0) "top-k values vs CPU reference mismatch ($(head))"
            @assert all(abs.(scores[gotl] .- gotv) .<= atol) "value/location inconsistency ($(head))"
        end
        append!(heads_all, bat.heads)
        seen += nb
    end
    foreach(wait, tks[]) # deterministic: all three stages have unwound
    @assert err[] === nothing "stream error: $(err[])"
    @assert seen == length(refw) "fragment count mismatch: $seen vs $(length(refw))"
    @assert nbatches == cld(seen, batch_size) "batch count mismatch ($nbatches for batch_size=$batch_size)"
    @assert sort(heads_all) == sort(collect(keys(refw))) "streamed heads mismatch"
    return nbatches
end

function run_flow_topk_test()
    @show CUDA.name(device())
    @show nthreads()

    small, _big = ensure_data3()
    kfrag = 20_000
    k = 20
    dir = mktempdir(prefix = "flowtopk_")

    # ==========================================================================
    # A. ENGINE vs an exact chunk-wise cuBLAS reference: ragged / w == N /
    #    single-row / real-geometry shapes, tail batch on a reused engine,
    #    pipelined == sequential.
    # ==========================================================================
    @info "A. batch_gemm_topk! engine vs exact chunked reference"
    test_seg_rowtopk_merge_kernel(k = k)    # the engine's top-k kernel, directly
    _check_engine_gemm_topk(k = k)

    # ==========================================================================
    # B. FLOW end-to-end on the small generated file (2^13 reads x ~20 kb):
    #    batch layouts (tail, exact fit, single batch), normalize 0/1, a
    #    second encoder config, values + locations vs the float64 CPU
    #    reference, heads/first bookkeeping.  B_test's columns are normalized
    #    N(0,1) vectors, like the production database.
    # ==========================================================================
    @info "B. rope_topk_stream on the small file vs the float64 CPU reference"
    re = RopeEncoder(k = kfrag, s = 8, m = 4, c = 4) # rdim = 2^11
    rdim = 2 * re.m * 4^re.c
    refw = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k = kfrag, parts = nthreads())
        refw[f.header] = f.words
    end
    @assert length(refw) == _V3_SMALL_READS "unexpected reference fragment count"

    N = 3000
    B_h = randn(Float32, rdim, N)
    B_h ./= sqrt.(sum(abs2, B_h; dims = 1))
    B_d = CuMatrix{Float16}(B_h)
    Bh = Float64.(B_h)

    for batch_size in (3000, 2^13, 10^6) # tail, exact fit, single batch
        nb = _check_topk_stream(re, small, B_d, Bh; k, w = 512, batch_size,
                                rows_cap = min(batch_size, _V3_SMALL_READS),
                                normalize = 0, kfrag, refw, sample = 4)
        @info "  normalize=0, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_topk_stream(re, small, B_d, Bh; k, w = 512, batch_size = 3000,
                            rows_cap = 3000, normalize = 1, kfrag, refw, sample = 4)
    @info "  normalize=1, batch_size=3000 OK ($nb batches)"

    # other encoder config through the whole flow (rdim = 2*1*4^4 = 512)
    re2 = RopeEncoder(k = kfrag, s = 5, m = 1, c = 4)
    rdim2 = 2 * re2.m * 4^re2.c
    N2 = 777
    B_h2 = randn(Float32, rdim2, N2)
    B_h2 ./= sqrt.(sum(abs2, B_h2; dims = 1))
    B_d2 = CuMatrix{Float16}(B_h2)
    nb = _check_topk_stream(re2, small, B_d2, Float64.(B_h2); k, w = 512,
                            batch_size = 3000, rows_cap = 3000, normalize = 0,
                            kfrag, refw, sample = 4)
    @info "  config (s=5, m=1, c=4) end-to-end OK ($nb batches)"

    # ==========================================================================
    # C. teardown: closing the result stream early must unwind the top-k stage
    #    AND the whole rope pipeline quietly (no hang, no spurious err_out)
    #    and leave everything reusable.
    # ==========================================================================
    @info "C. early close"
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    ch = rope_topk_stream(re, small, B_d; k, w = 512, batch_size = 3000,
                          rows_cap = 3000, err_out = err, tasks_out = tks)
    got = 0
    for _ in ch
        got += 1
        got == 1 && break
    end
    close(ch)
    foreach(wait, tks[]) # deterministic: topk stage + gather + encode all done
    @assert err[] === nothing "early close raised: $(err[])"
    @assert got == 1
    @assert all(istaskdone, tks[]) && length(tks[]) == 3
    nb = _check_topk_stream(re, small, B_d, Bh; k, w = 512, batch_size = 3000,
                            rows_cap = 3000, normalize = 0, kfrag, refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    # ==========================================================================
    # D. empty file -> zero batches, no error.
    # ==========================================================================
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    err = Ref{Any}(nothing)
    n = 0
    for _ in rope_topk_stream(re, empty_fasta, B_d; k, w = 512, err_out = err)
        n += 1
    end
    @assert n == 0 && err[] === nothing
    @info "D. empty file OK (0 batches)"

    @info "ALL FLOWTOPK CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks: end-to-end flow (fasta -> rope fp16 -> GEMM+top-k) on the big
# file, the compute-only stage (rope excluded, embeddings replayed), the rope
# stage alone, per-kernel timings, thread-count and w sweeps.  B is the
# production shape 2^11 x 2^21 fp16 with normalized columns.  Run under
# `julia -t N`.
# ==============================================================================

# (rdim, N) fp16 database of column-normalized N(0,1) vectors, built chunk by
# chunk in fp32 (RNG + normalization in fp32, store fp16).
function _randn_unit_fp16(rdim::Int, N::Int; chunk::Int = 2^12)
    B = CUDA.zeros(Float16, rdim, N)
    tmp = CuMatrix{Float32}(undef, rdim, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(tmp)
        v = @view tmp[:, 1:wi]
        v ./= sqrt.(sum(abs2, v; dims = 1))
        @views B[:, lo:lo+wi-1] .= v
    end
    return B
end

function bench_flow_topk(; reps = _reps())
    small, big = ensure_data3()
    _warm_cache(big)
    kfrag = 20_000
    re = RopeEncoder(k = kfrag, s = 8, m = 4, c = 4)
    rdim = 2 * re.m * 4^re.c # 2^11
    N = 2^21
    k = 20
    w = 2^15 # bench-measured optimum at M = 2^13 (fewer chunk boundaries;
    #         gemmtopk's 2^13 optimum was for its 16x taller M = 2^17 batches)
    rows = 2^13
    dev = CUDA.device()
    @info "flowtopk benchmarks" gpu = CUDA.name(dev) vram_gb = round(CUDA.totalmem(dev) / 2^30; digits = 1) julia_threads = nthreads() reps
    @info @sprintf("B: %d x %d fp16 (%.1f GiB), C_bat would be %.0f GiB, k=%d, w=%d, batch=%d x %d",
                   rdim, N, rdim * N * 2 / 2^30, rows * N * 2 / 2^30, k, w, rows, rdim)
    flops_total = 2.0 * _V3_BIG_READS * rdim * N # 2^50 FLOP over the big file
    nb_batches = cld(_V3_BIG_READS, rows)

    # ---- the resident database ----------------------------------------------
    @info "generating B on GPU (column-normalized N(0,1))..."
    tB = @elapsed B = _randn_unit_fp16(rdim, N)
    @info @sprintf("  B ready in %.1f s (device pool: %.2f GiB used)", tB, CUDA.used_memory() / 2^30)

    # ---- rope stage alone (accounting reference; ropeflowreal's fp16 line) ---
    _timed_min("rope stream only, no-op consumer fp16 (H2D+kernel+D2H)"; bytes = filesize(big), reps) do
        n = 0
        for nt in rope_encode_real_stream(re, big; k = kfrag, fp16 = true)
            n += length(nt.heads)
        end
        n
    end

    # ---- end-to-end: fasta -> rope fp16 -> GEMM+top-k -> TopKBatch drain -----
    t_e2e = _timed_min("flow END-TO-END: fasta -> rope fp16 -> gemm+topk (big file)"; bytes = filesize(big), reps) do
        n = 0
        for bat in rope_topk_stream(re, big, B; k, w, batch_size = rows)
            n += size(bat.vals, 1)
        end
        n
    end
    @printf("  end-to-end: %d fragments -> %d batches in %.2f s  =>  %.1f TFLOPS sustained over 2^50 FLOP\n",
            _V3_BIG_READS, nb_batches, t_e2e, flops_total / t_e2e / 1e12)

    # ---- compute-only: the rope stage excluded (embeddings replayed) --------
    # one real fp16 batch of embeddings, replayed 16x as the source
    embeds = Matrix{Float16}(undef, 0, rdim)
    for nt in rope_encode_real_stream(re, big; k = kfrag, fp16 = true)
        embeds = nt.embeds
        break
    end
    @assert size(embeds) == (rows, rdim)
    heads1 = [string("frag_", i) for i in 1:rows]
    norms1 = fill(1f0, re.m, rows)
    src = [RopeRealBatch(embeds, norms1, heads1, (i - 1) * rows + 1) for i in 1:nb_batches]
    eng = TopKEngine(B; k, w, rows_cap = rows)
    t_cmp = _timed_min("flow COMPUTE-ONLY: gemm+topk, rope excluded (replayed embeds)"; bytes = 0, reps) do
        n = 0
        for bat in topk_flow(src, eng)
            n += size(bat.vals, 1)
        end
        n
    end
    @printf("  compute-only: %.2f s  =>  %.1f TFLOPS  (gemmtopk's monolithic 2^17 x 2^11 x 2^21 baseline: ~4.5 s)\n",
            t_cmp, flops_total / t_cmp / 1e12)

    # sequential baseline on the same source (single rep)
    CUDA.device_synchronize(); t0 = time()
    for bat in topk_flow(src, eng)
        size(bat.vals, 1)
    end
    t_seq = time() - t0
    @printf("  compute-only sequential: %.2f s  (overlap speedup %.2fx)\n", t_seq, t_seq / t_cmp)

    # ---- batch-height sweep (compute-only, single runs) ----------------------
    # smaller streamed batches buy nothing: GEMM tiles get skinnier (lower
    # TFLOPS), the seg-rowtopk grid loses blocks (less read bandwidth), and
    # there are twice as many batch boundaries per halving; A_d fits L2 at
    # any of these sizes, so there is no cache upside either
    for nb in (2^11, 2^12, 2^14)
        eh = nb <= rows ? embeds[1:nb, :] : vcat(embeds, embeds)[1:nb, :]
        heads_nb = [string("frag_", i) for i in 1:nb]
        src_nb = [RopeRealBatch(eh, fill(1f0, re.m, nb), heads_nb, (i - 1) * nb + 1)
                  for i in 1:cld(_V3_BIG_READS, nb)]
        eng_nb = TopKEngine(B; k, w, rows_cap = nb)
        CUDA.device_synchronize(); t0 = time()
        n = 0
        for bat in topk_flow(src_nb, eng_nb)
            n += size(bat.vals, 1)
        end
        tw = time() - t0
        @printf("  batch=%5d (%3d batches, %4.0f MiB/buffer): %6.2f s  =>  %.1f TFLOPS\n",
                nb, cld(_V3_BIG_READS, nb), nb * w * 2 / 2^20, tw, flops_total / tw / 1e12)
        eng_nb = nothing
    end

    # ---- per-kernel timings at the real batch geometry -----------------------
    A_d = upload_batch(eng, embeds)
    D_val = CuMatrix{Float16}(undef, rows, k)
    D_loc = CuMatrix{Int32}(undef, rows, k)
    batch_gemm_topk!(D_val, D_loc, eng, A_d) # warm + fill buffer 1
    iters = 20
    tg = CUDA.@elapsed CUDA.@sync for _ in 1:iters
        mul!(@view(eng.buf1[:, 1:w]), A_d, @view(B[:, 1:w]))
    end
    @printf("  per-chunk (w=%d, M=%d): gemm %.3f ms (%.1f TFLOPS over the batch's chunks)\n",
            w, rows, tg / iters * 1e3, 2.0 * rows * rdim * w / (tg / iters) / 1e12)
    scratch = CuMatrix{Float16}(undef, rows, w) # one C buffer, reused by the sweep
    # warm each kernel instantiation OUTSIDE the timed loop (JIT would
    # otherwise dominate the first timed variant)
    launch_rowtopk_merge!(D_val, D_loc, scratch, 0, k; threads = 128)
    for segs in (2, 4, 8)
        launch_seg_rowtopk_merge!(D_val, D_loc, scratch, 0, k; segs)
    end
    tt0 = CUDA.@elapsed CUDA.@sync for _ in 1:iters
        launch_rowtopk_merge!(D_val, D_loc, scratch, 0, k; threads = 128)
    end
    @printf("  per-chunk rowtopk 1-thread/row (old kernel, threads=128): %.3f ms (%.0f GiB/s of C read)\n",
            tt0 / iters * 1e3, rows * w * 2 / (tt0 / iters) / 2^30)
    for segs in (2, 4, 8)   # 12+ needs > 48 KiB shared memory (K=20, fp16)
        tt = CUDA.@elapsed CUDA.@sync for _ in 1:iters
            launch_seg_rowtopk_merge!(D_val, D_loc, scratch, 0, k; segs)
        end
        @printf("  per-chunk seg-rowtopk segs=%2d: %.3f ms (%.0f GiB/s of C read)  ->  %.1f ms per %d-chunk batch\n",
                segs, tt / iters * 1e3, rows * w * 2 / (tt / iters) / 2^30, tt / iters * 1e3 * cld(N, w), cld(N, w))
    end

    # ---- chunk-width sweep (compute-only, single runs) -----------------------
    # (w trades C-buffer size against per-chunk launch/event overhead: at
    # M = 2^13 the chunk GEMM is sub-millisecond, so fewer, fatter chunks win)
    for w2 in (2^12, 2^14, 2^15)
        eng_w = TopKEngine(B; k, w = w2, rows_cap = rows)
        CUDA.device_synchronize(); t0 = time()
        n = 0
        for bat in topk_flow(src, eng_w)
            n += size(bat.vals, 1)
        end
        tw = time() - t0
        @printf("  w=2^%2d (%3d chunks/batch, %4.0f MiB/buffer): %6.2f s  =>  %.1f TFLOPS\n",
                log2(w2), cld(N, w2), rows * w2 * 2 / 2^20, tw, flops_total / tw / 1e12)
    end

    println("  (TFLOPS = 2*M*rdim*N over the whole big file; the rope stage costs")
    println("   ~0.5 s of the end-to-end time and hides behind the compute stage;")
    println("   rowtopk at M=2^13 launches far fewer blocks than gemmtopk's M=2^17 --")
    println("   the thread sweep above quantifies how exposed the top-k is)")
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)

    GC.gc(); CUDA.reclaim()
    return nothing
end

# ==============================================================================

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, score the top-k against the provenance
# (from legacy e2ehuman16.jl -- the fp16 variant; NOT co-includable names apply)
# ------------------------------------------------------------------------------

"""
    map_and_count(re, reads_file, B, db_heads, db_starts, dbk; ktop, w,
                  batch_size, segs, normalize, progress)
        -> (total, correct, rank_hist, seconds)

Run the full fp16 flow (`rope_topk_stream`: fasta -> rope fp16 -> GEMM +
per-row top-k) over `reads_file` and score every row against its provenance
header: CORRECTLY MAPPED iff the true window [start, start+len-1] in record
src intersects at least one of the ktop returned database windows
(db_heads[c], db_starts[c], length dbk).
"""
function map_and_count(re::RopeEncoder, reads_file::String, B::AbstractMatrix{Float16},
                       db_heads::Vector{String}, db_starts::Vector{Int}, dbk::Int;
                       ktop::Int = 20, w::Int = 2^15, batch_size::Int = 2^13,
                       segs::Int = 8, normalize::Int = 0,
                       progress::Bool = false,
                       stream::Function = (re, f, B; kw...) ->
                                           rope_topk_stream(re, f, B; kw...))
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(B, 2)
    # per-column record id: intern the handful of unique record headers once
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
    total = correct = nb_bat = 0
    rank_hist = zeros(Int, ktop)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in stream(re, reads_file, B; k = ktop, w,
                                   batch_size, rows_cap = batch_size,
                                   normalize, segs, progress,
                                   err_out = err, tasks_out = tks)
        nb_bat += 1
        nb = size(bat.vals, 1)
        for r in 1:nb
            h = parse_read_head(bat.heads[r])
            h === nothing &&
                error("unreadable provenance header: $(bat.heads[r])")
            h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rid = get(rec_id_acc, h.src, Int32(0)) # provenance names by accession
            hit = 0
            @inbounds for j in 1:ktop
                c = Int(bat.locs[r, j])
                # read window [h.start, h.start+h.len-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid &&
                   h.start <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= h.start + h.len - 1
                    hit = j
                    break
                end
            end
            total += 1
            if hit > 0
                correct += 1
                rank_hist[hit] += 1
            end
        end
        @printf("  batch %2d: +%d reads, %d/%d correctly mapped so far\n",
                nb_bat, nb, correct, total)
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    return (total = total, correct = correct, rank_hist = rank_hist, seconds = t)
end

# ------------------------------------------------------------------------------
# Tiny self-test fixture (shared with e2ehuman.jl)
# ------------------------------------------------------------------------------


if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data3()
    mode == "test" && run_flow_topk_test()
    mode == "bench" && bench_flow_topk()
    mode == "all" && (ensure_data3(); run_flow_topk_test(); bench_flow_topk())
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
