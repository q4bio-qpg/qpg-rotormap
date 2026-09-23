# ==============================================================================
# engine_compact16.jl -- the COMPACT ring flow in FP16: search/engine_compact.jl
# (the host-RAM database + triple-buffered device chunk ring) ported from the
# fp8 stack to the fp16 cuBLAS stack (search/engine_fp16.jl + gemm/fp16.jl).
# For GPUs WITHOUT fp8 tensor cores (cc < 8.9): identical auto-residency
# semantics and bitwise-stable results, fp16 numbers.
#
# WHY A SEPARATE FILE: engine_fp8.jl and engine_fp16.jl are not co-includable
# (both define TopKEngine/topk_flow/rope_topk_stream/map_and_count), and the
# fp8 CompactEngine's inner engine IS the fp8 TopKEngine.  This file therefore
# mirrors engine_compact.jl against the fp16 engine instead of parametrizing
# it.  REQUIRES (include beforehand): search/engine_fp16.jl (+ gemm/fp16.jl).
# NOT co-includable with engine_fp8.jl / engine_complex.jl / engine_compact.jl.
#
# DIFFERENCES vs the fp8 compact engine (everything else is the same
# mechanism: host database, depth-slot device ring, dedicated H2D stream,
# GPU-side event waits, resident delegation on auto):
#   * the host database is Matrix{Float16} (`index_database_host_f16`: an
#     fp8-layout .bin is WIDENED on the host -- 2 B/elem --, an fp16-row one
#     is transposed verbatim);
#   * the inner TopKEngine is engine_fp16's (no a8/a32 quantization staging,
#     no cout, no :mma engine -- cuBLAS fp16 `mul!` chunks only);
#   * batches are uploaded with the fp16 engine's `upload_batch` (no e4m3
#     quantize step) and the chunk GEMM is `mul!` into (nb, wi) C views --
#     NO rows_cap padding (the fp16 resident loop computes the actual batch
#     height), so D_val/D_loc are (nb, k) and D_val is Float16, not Float32;
#   * the auto-residency working-set model drops the fp8 staging terms
#     (`_resident_need16`: host bytes + the two fp16 C chunk buffers +
#     the same 96 MiB slack).
# ==============================================================================

using CUDA

"""
    index_database_host_f16(d) -> Matrix{Float16}

The (rdim, 2*n_frag) HOST fp16 database for the fp16 compact ring, from a
loaded index NamedTuple: an fp8-layout save (`indexreal.fp8.v1` or the legacy
`indexflowreal.fp8.v1`) is WIDENED to fp16 on the host (2 B/elem); a classic
fp16-row save (`indexreal.v1`) is transposed verbatim (fp32-row saves are
converted).  The fp8-layout widening carries the save-time quantization
noise -- for the no-quantization reference path save with INDEX_FP8=0.
"""
function index_database_host_f16(d)
    if get(d, :format, nothing) == E2H_FP8_FORMAT ||
       get(d, :format, nothing) == E2H_LEGACY_FP8_FORMAT
        return Float16.(Float32.(d.embeds8)) # host widen (DLFP8Types host conversions)
    end
    Bh = permutedims(d.embeds) # (2*n_frag, rdim) rows -> (rdim, N) columns
    return d.fp16 ? Bh : Float16.(Bh)
end

# ------------------------------------------------------------------------------
# Auto residency (the fp16 working-set model)
# ------------------------------------------------------------------------------

# Resident-path device working set beyond B itself: the two fp16 C chunk
# buffers (rows_cap x w, 2 B/elem) + per-batch temps (the fp16 upload staging
# ~nb*rdim*2 B, D_val/D_loc, the D2H temps -- a few tens of MiB) and
# context/pool/allocator slack.  The fp16 engine has no a8/a32 staging.
_resident_need16(host::Matrix{Float16}; rows_cap::Int, w::Int) =
    sizeof(host) +                    # B verbatim, as uploaded
    2 * rows_cap * w * 2 +            # buf1 + buf2 (fp16 C chunks)
    96 * 2^20                         # temps + context + pool slack

"""
    resident_fits16(host; rows_cap, w) -> Bool

The auto-residency policy of `rope_topk_stream_compact16` (`resident =
nothing`): true when the fp16 `host` database plus the fp16 engine's working
set fits in the device memory free right now -- i.e. when the compact flow
will upload the database ONCE and run the resident engine instead of the
chunk ring.  Conservative by construction (see the fp8 `resident_fits` for
the full rationale: `CUDA.reclaim()` first, driver's free-VRAM reading,
underestimate would OOM while the ring is always safe).
"""
function resident_fits16(host::Matrix{Float16}; rows_cap::Int, w::Int)
    CUDA.reclaim() # judge free VRAM, not the pool's cached blocks
    return _resident_need16(host; rows_cap, w) <= CUDA.free_memory()
end

# ------------------------------------------------------------------------------
# The compact engine: host database + device chunk ring + streamed sweep
# ------------------------------------------------------------------------------

"""
Per-batch fp16 GEMM+top-k state for a HOST database: the resident fp16
`TopKEngine` (its B field is ring slot 1 -- shape only), the host matrix, a
depth-slot device ring of (rdim, w) fp16 chunk buffers, a dedicated
non-blocking H2D stream and per-slot event pairs.  `depth >= 2`; 3 keeps the
pipe full when H2D and compute are near-balanced.
"""
struct CompactEngine16
    eng::TopKEngine{Float16}       # buf1/buf2 C bufs, sg/st, evg/evt
    host::Matrix{Float16}          # (rdim, N) the whole database, host RAM
    depth::Int                     # device ring depth (slots)
    dev::Vector{CuMatrix{Float16}} # depth x (rdim, w) chunk ring
    h2d::CuStream                  # chunk H2D stream
    ev_h2d::Vector{CuEvent}        # ev_h2d[b]: the copy into slot b is done
    ev_dev::Vector{CuEvent}        # ev_dev[b]: the GEMM reading slot b is done
end

function CompactEngine16(host::Matrix{Float16}; k::Int = 20, w::Int = 2^15,
                         rows_cap::Int = 2^13, segs::Int = 8, depth::Int = 3)
    rdim, N = size(host)
    1 <= k <= w <= N ||
        throw(ArgumentError("need 1 <= k <= w <= N (got k = $k, w = $w, N = $N)"))
    depth >= 2 || throw(ArgumentError("ring depth must be >= 2 (got $depth)"))
    dev = [CuMatrix{Float16}(undef, rdim, w) for _ in 1:depth]
    # the inner engine's B is ring slot 1: a real (rdim, w) fp16 matrix that
    # satisfies every constructor assert; only size(eng.B, 1) is ever read
    eng = TopKEngine(dev[1]; k, w, rows_cap, segs)
    h2d = CuStream(; flags = CUDA.STREAM_NON_BLOCKING)
    CompactEngine16(eng, host, depth, dev, h2d,
                    [CuEvent() for _ in 1:depth], [CuEvent() for _ in 1:depth])
end

"""
    batch_gemm_topk_compact16!(D_val, D_loc, ceng, A) -> (D_val, D_loc)

The fp16 `batch_gemm_topk!` with the database streamed from host RAM: same
contract (running top-k reset first, `A` the batch's device embeddings
(nb, rdim), D_val/D_loc (nb, k) fp16/Int32, returns device-synchronized) --
plus the per-chunk H2D stage on `ceng.h2d` and the ring's event bookkeeping
(slot b is refilled only after the GEMM `depth` chunks back has consumed it;
every wait is GPU-side, the host loop only enqueues).  No rows_cap padding:
the chunk GEMM computes the actual batch height (fp16 cuBLAS needs no
alignment), so D carries exactly `size(A, 1)` rows.
"""
function batch_gemm_topk_compact16!(D_val::CuMatrix{Float16}, D_loc::CuMatrix{Int32},
                                    ceng::CompactEngine16, A::CuMatrix{Float16})
    eng = ceng.eng
    M, KA = size(A)
    rdim, N = size(ceng.host)
    @assert KA == rdim "A's inner dim must match B's rows"
    @assert M <= size(eng.buf1, 1) "batch height $M > engine rows_cap $(size(eng.buf1, 1))"
    @assert size(D_val) == (M, eng.k) && size(D_loc) == (M, eng.k)

    fill!(D_val, typemin(Float16)) # reset the running top-k; the kernel merges into it
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
        # range -- column-major makes the slice one flat memcpy (pageable
        # H2D ~20 GB/s; see engine_fp16's sized staging and the fp8
        # engine_compact.jl header for the pinning non-option)
        CUDA.stream!(h2d) do
            i > ceng.depth && CUDA.wait(ceng.ev_dev[b]) # slot b free (gemm i-depth)
            copyto!(ceng.dev[b], 1, ceng.host, (lo - 1) * rdim + 1, rdim * wi)
            CUDA.record(ceng.ev_h2d[b])
        end
        Cv = @view cbuf[1:M, 1:wi]
        Bv = @view ceng.dev[b][:, 1:wi]
        CUDA.stream!(sg) do
            CUDA.wait(ceng.ev_h2d[b])        # chunk resident
            i > 2 && CUDA.wait(eng.evt[b2])  # buffer b2's previous top-k finished
            mul!(Cv, A, Bv)                  # cuBLAS fp16 GEMM (tensor cores)
            CUDA.record(eng.evg[b2])
            CUDA.record(ceng.ev_dev[b])      # slot b consumed
        end
        CUDA.stream!(st) do
            CUDA.wait(eng.evg[b2])           # C chunk is ready
            launch_seg_rowtopk_merge!(D_val, D_loc, Cv, lo - 1, eng.k; segs = eng.segs)
            CUDA.record(eng.evt[b2])
        end
    end

    CUDA.device_synchronize()
    return D_val, D_loc
end

"""
    topk_flow_compact16(source, ceng::CompactEngine16; out_cap = 2, err_out, tasks_out)
        -> Channel{TopKBatch{Float16}}

The fp16 `topk_flow` against a CompactEngine16: identical contract (fp16
rope batches in, fresh host TopKBatch matrices out, early-close teardown,
error routing) -- only the per-batch GEMM+top-k call is the chunk-streaming
`batch_gemm_topk_compact16!` (the batch uploaded with the fp16 engine's
`upload_batch` on the GEMM stream).
"""
function topk_flow_compact16(source, ceng::CompactEngine16;
                             out_cap::Int = 2,
                             err_out::Ref{Any} = Ref{Any}(nothing),
                             tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k = ceng.eng.k
    out = Channel{TopKBatch{Float16}}(out_cap)
    t_topk = Threads.@spawn begin
        try
            for bat in source
                emb = bat.embeds
                eltype(emb) == Float16 ||
                    throw(ArgumentError("batch eltype $(eltype(emb)) != Float16 (build the " *
                                        "rope stream with fp16 = true)"))
                nb = size(emb, 1)
                nb == 0 && continue
                nb <= size(ceng.eng.buf1, 1) ||
                    throw(ArgumentError("batch height $nb > engine rows_cap $(size(ceng.eng.buf1, 1))"))
                # upload on the GEMM stream (keeps multi-ms kernel queues OFF
                # the default stream -- the resident flow's note verbatim)
                A_d = upload_batch(ceng.eng, emb)
                D_val = CuMatrix{Float16}(undef, nb, k)
                D_loc = CuMatrix{Int32}(undef, nb, k)
                batch_gemm_topk_compact16!(D_val, D_loc, ceng, A_d)
                vals = Matrix{Float16}(undef, nb, k)
                locs = Matrix{Int32}(undef, nb, k)
                copyto!(vals, D_val) # synchronizing D2H, ~1 MiB at the default config
                copyto!(locs, D_loc)
                A_d = D_val = D_loc = nothing # device pool reuses them next batch
                put!(out, TopKBatch(vals, locs, bat.norms, bat.heads, bat.first))
            end
        catch err
            if _topk_flow_error(err, err_out, out; who = "topk16-compact")
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
    rope_topk_stream_compact16(re, file, Bh; <rope_topk_stream's kwargs>,
                               depth = 3, resident = nothing)
        -> Channel{TopKBatch{Float16}}

The full fp16 flow for a HOST-resident database `Bh` (rdim x N fp16, rdim
must equal the encoder's 2*m*4^c).  `resident` picks the engine:
  `nothing` (default) -- AUTO: when `Bh` + the engine's working set fits in
      the free VRAM (`resident_fits16`), upload `Bh` once and run the
      RESIDENT engine (`topk_flow`): no per-batch PCIe sweep.  Otherwise the
      chunk-streaming ring below.
  `false` -- always the ring: rope-encode the fragments and sweep all N
      columns per read batch through the depth-slot device ring (`--batch`
      fragments share each sweep -- the lever that amortizes the H2D).
  `true` -- force the resident upload (fails fast when it cannot fit).
Same kwargs (plus `resident`/`depth`), output contract and teardown as
`rope_topk_stream`.
"""
function rope_topk_stream_compact16(re::RopeEncoder, file::String, Bh::Matrix{Float16};
                                    k::Int = 20, w::Int = 2^15,
                                    batch_size::Int = 2^13, rows_cap::Int = batch_size,
                                    normalize::Int = 0, parts::Int = Threads.nthreads(),
                                    in_cap::Int = 2, out_cap::Int = 2, topk_out_cap::Int = 2,
                                    segs::Int = 8, depth::Int = 3,
                                    resident::Union{Nothing,Bool} = nothing,
                                    progress::Bool = false,
                                    err_out::Ref{Any} = Ref{Any}(nothing),
                                    tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    rdim = 2 * re.m * 4^re.c
    size(Bh, 1) == rdim ||
        throw(ArgumentError("Bh is $(size(Bh, 1)) x $(size(Bh, 2)); the encoder's real " *
                            "embedding dim is 2*m*4^c = $rdim"))
    if resident === nothing
        resident = resident_fits16(Bh; rows_cap, w)
    end
    if resident # the whole database admits VRAM residency: NO per-batch sweep
        need = _resident_need16(Bh; rows_cap, w)
        free = (CUDA.reclaim(); CUDA.free_memory())
        need <= free ||
            throw(ArgumentError("resident = true but the database + engine working set " *
                                "$(round(need / 2^30; digits = 1)) GiB exceeds the " *
                                "$(round(free / 2^30; digits = 1)) GiB free " *
                                "(use resident = nothing/false)"))
        @info "compact16 flow: the index fits in VRAM -> RESIDENT engine " *
              "(one-time upload, no per-batch sweep)" db_gib =
            round(sizeof(Bh) / 2^30; digits = 2) free_gib = round(free / 2^30; digits = 1)
        eng = TopKEngine(CuMatrix{Float16}(Bh); k, w, rows_cap, segs)
        rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                          fp16 = true, parts, in_cap, out_cap,
                                          progress, err_out, tasks_out)
        return topk_flow(rope_ch, eng; out_cap = topk_out_cap, err_out, tasks_out)
    end
    @info "compact16 flow: the index exceeds free VRAM -> chunk-streaming ring" db_gib =
        round(sizeof(Bh) / 2^30; digits = 2)
    ceng = CompactEngine16(Bh; k, w, rows_cap, segs, depth)
    rope_ch = rope_encode_real_stream(re, file; k = re.k, batch_size, normalize,
                                      fp16 = true, parts, in_cap, out_cap,
                                      progress, err_out, tasks_out)
    return topk_flow_compact16(rope_ch, ceng; out_cap = topk_out_cap, err_out, tasks_out)
end
