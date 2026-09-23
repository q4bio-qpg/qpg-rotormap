# ==============================================================================
# bench_gemm_fp8.jl -- FP8 (e4m3) GEMM benchmarks on the RTX 5090 (sm_120):
# the MERGE of the two legacy fp8 GEMM benchmark scripts.
#
#   1. legacy/test/gemmfp8bench.jl -- the cuTile fp8 GEMM (lab/fp8_cutile.jl's
#      basic `fp8_gemm!`) vs FP16 cuBLAS, square-GEMM size sweep.
#      The legacy file included gemmfp8.jl for `fp8_gemm!` AND its include-time
#      correctness test; the include is retargeted to ../lab/fp8_cutile.jl and
#      the test is now the explicit `correctness_test()` call right below
#      (same behavior: the benchmark only runs after the GEMM checked out).
#      Its top-level driver is preserved as `main_cutile()` (the name
#      `main` belongs to the primary driver below).
#   2. legacy/test/gemmtopkfp8.jl -- the batched fp8 GEMM + per-row top-k
#      driver (PRIMARY): nvcc-compiled mma.sync tensor-core kernel (gemm/
#      fp8_ptx.jl) or raw-ccall cuBLASLt fp8 path (gemm/fp8_lt.jl), the fused
#      rowtopk_merge! (gemm/topk_kernels.jl), engine x cout sweeps, in-process
#      fp16 baseline (--baseline16), exact fp32 reference (--exact),
#      normalization (--normalize), w sweep.  Its benchmark/validation
#      functions and `main` + ARGS dispatch are taken verbatim and remain the
#      file's primary entry point.
#
# SOURCE FILES: legacy/test/gemmfp8bench.jl (bench_size + its driver),
#               legacy/test/gemmtopkfp8.jl (test_fp8_gemm, exact_topk!,
#               spot_check, bench_gemm, bench_topk_threads, main, dispatch).
#               Full MEASURED RESULTS (420-445 TFLOPS best config, fp8-out
#               experiment verdict, w sweep, memory footprint) live in
#               gemmtopkfp8.jl's header.
#
# INCLUDES: common/util.jl + the gemm fp8 set (gemm/fp8_convert.jl,
#   gemm/fp8_ptx.jl, gemm/fp8_lt.jl, gemm/topk_kernels.jl -- providing F8,
#   byteptr, f32_to_f8!/f8_to_f32!, randn_fp8!, fp16_from_fp8!, build_kernel/
#   k_gemm, fp8_gemm_mma!, fp8_gemm_lt!, gemm_topk_fp8!/gemm_topk_fp16!,
#   launch_rowtopk_merge!, test_rowtopk_merge_kernel) + the self-contained
#   lab copy ../lab/fp8_cutile.jl (fp8_gemm!, fp8_gemm_cutile!, F8,
#   correctness_test).
#
# USAGE
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl   # full run + sweeps
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --quick
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --engine=mma
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --baseline16
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --exact
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --normalize
#   julia --project=. -t 16 RotorMap/experiments/bench_gemm_fp8.jl --no-sweep
#   (the cuTile-vs-fp16 square sweep of gemmfp8bench.jl: call main_cutile()
#    interactively, or run lab/fp8_cutile.jl directly)
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

inc("common", "util.jl")
inc("gemm", "fp8_convert.jl")
inc("gemm", "fp8_ptx.jl")
inc("gemm", "fp8_lt.jl")
inc("gemm", "topk_kernels.jl")

# the cuTile fp8 GEMM study (self-contained lab copy of gemmfp8.jl +
# gemmfp8cutile.jl).  The legacy gemmfp8bench.jl included gemmfp8.jl for
# fp8_gemm! -- and for its include-time correctness test; preserved here as
# the explicit call below.
include(joinpath(@__DIR__, "..", "lab", "fp8_cutile.jl"))
correctness_test()

using CUDA
using DLFP8Types: Float8_E4M3FN
using LinearAlgebra
using Statistics
using Printf
using Random

CUDA.allowscalar(false) # the legacy gemmtopkfp8.jl include side effect

# ##############################################################################
# SECTION 1 -- the cuTile square-GEMM benchmark (verbatim from
# legacy/test/gemmfp8bench.jl; `F8` and `fp8_gemm!` come from
# ../lab/fp8_cutile.jl)
# ##############################################################################

function bench_size(M::Int, N::Int, K::Int; tm = 64, tn = 64, tk = 32,
                    warmup = 2, nruns = 5)
    flops = 2.0 * M * N * K

    # --- FP8 path: quantized operands, cuTile kernel with Float32 accum ---
    A8 = CuArray(F8.(randn(Float32, M, K) .* 0.05f0))
    B8 = CuArray(F8.(randn(Float32, K, N) .* 0.02f0))
    C8 = CuArray{Float32}(undef, M, N)
    for _ in 1:warmup
        fp8_gemm!(A8, B8, C8; tm, tn, tk)
    end
    CUDA.synchronize()
    t8 = Inf
    for _ in 1:nruns
        t8 = min(t8, CUDA.@elapsed fp8_gemm!(A8, B8, C8; tm, tn, tk))
    end

    # --- FP16 path: cuBLAS via LinearAlgebra.mul! ---
    A16 = Float16.(Float32.(A8)) .* Float16(20.0f0)   # same values, dequantized
    B16 = Float16.(Float32.(B8)) .* Float16(50.0f0)
    A16 = CuArray(A16); B16 = CuArray(B16)
    C16 = CuArray{Float16}(undef, M, N)
    for _ in 1:warmup
        mul!(C16, A16, B16)
    end
    CUDA.synchronize()
    t16 = Inf
    for _ in 1:nruns
        t16 = min(t16, CUDA.@elapsed mul!(C16, A16, B16))
    end

    CUDA.unsafe_free!(A8); CUDA.unsafe_free!(B8); CUDA.unsafe_free!(C8)
    CUDA.unsafe_free!(A16); CUDA.unsafe_free!(B16); CUDA.unsafe_free!(C16)

    return (fp8 = t8, fp16 = t16, flops)
end

# ##############################################################################
# SECTION 2 -- the fp8 GEMM + top-k benchmark suite (verbatim from
# legacy/test/gemmtopkfp8.jl; the engines/kernels/converters come from the
# gemm/ layers included above)
# ##############################################################################

# GEMM engines vs a dequantized fp64 reference, plus the chunked-ldb path
# (B consumed in column chunks through the base-pointer/ldb interface) which
# must reproduce the whole-matrix GEMM bitwise.
function exact_topk!(D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                     A::CuMatrix{F8}, B::CuMatrix{F8}; k::Integer = 20,
                     wref::Integer = 2^15, rowblk::Integer = 2^13)
    M, K = size(A)
    N = size(B, 2)
    A32 = CuMatrix{Float32}(undef, M, K)
    f8_to_f32!(pointer(A32), byteptr(pointer(A)), M * K)
    Ab = CuMatrix{Float32}(undef, rowblk, K)      # row block, contiguous
    B32 = CuMatrix{Float32}(undef, K, wref)
    Cb = CuMatrix{Float32}(undef, rowblk, wref)
    fill!(D_val, typemin(Float32))
    fill!(D_loc, Int32(0))
    # row blocks: each launch merges into D rows (rb-1)+1 .. rb-1+rowblk
    # (row_offset), so Cb — always dense — is read via its own row indexing.
    @assert M % rowblk == 0 "rowblk must divide M"
    for lo in 1:wref:N
        wi = min(N, lo + wref - 1) - lo + 1
        f8_to_f32!(pointer(B32), byteptr(pointer(B), (lo - 1) * K), K * wi)
        Bv = @view B32[:, 1:wi]
        for rb in 1:rowblk:M
            broadcast!(identity, Ab, A32[rb:rb+rowblk-1, :])
            mul!(Cb, Ab, Bv)                      # fp32 gemm (no TF32 by default)
            launch_rowtopk_merge!(D_val, D_loc, Cb, lo - 1, k; row_offset = rb - 1)
        end
    end
    CUDA.unsafe_free!.((A32, Ab, B32, Cb))
    return D_val, D_loc
end

# end-to-end accuracy: recompute a few random rows of C = A*B exactly in fp32
# (dequantized operands, chunked gemv) and compare top-k. e4m3×e4m3 products
# are exact in fp32 and the mma accumulates in fp32, so `tol` only needs to
# cover the C storage rounding (~1 ulp of |C|) plus accumulation-order noise.
function spot_check(A::CuMatrix{F8}, B::CuMatrix{F8}, D_val::CuMatrix{Float32},
                    D_loc::CuMatrix{Int32}, k::Integer;
                    nrows::Integer = 8, seed::Integer = 42, tol::Real = 1.0)
    Random.seed!(seed)
    M, K = size(A)
    N = size(B, 2)
    wref = 2^15                          # fp32 reference chunks (fits easily)
    a32 = CuVector{Float32}(undef, K)
    B32 = CuMatrix{Float32}(undef, K, wref)
    ref = CuVector{Float32}(undef, N)

    maxvalerr = 0.0
    locbad = 0
    for _ in 1:nrows
        r = rand(1:M)
        copyto!(a32, Float32.(Array(@view A[r, :])))   # dequantize the row (small)
        fill!(ref, 0f0)
        for lo in 1:wref:N
            wi = min(N, lo + wref - 1) - lo + 1
            # dequantize the B chunk on GPU (e4m3 → fp32 is exact)
            f8_to_f32!(byteptr(pointer(B32)), byteptr(pointer(B), (lo - 1) * K), K * wi)
            # ref chunk = B32ᵀ * a32  (fp32 gemv)
            mul!(view(ref, lo:lo+wi-1), transpose(@view B32[:, 1:wi]), a32)
        end
        refh = Array(ref)
        refv = sort(refh; rev = true)[1:k]
        gotv = Array(@view D_val[r, :])
        goti = Array(@view D_loc[r, :])

        @assert all(1 .<= goti .<= N) "row $r: locations out of range"
        @assert issorted(gotv; rev = true) "row $r: values not sorted"
        maxvalerr = max(maxvalerr, maximum(abs.(refv .- gotv)))
        # a location is "wrong" only if the value stored there disagrees with
        # the claimed top-k value by more than the storage tolerance
        locbad += count(abs.(refh[goti] .- gotv) .> tol)
    end
    @assert maxvalerr < tol "top-$k values deviate too much from fp32 reference (max err $maxvalerr)"
    @assert locbad == 0 "$locbad locations are inconsistent with their values"
    return maxvalerr
end

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------

# per-chunk GEMM time for every engine×cout, and the top-k read cost per C
# element type; returns Dict((engine, cout) => seconds/chunk)
function bench_gemm(A::CuMatrix{F8}, B::CuMatrix{F8}, D_val::CuMatrix{Float32},
                    D_loc::CuMatrix{Int32}, k::Integer, w::Integer;
                    engines::Tuple = (:mma, :lt), iters::Integer = 20)
    M, K = size(A)
    flops = 2.0 * M * K * w
    Bp1 = byteptr(pointer(B))
    times = Dict{Tuple{Symbol, Symbol}, Float64}()
    @printf("%-6s %-5s | %10s %11s\n", "engine", "cout", "ms/chunk", "TFLOPS")
    for cout in (:f16, :f32, :f8)
        Tc = cout === :f16 ? Float16 : cout === :f32 ? Float32 : UInt8
        Cv = CuMatrix{Tc}(undef, M, w)
        for eng in engines
            gemmf! = eng === :mma ?
                     (Cbuf, Bp, wi) -> fp8_gemm_mma!(cout, Cbuf, A, Bp, wi, K) :
                     (Cbuf, Bp, wi) -> fp8_gemm_lt!(Cbuf, A, Bp, wi)
            try
                gemmf!(Cv, Bp1, w)                       # warmup
                CUDA.device_synchronize()
                t = CUDA.@elapsed CUDA.@sync for _ in 1:iters
                    gemmf!(Cv, Bp1, w)
                end
                times[(eng, cout)] = t / iters
                @printf("%-6s %-5s | %10.2f %11.1f\n", string(eng), string(cout),
                        t / iters * 1e3, flops / (t / iters) / 1e12)
            catch e
                @warn "fp8 GEMM engine ($eng, $cout) unavailable" e
            end
        end
        # top-k scan cost for this C element type
        launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k)
        CUDA.device_synchronize()
        tt = CUDA.@elapsed CUDA.@sync for _ in 1:iters
            launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k)
        end
        @printf("%-6s %-5s | %10.2f %11s   <- top-k only (%.0f GiB/s of C read)\n",
                "topk", string(cout), tt / iters * 1e3, "", M * w * sizeof(Tc) / (tt / iters) / 2^30)
        CUDA.unsafe_free!(Cv)
    end
    return times
end

function bench_topk_threads(Cv::CuMatrix, D_val::CuMatrix{Float32}, D_loc::CuMatrix{Int32},
                            k::Integer; iters::Integer = 20)
    best = (128, Inf)
    for th in (64, 128, 256, 512)
        launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k; threads = th)
        CUDA.device_synchronize()
        t = CUDA.@elapsed CUDA.@sync for _ in 1:iters
            launch_rowtopk_merge!(D_val, D_loc, Cv, 0, k; threads = th)
        end
        @printf("  top-k threads=%3d: %.3f ms/chunk\n", th, t / iters * 1e3)
        t < best[2] && (best = (th, t / iters))
    end
    return best[1]
end

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

function main(; M::Integer = 2^17, K::Integer = 2^11, N::Integer = 2^21, k::Integer = 20,
              w::Integer = 2^13, sweep::Bool = true, engines::Tuple = (:mma, :lt),
              baseline16::Bool = false, exact::Bool = false, normalize::Bool = false)
    dev = CUDA.device()
    @info @sprintf("GPU: %s, %.1f GiB", CUDA.name(dev), CUDA.totalmem(dev) / 2^30)
    @info @sprintf("A: %d×%d e4m3 (%.0f MiB),  B: %d×%d e4m3 (%.1f GiB),  C would be %.0f GiB (fp32),  k=%d",
                   M, K, M * K / 2^20, K, N, K * N / 2^30, M * N * 4 / 2^30, k)
    flops = 2.0 * M * K * N

    @info "compiling C++ mma kernel (nvcc → PTX → CuModule)..."
    build_kernel()

    test_rowtopk_merge_kernel(k = k)
    test_fp8_gemm(engines = engines)

    @info("generating fp8 data on GPU" * (normalize ? " (rows of A / cols of B normalized → cosine-similarity C)" : ""))
    A = CuMatrix{F8}(undef, M, K)
    B = CuMatrix{F8}(undef, K, N)
    if normalize
        randn_fp8_normrows!(A)
        randn_fp8_normcols!(B)
    else
        randn_fp8!(A)
        randn_fp8!(B)
    end

    D_val = CuMatrix{Float32}(undef, M, k)
    D_loc = CuMatrix{Int32}(undef, M, k)

    @info "warmup (2 chunks per engine×cout)..."
    for (eng, co) in ((:mma, :f16), (:mma, :f32), (:mma, :f8),
                      (:lt, :f16), (:lt, :f32), (:lt, :f8))
        eng in engines || continue
        try
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w, max_chunks = 2, engine = eng, cout = co)
        catch e
            @warn "warmup failed for ($eng, $co)" e
        end
    end

    @info "per-chunk micro-benchmark (w=$w)..."
    tg = bench_gemm(A, B, D_val, D_loc, k, w; engines)
    Cv16 = CuMatrix{Float16}(undef, M, w)
    best_threads = bench_topk_threads(Cv16, D_val, D_loc, k)
    @info @sprintf("top-k thread count: using %d", best_threads)
    CUDA.unsafe_free!(Cv16)

    @info "end-to-end runs (pipelined) per engine×cout..."
    best = ((:mma, :f16), Inf)
    for eng in engines, co in (:f16, :f32, :f8)
        GC.gc(); CUDA.reclaim()
        try
            CUDA.device_synchronize(); t0 = time()
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w, engine = eng, cout = co)
            CUDA.device_synchronize()
            t = time() - t0
            @printf("  %-4s %-4s: %6.2f s  →  %.1f TFLOPS\n", string(eng), string(co), t, flops / t / 1e12)
            t < best[2] && (best = ((eng, co), t))
        catch e
            @warn "full run failed for ($eng, $co)" e
        end
    end
    (engine, cout), t_pipe = best
    @info @sprintf("best config: engine=%s cout=%s (%.2f s, %.1f TFLOPS)",
                   engine, cout, t_pipe, flops / t_pipe / 1e12)
    gemm_floor = get(tg, (engine, cout), NaN) * (N / w)
    @printf("(GEMM-only floor ≈ %.2f s → top-k exposes ≈ %.2f s over the floor)\n",
            gemm_floor, max(0.0, t_pipe - gemm_floor))

    # re-run the best config to have its results in D_val/D_loc
    gemm_topk_fp8!(D_val, D_loc, A, B; k, w, engine = engine, cout = cout)

    @info "full run, sequential (no overlap)..."
    D_val2 = similar(D_val); D_loc2 = similar(D_loc)
    CUDA.device_synchronize(); t0 = time()
    gemm_topk_fp8!(D_val2, D_loc2, A, B; k, w, engine = engine, cout = cout, overlap = false)
    t_seq = time() - t0
    @printf("sequential: %6.2f s  →  %.1f TFLOPS   (overlap speedup %.2fx)\n",
            t_seq, flops / t_seq / 1e12, t_seq / t_pipe)

    @info "checking pipelined == sequential (determinism)..."
    @assert Array(D_val) == Array(D_val2) && Array(D_loc) == Array(D_loc2) "pipelined and sequential results differ"

    @info "spot check vs dequantized fp32 reference (8 random rows)..."
    # tolerance covers C storage rounding: ~fp16 ulp of |C| (:f16), ~1e-4
    # accumulation noise (:f32), or 1 e4m3 ulp of the tail values, up to 16
    # at |C|≈250 (:f8)
    tol = cout === :f16 ? 1.0 : cout === :f32 ? 0.05 : 40.0
    maxerr = spot_check(A, B, D_val, D_loc, k; tol)
    @printf("accuracy: max |top-k value - fp32 reference| = %.4g (tolerance %.2g, cout=%s)\n",
            maxerr, tol, string(cout))

    if cout !== :f8
        # the fp8-out experiment: C quantized to e4m3 in DRAM — halves top-k
        # read traffic; does the scan actually get faster, and what does the
        # ~6% value quantization do to the top-k? cuBLASLt rejects e4m3 D
        # (AlgoGetHeuristic status 15), so both passes run on engine=:mma for
        # a controlled cout comparison.
        @info "fp8-out experiment (engine=:mma, cout=:f8 vs :f16)..."
        D_val8 = CuMatrix{Float32}(undef, M, k)
        D_loc8 = CuMatrix{Int32}(undef, M, k)
        D_valb = CuMatrix{Float32}(undef, M, k)
        D_locb = CuMatrix{Int32}(undef, M, k)
        try
            for (co, Dv_, Dl_) in ((:f16, D_valb, D_locb), (:f8, D_val8, D_loc8))
                CUDA.device_synchronize(); t0 = time()
                gemm_topk_fp8!(Dv_, Dl_, A, B; k, w, engine = :mma, cout = co)
                t_ = time() - t0
                @printf("  %-4s %-4s: %6.2f s  →  %.1f TFLOPS\n", "mma", string(co), t_,
                        flops / t_ / 1e12)
            end
            spot_check(A, B, D_val8, D_loc8, k; tol = 40.0)
            v8 = Array(D_val8); l8 = Array(D_loc8)
            vb = Array(D_valb); lb = Array(D_locb)
            @printf("fp8-out vs f16-out: max |Δvalue| = %.2f, %d/%d locations differ (%.1f%% agree)\n",
                    maximum(abs.(v8 .- vb)), count(l8 .!= lb), length(l8),
                    100 * count(l8 .== lb) / length(lb))
            CUDA.unsafe_free!.((D_val8, D_loc8, D_valb, D_locb))
        catch e
            @warn "fp8-out run failed" e
        end
    end

    if exact
        @info "exact fp32 reference: full dequantized sgemm + same top-k kernel..."
        D_ref_val = CuMatrix{Float32}(undef, M, k)
        D_ref_loc = CuMatrix{Int32}(undef, M, k)
        CUDA.device_synchronize(); t0 = time()
        exact_topk!(D_ref_val, D_ref_loc, A, B; k)
        @printf("  exact reference: %.1f s\n", time() - t0)
        v = Array(D_val); l = Array(D_loc)
        ve = Array(D_ref_val); le = Array(D_ref_loc)
        overlap = [length(intersect(@view(l[r, :]), @view(le[r, :]))) for r in 1:M]
        @printf("location agreement vs exact top-%d: %.2f%% of entries (%.2f%% of rows perfect, %.2f%% with ≥%d/%d)\n",
                k, 100 * sum(overlap) / (M * k), 100 * count(==(k), overlap) / M,
                100 * count(>=(k - 2), overlap) / M, k - 2, k)
        @printf("value agreement: max rank-matched |Δ| = %.3f (order-stat gaps when locations swap; ulp-level otherwise)\n",
                maximum(abs.(v .- ve)))
        @printf("k-th value: max |Δ| = %.3f, mean |Δ| = %.4f\n",
                maximum(abs.(v[:, end] .- ve[:, end])), mean(abs.(v[:, end] .- ve[:, end])))
        CUDA.unsafe_free!.((D_ref_val, D_ref_loc))
    end

    if sweep
        @info "chunk width sweep (pipelined, full runs, best config)..."
        for w2 in (2^11, 2^12, 2^14)
            GC.gc(); CUDA.reclaim()
            CUDA.device_synchronize(); t0 = time()
            gemm_topk_fp8!(D_val, D_loc, A, B; k, w = w2, engine = engine, cout = cout)
            tw = time() - t0
            @printf("  w=2^%2d (%2d chunks, %4.0f MiB/buffer): %6.2f s  →  %.1f TFLOPS\n",
                    Int(log2(w2)), cld(N, w2), M * w2 * (cout === :f16 ? 2 : cout === :f32 ? 4 : 1) / 2^20,
                    tw, flops / tw / 1e12)
        end
    end

    if baseline16
        @info "in-process fp16 baseline (cuBLAS mul! GEMM, same top-k kernel)..."
        A16 = CuMatrix{Float16}(undef, M, K); fp16_from_fp8!(A16, A)
        B16 = CuMatrix{Float16}(undef, K, N); fp16_from_fp8!(B16, B)
        GC.gc(); CUDA.reclaim()
        D_val16 = CuMatrix{Float32}(undef, M, k)
        D_loc16 = CuMatrix{Int32}(undef, M, k)
        CUDA.device_synchronize(); t0 = time()
        gemm_topk_fp16!(D_val16, D_loc16, A16, B16; k, w)
        t16 = time() - t0
        @printf("fp16     : %6.2f s  →  %.1f TFLOPS   (fp8 speedup %.2fx)\n",
                t16, flops / t16 / 1e12, t16 / t_pipe)
        v8 = Array(D_val); l8 = Array(D_loc)
        v16 = Array(D_val16); l16 = Array(D_loc16)
        @printf("agreement: max |D_val8 - D_val16| = %.3f, %d/%d top-k locations differ\n",
                maximum(abs.(v8 .- v16)), count(l8 .!= l16), length(l8))
        CUDA.unsafe_free!.((A16, B16, D_val16, D_loc16))
    end

    GC.gc(); CUDA.reclaim()
    return D_val, D_loc
end

# ##############################################################################
# SECTION 3 -- main_cutile: the gemmfp8bench.jl top-level driver (verbatim
# body, renamed; `main` is the primary driver above)
# ##############################################################################

"""
    main_cutile()

The legacy gemmfp8bench.jl driver: cuTile fp8 (`fp8_gemm!`, the basic
lab kernel) vs FP16 cuBLAS on square GEMMs -- sizes from ARGS (default
512 1024 2048 4096 8192).
"""
function main_cutile()
    sizes = isempty(ARGS) ? [512, 1024, 2048, 4096, 8192] : parse.(Int, ARGS)

    println("RTX 5090 — FP8 (cuTile, e4m3) vs FP16 (cuBLAS), square GEMM\n")
    @printf "%6s | %10s %12s | %10s %12s | %8s\n" "N" "fp8 ms" "fp8 TFLOPS" "fp16 ms" "fp16 TFLOPS" "speedup"
    println("-"^72)
    for n in sizes
        (; fp8, fp16, flops) = bench_size(n, n, n)
        @printf "%6d | %10.3f %12.1f | %10.3f %12.1f | %7.2fx\n" n fp8*1e3 flops/fp8/1e12 fp16*1e3 flops/fp16/1e12 fp16/fp8
    end
    return nothing
end

# ==============================================================================
# Mode dispatch -- gemmtopkfp8.jl's bottom-of-file code, verbatim (PRIMARY);
# called under the old PROGRAM_FILE guard
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    quick = "--quick" in ARGS
    sweep = !("--no-sweep" in ARGS)
    base16 = "--baseline16" in ARGS
    exactref = "--exact" in ARGS
    normalize = "--normalize" in ARGS
    engstr = getflag("engine", "both")
    engines = engstr == "mma" ? (:mma,) : engstr == "lt" ? (:lt,) : (:mma, :lt)
    if quick
        main(M = 2^14, K = 2^11, N = 2^16, k = 20, w = 2^11, sweep = false,
             engines = engines, baseline16 = base16, exact = exactref, normalize = normalize)
    else
        main(k = 20, sweep = sweep, engines = engines, baseline16 = base16,
             exact = exactref, normalize = normalize)
    end
end
