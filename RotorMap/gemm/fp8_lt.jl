# ==============================================================================
# fp8_lt.jl — the cuBLASLt ccall fp8 GEMM engine.
#
# PURPOSE
#   D = A * B in fp8 (e4m3 inputs, fp32 compute) via cublasLtMatmul, ccalled
#   directly into the toolkit's libcublasLt.  fp8 requires op(A) = A^T /
#   op(B) = B: the col-major M x K A buffer is exactly a row-major K x M
#   matrix (ORDER_ROW); the B chunk is col-major K x wi with ld = K over its
#   base pointer; D is col-major M x wi.  Layouts + heuristic algo are cached
#   per (M, wi, K, Tv).
#
# SOURCES
#   legacy/test/gemmtopkfp8.jl: LT_* consts, lt_lib/lt_handle/lt_opdesc/
#   lt_pref/p_ltMatmul/lt_workspace/lt_cache, checklt, cublaslt_init,
#   lt_entries, fp8_gemm_lt! (bodies verbatim).
#
# DEPS
#   include fp8_convert.jl (F8, byteptr); using CUDA, Libdl.
#
# NOTES
#   D (Cv) may be Float32, Float16 or UInt8 (raw e4m3 bytes) — though the
#   toolkit rejects e4m3 D output for some shapes (heuristic status 15).
#   Needs the matching 595 driver userland on LD_LIBRARY_PATH (the container's
#   stale libcuda otherwise fails with CUDA_ERROR_SYSTEM_DRIVER_MISMATCH).
#   Run cublaslt_init() (or the first fp8_gemm_lt!) once before use.
# ==============================================================================

include(joinpath(@__DIR__, "fp8_convert.jl"))

using CUDA
using Libdl

const LT_R_32F       = Cint(0)    # cudaDataType CUDA_R_32F
const LT_R_16F       = Cint(2)    # cudaDataType CUDA_R_16F
const LT_R_8F_E4M3   = Cint(28)   # cudaDataType CUDA_R_8F_E4M3 (NOT 14 = CUDA_R_16BF!)
const LT_OP_T        = Cint(1)
const LT_OP_N        = Cint(0)
const LT_COMPUTE_32F = Cint(68)   # CUBLAS_COMPUTE_32F
const LT_DESC_TRANSA = Cint(3)
const LT_DESC_TRANSB = Cint(4)
const LT_PREF_MAX_WS = Cint(1)    # CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
const LT_LAYOUT_ORDER = Cint(1)   # CUBLASLT_MATRIX_LAYOUT_ORDER
const LT_ORDER_ROW    = Cint(1)   # CUBLASLT_ORDER_ROW
const LT_ALGO_SZ     = 64         # sizeof(cublasLtMatmulAlgo_t)
const LT_HEUR_SZ     = 96         # sizeof(cublasLtMatmulHeuristicResult_t)

const lt_lib       = Ref{Ptr{Cvoid}}(C_NULL)
const lt_handle    = Ref{Ptr{Cvoid}}(C_NULL)
const lt_opdesc    = Ref{Ptr{Cvoid}}(C_NULL)
const lt_pref      = Ref{Ptr{Cvoid}}(C_NULL)
const p_ltMatmul   = Ref{Ptr{Cvoid}}(C_NULL)
const lt_workspace = Ref{CuVector{UInt8}}()
const lt_cache = Dict{Tuple{Int,Int,Int,Type}, Tuple{Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Vector{UInt8}}}()

checklt(s::Integer, what::AbstractString) = s == 0 || error("cublasLt: $what failed (status $s)")

function cublaslt_init()
    lib = ""
    for dir in ("/usr/local/cuda-13.4", "/usr/local/cuda-13.3")
        p = joinpath(dir, "lib64", "libcublasLt.so.13")
        if isfile(p); lib = p; break; end
    end
    isempty(lib) && error("libcublasLt.so.13 not found in /usr/local/cuda-13.{4,3}")
    lt_lib[] = Libdl.dlopen(lib)
    sym(s) = Libdl.dlsym(lt_lib[], s)

    h = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtCreate), Cint, (Ptr{Ptr{Cvoid}},), h), "cublasLtCreate")
    lt_handle[] = h[]

    op = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtMatmulDescCreate), Cint, (Ptr{Ptr{Cvoid}}, Cint, Cint),
                  op, LT_COMPUTE_32F, LT_R_32F), "MatmulDescCreate")
    ta, tb = LT_OP_T, LT_OP_N
    checklt(ccall(sym(:cublasLtMatmulDescSetAttribute), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                  op[], LT_DESC_TRANSA, Ref(ta), sizeof(Cint)), "Set TRANSA")
    checklt(ccall(sym(:cublasLtMatmulDescSetAttribute), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                  op[], LT_DESC_TRANSB, Ref(tb), sizeof(Cint)), "Set TRANSB")
    lt_opdesc[] = op[]

    pref = Ref{Ptr{Cvoid}}(C_NULL)
    checklt(ccall(sym(:cublasLtMatmulPreferenceCreate), Cint, (Ptr{Ptr{Cvoid}},), pref),
            "PreferenceCreate")
    wsz = Csize_t(32 << 20)
    checklt(ccall(sym(:cublasLtMatmulPreferenceSetAttribute), Cint,
                  (Ptr{Cvoid}, Cint, Ptr{Csize_t}, Csize_t),
                  pref[], LT_PREF_MAX_WS, Ref(wsz), sizeof(Csize_t)), "Pref MAX_WS")
    lt_pref[] = pref[]

    p_ltMatmul[] = sym(:cublasLtMatmul)
    lt_workspace[] = CUDA.zeros(UInt8, 32 << 20)
    return nothing
end

function lt_entries(M::Int, N::Int, K::Int, ::Type{Tv}) where {Tv}
    get!(lt_cache, (M, N, K, Tv)) do
        sym(s) = Libdl.dlsym(lt_lib[], s)
        layout(dt, rows, cols, ld) = begin
            l = Ref{Ptr{Cvoid}}(C_NULL)
            checklt(ccall(sym(:cublasLtMatrixLayoutCreate), Cint,
                          (Ptr{Ptr{Cvoid}}, Cint, Culonglong, Culonglong, Clonglong),
                          l, dt, rows, cols, ld), "LayoutCreate")
            l[]
        end
        d_dt = Tv === Float32 ? LT_R_32F : Tv === Float16 ? LT_R_16F : LT_R_8F_E4M3
        # A: our col-major M x K buffer seen as row-major K x M (op(A) = A^T).
        # (The canonical-looking ORDER_COL instead runs without error but
        # contracts the wrong axis; only randomized checks catch it.)
        la = layout(LT_R_8F_E4M3, K, M, M)
        order = Ref(LT_ORDER_ROW)
        checklt(ccall(sym(:cublasLtMatrixLayoutSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cint}, Csize_t),
                      la, LT_LAYOUT_ORDER, order, sizeof(Cint)), "Layout ORDER")
        lb = layout(LT_R_8F_E4M3, K, N, K)                    # B chunk, ld = parent K
        ld = layout(d_dt, M, N, M)
        buf = Vector{UInt8}(undef, LT_HEUR_SZ * 4)
        nres = Ref{Cint}(0)
        checklt(ccall(sym(:cublasLtMatmulAlgoGetHeuristic), Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{UInt8}, Ptr{Cint}),
                      lt_handle[], lt_opdesc[], la, lb, ld, ld, lt_pref[], 4, buf, nres),
                "AlgoGetHeuristic")
        nres[] >= 1 || error("cublasLt: no fp8 algorithm for ($M, $N, $K, $Tv)")
        (la, lb, ld, copy(buf[1:LT_ALGO_SZ]))
    end
end

function fp8_gemm_lt!(Cv::CuMatrix{Tv}, A::CuMatrix{F8}, Bp::CuPtr{UInt8}, wi::Integer) where {Tv}
    lt_handle[] == C_NULL && cublaslt_init()
    M, K = size(A)
    @assert M % 16 == 0 && K % 16 == 0 "cuBLASLt fp8 needs 16-byte-multiple leading dims"
    la, lb, ld, algo = lt_entries(M, wi, K, Tv)
    alpha, beta = Ref(1.0f0), Ref(0.0f0)
    # ccall's type slot must be a literal tuple, so branch on Tv here; the
    # tuples differ only in the C/D pointer types (slot 9 and 11)
    if Tv === Float32
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{Float32}, Ptr{Cvoid},
                       CuPtr{Float32}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    elseif Tv === Float16
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{Float16}, Ptr{Cvoid},
                       CuPtr{Float16}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    else
        checklt(ccall(p_ltMatmul[], Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{Float32}, CuPtr{UInt8}, Ptr{Cvoid},
                       CuPtr{UInt8}, Ptr{Cvoid}, Ptr{UInt8}, CuPtr{UInt8}, Csize_t, Ptr{Cvoid}),
                      lt_handle[], lt_opdesc[], alpha, byteptr(pointer(A)), la, Bp, lb,
                      beta, pointer(Cv), ld, pointer(Cv), ld,
                      pointer(algo), lt_workspace[], sizeof(lt_workspace[]),
                      CUDA.stream().handle),
                "cublasLtMatmul")
    end
    return Cv
end
