# ==============================================================================
# fp8_convert.jl — fp8 (e4m3) type alias + GPU conversion/quantization utils.
#
# PURPOSE
#   `F8` (DLFP8Types.Float8_E4M3FN) plus the on-GPU fp8 <-> fp32/fp16 helpers
#   shared by the GEMM engines (fp8_ptx.jl, fp8_lt.jl) and the top-k kernels
#   (topk_kernels.jl): chunked standard-normal data generation (optionally
#   row-/column-normalized), chunked e4m3 -> fp16 conversion, and the raw
#   e4m3 <-> fp32 converter launches.
#
# SOURCES
#   legacy/test/gemmtopkfp8.jl: F8, byteptr, randn_fp8!, randn_fp8_normrows!,
#   randn_fp8_normcols!, fp16_from_fp8!, f32_to_f8!, f8_to_f32! (bodies
#   verbatim).
#
# DEPS
#   using CUDA, DLFP8Types (Float8_E4M3FN), Random.
#
# NOTES
#   Included by fp8_ptx.jl, fp8_lt.jl and topk_kernels.jl, so `const F8` may
#   be evaluated several times per session (same-value const redeclaration is
#   a no-op).  f32_to_f8!/f8_to_f32! launch the converter kernels registered
#   in `k_conv` by fp8_ptx.jl's build_kernel() — they only work after that;
#   everything else in this file is self-contained.
# ==============================================================================

using CUDA
using DLFP8Types: Float8_E4M3FN
using Random

const F8 = Float8_E4M3FN

# byte address of a GPU array's base plus an element offset, as CuPtr{UInt8}.
# Used instead of pointer(::SubArray) so every C++ launch takes a raw parent
# pointer + explicit element offset (no view-pointer semantics to trust).
@inline byteptr(p::CuPtr{T}, off::Integer = 0) where {T} =
    reinterpret(CuPtr{UInt8}, reinterpret(UInt64, p) + UInt(off) * sizeof(T))

# --------------------------------------------------------------------------
# Data generation / conversion (chunked, all on GPU)
# --------------------------------------------------------------------------

# fills an fp8 matrix with standard normals (fp32 randn → e4m3 satfinite,
# converted by the CUDA kernel; DLFP8Types' host conversions are not assumed
# to compile into GPU broadcast kernels)
function randn_fp8!(X::CuMatrix{F8}; chunk::Integer = 2^12)
    M, N = size(X)
    tmp = CuMatrix{Float32}(undef, M, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(tmp)
        f32_to_f8!(byteptr(pointer(X), (lo - 1) * M), pointer(tmp), M * wi)
    end
    CUDA.unsafe_free!(tmp)
    return X
end

# normalized variants: rows of A / columns of B are scaled to unit L2 norm in
# fp32 BEFORE e4m3 quantization, so C = A·B holds cosine similarities
# (|C| ≤ 1) — the production-like regime. Column chunks are self-contained
# for B; row normalization of A needs the full fp32 matrix (1 GiB).
function randn_fp8_normrows!(A::CuMatrix{F8})
    M, K = size(A)
    T = CuMatrix{Float32}(undef, M, K)
    randn!(T)
    T ./= sqrt.(sum(abs2, T; dims = 2))
    f32_to_f8!(byteptr(pointer(A)), pointer(T), M * K)
    CUDA.unsafe_free!(T)
    return A
end

function randn_fp8_normcols!(B::CuMatrix{F8}; chunk::Integer = 2^12)
    K, N = size(B)
    T = CuMatrix{Float32}(undef, K, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        randn!(T)
        T ./= sqrt.(sum(abs2, T; dims = 1))   # per-column norms
        f32_to_f8!(byteptr(pointer(B), (lo - 1) * K), pointer(T), K * wi)
    end
    CUDA.unsafe_free!(T)
    return B
end

# chunked e4m3 → fp16 conversion (via exact fp32), for the fp16 baseline
function fp16_from_fp8!(X16::CuMatrix{Float16}, X8::CuMatrix{F8}; chunk::Integer = 2^12)
    M, N = size(X8)
    tmp = CuMatrix{Float32}(undef, M, min(chunk, N))
    for lo in 1:chunk:N
        wi = min(N, lo + chunk - 1) - lo + 1
        f8_to_f32!(pointer(tmp), byteptr(pointer(X8), (lo - 1) * M), M * wi)
        broadcast!(Float16, @view(X16[:, lo:lo+wi-1]), @view(tmp[:, 1:wi]))
    end
    CUDA.unsafe_free!(tmp)
    return X16
end

# (dst/src pointers arrive via byteptr, so both are CuPtr{UInt8} — only the
# element offset encodes the type; the sig tuple fixes the real pointer types)
function f32_to_f8!(dst::CuPtr, src::CuPtr, n::Integer)
    CUDA.cudacall(k_conv[:f32_f8], (CuPtr{Float32}, CuPtr{UInt8}, Int32),
                  src, dst, Int32(n); blocks = cld(n, 256), threads = 256)
end

function f8_to_f32!(dst::CuPtr, src::CuPtr, n::Integer)
    CUDA.cudacall(k_conv[:f8_f32], (CuPtr{UInt8}, CuPtr{Float32}, Int32),
                  src, dst, Int32(n); blocks = cld(n, 256), threads = 256)
end
