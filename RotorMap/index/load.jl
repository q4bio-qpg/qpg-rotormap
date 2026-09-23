# =============================================================================
using DLFP8Types: Float8_E4M3FN # fp8-format asserts (see REQUIRES above)
# load.jl -- load a saved index .bin and build the resident GPU database
# (UNIFIED: the legacy tree carried three copies -- e2ehuman.jl's handles both
# REAL formats and is kept here as THE load_index; e2ehuman16.jl's fp16-only
# subset was redundant (same checks, fewer formats) and was dropped;
# e2ecomplex.jl's complex-format loader lives in index/complex.jl as
# load_index_complex to avoid same-name redefinition when both files are
# included in one session).
#
#   load_index        "indexreal.v1" fp16-row  OR  "indexreal.fp8.v1"
#                     transposed-e4m3 .bin -> NamedTuple (structural asserts)
#   index_to_f8       fp16/fp32 row matrix -> fp8 GEMM database B (H2D +
#                     transpose + e4m3 quantize, blockwise)
#   index_to_f16      fp16 row matrix -> fp16 GEMM database B (pure transpose)
#   index_database    loaded index -> resident B (fp8 layout: pure H2D fast path)
#
# CHANGES vs legacy: `Utils.load` -> bare `load` (common/dna.jl); the default
# path used to be the experiments' `E2H_INDEX_BIN` const -- a layer file
# cannot reference it at include time, so the default is now `nothing` and
# resolved to `E2H_INDEX_BIN` at CALL time (same behavior, error text
# unchanged for explicit paths).
# REQUIRES: common/dna.jl (load), DLFP8Types (Float8_E4M3FN).  The fp8-format
# asserts use Float8_E4M3FN directly so loading an fp8-layout index does not
# require the gemm layer; index_database/index_to_f8 still produce F8 columns
# (define F8 = Float8_E4M3FN or include gemm/fp8_convert.jl before calling them).
# =============================================================================
using DLFP8Types: Float8_E4M3FN # fp8-format asserts (see REQUIRES above)

const E2H_FORMAT = "indexreal.v1"         # the fp16-row .bin format tag
const E2H_FP8_FORMAT = "indexreal.fp8.v1" # the transposed-e4m3 .bin format tag
# pre-rename saves carry the indexflowreal tags and KEEP LOADING (data compat):
const E2H_LEGACY_FORMAT = "indexflowreal.v1"
const E2H_LEGACY_FP8_FORMAT = "indexflowreal.fp8.v1"
_is_e2h_format(fmt) = fmt == E2H_FORMAT || fmt == E2H_LEGACY_FORMAT
_is_e2h_fp8_format(fmt) = fmt == E2H_FP8_FORMAT || fmt == E2H_LEGACY_FP8_FORMAT

"""
    load_index(path = E2H_INDEX_BIN)

Load an indexreal-format index .bin (Julia-serialized NamedTuple): either
the fp8 mapping layout (`format = "indexreal.fp8.v1"`: embeds8
(rdim, 2*n_frag) Float8_E4M3FN -- the database columns, quantized at save
time) or the classic fp16 row layout (`"indexreal.v1"`: embeds
(2*n_frag, rdim)).  Pre-rename saves with the `indexflowreal.*` tags load
unchanged.  This flow assumes (asserts) `normalize = 0`: unit-energy
rows => normalized database columns.
"""
function load_index(path::Union{String,Nothing} = nothing)
    path === nothing && (path = E2H_INDEX_BIN)
    isfile(path) || error("index .bin not found: $path (generate with " *
                          "index/build.jl `save`)")
    d = load(path)
    fmt = get(d, :format, nothing)
    _is_e2h_format(fmt) || _is_e2h_fp8_format(fmt) ||
        error("not an $E2H_FORMAT/$E2H_FP8_FORMAT index: $path")
    d.normalize == 0 ||
        error("e2ehuman assumes a normalize = 0 index (got $(d.normalize)): " *
              "unit-energy rows are what makes B's columns normalized")
    rdim = 2 * d.m * 4^d.c
    N = 2 * d.n_frag
    if _is_e2h_fp8_format(fmt)
        @assert eltype(d.embeds8) == Float8_E4M3FN "embeds8 eltype mismatch"
        @assert size(d.embeds8) == (rdim, N) "embeds8 shape mismatch"
    else
        @assert eltype(d.embeds) == (d.fp16 ? Float16 : Float32) "embeds eltype mismatch"
        @assert size(d.embeds) == (N, rdim) "embeds shape mismatch"
    end
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"
    @info "index loaded" file = basename(path) format = fmt k = d.k kstep = d.kstep n_frag =
        d.n_frag cols = N rdim encoder = (d.s, d.m, d.c) normalize = d.normalize
    return d
end

"""
    index_to_f8(embeds; block = 2^16) -> B::CuMatrix{F8}

Upload the index matrix (rows = reference-window encodings, fp16/fp32) as the
fp8 GEMM database B (columns = reference windows): per row-block H2D + GPU
transpose + e4m3 quantize, `F8.(Float32.(.))` -- the pipeline's exact
quantization model.  Peak device memory: B + one block; peak host memory: the
index matrix itself (12.7 GiB fp16 for the human genome).  Runs on the
default stream and returns synchronized.
"""
function index_to_f8(embeds::AbstractMatrix{T}; block::Int = 2^16) where {T<:Union{Float16,Float32}}
    N, rdim = size(embeds)
    B = CuMatrix{F8}(undef, rdim, N)
    for j0 in 1:block:N
        j1 = min(j0 + block - 1, N)
        db = CuArray{T}(@view embeds[j0:j1, :]) # (nb, rdim) H2D, contiguous rows
        @views B[:, j0:j1] .= F8.(Float32.(permutedims(db))) # fused transpose+quantize
    end
    CUDA.synchronize()
    return B
end

"""
    index_to_f16(embeds; block = 2^16) -> B::CuMatrix{Float16}

Upload the index matrix (rows = reference-window encodings) as the fp16 GEMM
database B (columns = reference windows): per row-block H2D + GPU transpose.
The fp16 index IS the database -- pure data movement, no quantization (the
fp8 flow quantizes the very same blocks to e4m3).  Peak device memory:
B + one block.  Runs on the default stream, returns synchronized.
(legacy e2ehuman16.jl, verbatim)
"""
function index_to_f16(embeds::AbstractMatrix{Float16}; block::Int = 2^16)
    N, rdim = size(embeds)
    B = CuMatrix{Float16}(undef, rdim, N)
    for j0 in 1:block:N
        j1 = min(j0 + block - 1, N)
        db = CuArray{Float16}(@view embeds[j0:j1, :]) # (nb, rdim) H2D, contiguous rows
        @views B[:, j0:j1] .= permutedims(db) # fused GPU transpose, pure fp16 moves
    end
    CUDA.synchronize()
    return B
end

"""
    index_database_f16(d) -> B::CuMatrix{Float16}

The resident FP16 database for a loaded index (the fp16 engine's B).  For the
classic fp16-row layout ("indexreal.v1") this is `index_to_f16`: blockwise
H2D + GPU transpose.  For the fp8 mapping layout ("indexreal.fp8.v1") the
stored e4m3 columns are widened EXACTLY (e4m3 -> fp16 is bit-exact) blockwise:
H2D of the stored bytes + fused dtype widen -- no requantization, so the fp16
flow sees exactly the database values the fp8 flow does.  Peak device memory:
B + one block.
"""
function index_database_f16(d; block::Int = 2^16)
    if _is_e2h_fp8_format(get(d, :format, nothing))
        rdim, N = size(d.embeds8)
        src = d.embeds8 # @view needs a plain reference expression
        B = CuMatrix{Float16}(undef, rdim, N)
        for j0 in 1:block:N
            j1 = min(j0 + block - 1, N)
            @views B[:, j0:j1] .= Float16.(CuArray(src[:, j0:j1]))
        end
        CUDA.synchronize()
        return B
    end
    return index_to_f16(d.embeds)
end

"""
    index_database(d) -> B::CuMatrix{F8}

The resident fp8 database for a loaded index.  For the fp8 mapping layout
("indexreal.fp8.v1") the stored matrix IS B -- (rdim, 2*n_frag) e4m3
columns, quantized at save time -- so this is a pure H2D with no transpose
and no arithmetic (the fast path: half the bytes of the fp16 rows, which
matters double on a PCIe Gen1 x16 link).  For the classic fp16 row layout
("indexreal.v1") it is built on the GPU blockwise (`index_to_f8`: H2D +
transpose + e4m3) -- bitwise the same bytes, at ~4x the build cost.
"""
function index_database(d)
    if _is_e2h_fp8_format(get(d, :format, nothing))
        return CuArray(d.embeds8)
    end
    return index_to_f8(d.embeds)
end
