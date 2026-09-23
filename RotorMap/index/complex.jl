# =============================================================================
# complex.jl -- the SPLIT complex index format (from legacy e2ecomplex.jl).
# The complex save stores the [Re; Im] row stack SPLIT into two e4m3 column
# matrices (embeds8_re, embeds8_im; D = m*4^c rows each) instead of the real
# flow's single (2D, 2n) stack; the databases are derived on the GPU at load.
#
#   load_index_complex      complex/legacy-fp8 .bin -> NamedTuple (renamed from
#                           the legacy `load_index` overload so both loaders can
#                           coexist in one session)
#   _complex_databases      split host halves -> (Bre, Bim) resident pair
#   index_database_complex  loaded complex index -> (Bre, Bim)  (renamed from the
#                           legacy `index_database` method for the same reason;
#                           the REAL flow's index_database lives in index/load.jl)
#
# CHANGES vs legacy: `Utils.load` -> bare `load` (common/dna.jl); the default
# path used to be the experiments' `E2C_INDEX_BIN` const -- now resolved from
# `nothing` at CALL time (same behavior).
# REQUIRES: common/dna.jl (load), gemm/fp8_convert.jl (F8), index/load.jl
#   (included beforehand in the canonical order).
# =============================================================================

const E2C_FORMAT = "indexflowcomplex.fp8.v1"         # the split-form complex save
const E2C_LEGACY_FP8_FORMAT = "indexflowreal.fp8.v1" # the pre-rename real flow's
                                                     # save: stored Bre verbatim,
                                                     # Bim derived
const E2C_REAL_FP8_FORMAT = "indexreal.fp8.v1"       # the real flow's save since
                                                     # the indexreal rename

"""
    load_index_complex(path = E2C_INDEX_BIN)

Load the complex index .bin (Julia-serialized NamedTuple): either the split
complex format (`format = "indexflowcomplex.fp8.v1"`: embeds8_re / embeds8_im,
each (D, 2*n_frag) Float8_E4M3FN with D = m*4^c -- the row-stacked [Re; Im]
columns split; Bre is derived from them at database-build time) or the legacy
real fp8 save ("indexflowreal.fp8.v1": embeds8 (2D, 2*n_frag) -- Bre is that
matrix verbatim, Bim is derived on the GPU at load).  This flow assumes
(asserts) `normalize = 0`: unit-MODULUS complex columns (|z|^2 = Re^2 + Im^2
= 1) are what makes both databases' columns normalized.
"""
function load_index_complex(path::Union{String,Nothing} = nothing)
    path === nothing && (path = E2C_INDEX_BIN)
    isfile(path) || error("complex index .bin not found: $path (generate with " *
                          "`experiments/e2e_complex.jl save`, or point E2COMPLEX_INDEX at an " *
                          "indexreal.fp8.v1 .bin for the GPU-derived legacy path)")
    d = load(path)
    fmt = get(d, :format, nothing)
    fmt in (E2C_FORMAT, E2C_LEGACY_FP8_FORMAT, E2C_REAL_FP8_FORMAT) ||
        error("not an $E2C_FORMAT/$E2C_REAL_FP8_FORMAT (or legacy " *
              "$E2C_LEGACY_FP8_FORMAT) index: $path")
    d.normalize == 0 ||
        error("e2ecomplex assumes a normalize = 0 index (got $(d.normalize)): " *
              "unit-MODULUS complex columns are what makes Bre/Bim normalized")
    rdim = 2 * d.m * 4^d.c
    D = rdim ÷ 2
    N = 2 * d.n_frag
    if fmt == E2C_FORMAT
        @assert get(d, :complex, false) "complex flag mismatch"
        @assert eltype(d.embeds8_re) == F8 && size(d.embeds8_re) == (D, N) "embeds8_re shape/eltype mismatch"
        @assert eltype(d.embeds8_im) == F8 && size(d.embeds8_im) == (D, N) "embeds8_im shape/eltype mismatch"
    else
        @assert eltype(d.embeds8) == F8 && size(d.embeds8) == (rdim, N) "embeds8 shape/eltype mismatch"
    end
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"
    @info "index loaded" file = basename(path) format = fmt k = d.k kstep = d.kstep n_frag =
        d.n_frag cols = N rdim = rdim encoder = (d.s, d.m, d.c) normalize = d.normalize
    return d
end

"""
    _complex_databases(re_h, im_h) -> (Bre, Bim)

Build both resident GEMM databases from the SPLIT complex columns (host e4m3
matrices (D, N)): Bre = [re; im] and Bim = [im; -re], both (2D, N).  Per
column block: one H2D per half, then the four quarter-writes (the negate is
a fused GPU broadcast; fp8 negation is exact).  Peak device memory: the two
databases + one block of each half.
"""
function _complex_databases(re_h::Matrix{F8}, im_h::Matrix{F8})
    D, N = size(re_h)
    size(im_h) == (D, N) ||
        throw(ArgumentError("split halves must match ($(size(re_h)) vs $(size(im_h)))"))
    Bre = CuMatrix{F8}(undef, 2 * D, N)
    Bim = CuMatrix{F8}(undef, 2 * D, N)
    blk = 2^21 # 2 GiB e4m3 per half per block
    for j0 in 1:blk:N
        j1 = min(j0 + blk - 1, N)
        dre8 = CuArray(@view re_h[:, j0:j1]) # H2D, contiguous column block
        dim8 = CuArray(@view im_h[:, j0:j1])
        @views Bre[1:D, j0:j1] .= dre8
        @views Bre[D+1:2D, j0:j1] .= dim8
        @views Bim[1:D, j0:j1] .= dim8
        @views Bim[D+1:2D, j0:j1] .= .- dre8
    end
    CUDA.synchronize()
    return (Bre, Bim)
end

"""
    index_database_complex(d) -> (Bre, Bim)::Tuple{CuMatrix{F8},CuMatrix{F8}}

The resident fp8 database pair for a loaded index.  For the split-form
complex save this is `_complex_databases` on the stored halves.  For the
legacy real fp8 save ("indexflowreal.fp8.v1") the stored matrix IS Bre --
the real flow's B verbatim (pure H2D) -- and Bim = [Im; -Re] is derived ON
THE GPU per column block (row-half swap + fused negate; bitwise exact).
"""
function index_database_complex(d)
    if get(d, :format, nothing) == E2C_FORMAT
        return _complex_databases(d.embeds8_re, d.embeds8_im)
    end
    Bre = CuArray(d.embeds8) # pure H2D (the real flow's fast path verbatim)
    rdim, N = size(Bre)
    D = rdim ÷ 2
    Bim = CuMatrix{F8}(undef, rdim, N)
    blk = 2^21
    for j0 in 1:blk:N
        j1 = min(j0 + blk - 1, N)
        @views Bim[1:D, j0:j1] .= Bre[D+1:2D, j0:j1]
        @views Bim[D+1:2D, j0:j1] .= .- Bre[1:D, j0:j1]
    end
    CUDA.synchronize()
    return (Bre, Bim)
end
