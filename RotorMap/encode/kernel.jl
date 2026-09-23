# ==============================================================================
# kernel.jl -- the real-output GPU rope fragment kernel + batch encoders.
#
# PURPOSE: rope-encode fixed-length DNA fragments (k bases each, forward 2-bit
#   packed) on the GPU and emit REAL vectors: the complex embedding is
#   emitted as the split [Re; Im] concatenation instead of ComplexF32, plus an
#   fp16 option for the output storage.  Contains the kernel
#   `kernel_rope_frag_real`, the batch encoders `encode_frag_real_batch!` /
#   `encode_frag_real_batch`, and the kernel self-check `_check_frag_kernel_real`.
#
# THE REAL LAYOUT (a FIXED contract for consumers -- never varies per call):
#   a fragment's complex embedding is a D = m*4^c vector over s-mer bins (bin
#   = (csmer-1)*m + idm, the v3 convention); its real form is the 2D-vector
#       [ Re(b_1) .. Re(b_D) , Im(b_1) .. Im(b_D) ]
#   i.e. the SPLIT/planar order: ALL REAL PARTS FIRST, then ALL IMAGINARY
#   PARTS.  Chosen over interleaved because the downstream consumers
#   (SplitComplexMatrix(.Re, .Im), Mapper.jl) want contiguous blocks: row r's
#   Re block is embeds[r, 1:D], its Im block embeds[r, D+1:2D] -- plain
#   contiguous slices, no strides.  Consumer guarantees: the real inner
#   product equals the complex one (<x, y> = Re<z_x, z_y>) and real distance
#   equals complex distance (|x - y|^2 = |z_x - z_y|^2).  (The shared-memory
#   histogram stays interleaved; only the final shared->global copy splits it
#   into the two halves.)
#
# THE FP16 OPTION (T = Float16 on the encoders): compute and the norms stay
#   Float32; only the OUTPUT storage is Float16 -- the conversion happens
#   inside the kernel (no intermediate buffer, one fused pass over shared
#   memory), so the embedding D2H transfer HALVES (~4 KiB/fragment at the
#   default config instead of ~8 KiB).  Norms remain Float32 (m, nb)
#   regardless.  The unit-energy invariant survives to ~1e-2 relative (fp16
#   rounding).
#
# KERNEL SEMANTICS: one block per fragment, 512 threads, dynamic shared memory
#   [histogram | norm workspace | staged words]; staging is a DIRECT copy
#   (each fragment is word-aligned to its own row of the batch, no sub-word
#   alignment, one zero guard word); windows are walked FORWARD (0-based
#   window p covers bases [p, p+s)); the s-mer is the little-endian 2s-bit
#   field `v & bmask`; per-window phasor exp(+i*theta_idm*p) with
#   theta_idm = factor*2pi/k (factor = 1 for m=1, else 2(idm-1)/(m-1)+1,
#   float32 trig on pre-reduced angles, the same convention as
#   kernel_best_v2); scrambled histogram accumulation, G-bin zeroing (the
#   established `2*allones+1` convention) and normalize modes 0-3 replicate
#   kernel_best_v3 verbatim; maxregs = 128 register cap as in kernel_best_v3.
#
# SOURCES: legacy/test/ropeflowreal.jl (kernel_rope_frag_real,
#   encode_frag_real_batch!, encode_frag_real_batch, _check_frag_kernel_real).
#
# DEPS: includes ropeencoder.jl (RopeEncoder) and reference.jl
#   (_frag_codes, _ref_rope_frag_real -- used by the self-check).  A consumer
#   must include ropeencoder.jl and reference.jl first (or simply include
#   this file, which does it).  `using CUDA, Random` required.
#   encoder_v3.jl is NOT needed: the moved bodies reference the v2/v3
#   encoders only in comments, never in code.
#
# NOTES: the complex-output variant (kernel_rope_frag, encode_frag_batch!,
#   encode_frag_batch) is superseded by this real flow and is NOT carried.
# ==============================================================================

include(joinpath(@__DIR__, "ropeencoder.jl"))
include(joinpath(@__DIR__, "reference.jl"))

using CUDA
using Random

# ==============================================================================
# The forward-fragment kernel, real output
# ==============================================================================

"""
Real-output forward-fragment rope kernel -- identical semantics and work
distribution as the (superseded, not carried) complex-output variant; only
the output stage differs:

  * `embeds` holds REAL vectors: an (n, 2*m*4^c) row-major-batch matrix where
    row `idr` is the SPLIT concatenation [Re; Im] of the fragment's complex
    histogram -- `embeds[idr, bin] = Re(bin)`, `embeds[idr, bin + m*4^c] =
    Im(bin)` (coalesced per block in both halves);
  * the output eltype is `T` (Float32, or Float16 with the `fp16` option --
    the conversion from the Float32 shared histogram happens right here in
    the kernel, so no intermediate buffer and a half-size D2H);
  * `embeds_norms[t, idr]` (m x n, Float32 always) is unchanged.

The s-mer fields, scramble key, G-bin convention, phase convention and
normalize modes 0-3 replicate kernel_best_v3 verbatim (see the file header).
"""
function kernel_rope_frag_real(embeds::CuDeviceArray{T}, embeds_norms, dnas,
                               s, m, c, k, W, n, normalize) where {T<:Real}
    idr = blockIdx().x # one block per fragment
    idr > n && return

    t = threadIdx().x
    B = blockDim().x

    nw = Int32(k) - Int32(s) + Int32(1) # number of sliding windows (0: degenerate)
    nbin = m * 4^c

    hist = CuDynamicSharedArray(Float32, 2 * nbin)
    sh_norms = CuDynamicSharedArray(Float32, m, 2 * nbin * sizeof(Float32))
    dna_sh = CuDynamicSharedArray(UInt32, W + 1, (2 * nbin + m) * sizeof(Float32))

    # stage the fragment's words (direct copy: the fragment is self-aligned)
    # and zero the histogram
    g0 = (idr - Int32(1)) * Int32(W) # the fragment's first word in the flat batch
    for wo = t:B:W
        @inbounds dna_sh[wo] = dnas[g0 + wo]
    end
    if t == 1
        @inbounds dna_sh[W+1] = UInt32(0) # guard word for funnel word pairs at
    end                                   # the fragment end (never unmasked)
    for i = t:B:(2*nbin)
        @inbounds hist[i] = 0f0
    end
    sync_threads()

    bmask = UInt32(4^s - 1) # requires s <= 16
    cbmask = UInt32(4^c - 1)
    mixer = 0x9E3779B1 # based on the golden ratio

    # thread t accumulates a consecutive chunk of FORWARD windows; window p
    # (0-based) gets phase exp(+i*theta*p) -- same convention as kernel_best_v2,
    # reached directly (the reversed-stream kernel needed the conj trick).  The
    # chunk length is made odd so that lanes of a warp, reading the staged words
    # at stride W_chunk, touch distinct banks instead of conflicting
    W_chunk = cld(nw, B)
    iszero(W_chunk & 1) && (W_chunk += 1)
    for idm = 1:m
        θ = (m == 1 ? 1f0 : Float32(2 * (idm - 1) / (m - 1) + 1)) * Float32(2π / k)
        ω_x = ComplexF32(cos(θ), sin(θ)) # exp(+i*theta)

        start_w = (t - Int32(1)) * W_chunk + Int32(1) # the thread's first window
        if start_w <= nw
            # software funnel shift over the staged words: the window's 32-bit
            # neighborhood (w0, w1) + bit offset sh; the little-endian s-mer
            # field is v & bmask
            p0 = start_w - Int32(1) # 0-based base position of the window
            wo = p0 >> 4            # staged word containing the window start
            sh = UInt32((p0 & 15) << 1)
            w0 = dna_sh[wo+1]
            w1 = dna_sh[wo+2]
            v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))

            a = θ * (start_w - Int32(1)) # phase of the thread's first window
            ω = ComplexF32(cos(a), sin(a))

            end_w = min(start_w + W_chunk - Int32(1), nw)
            for w = start_w:end_w
                smer = v & bmask # little-endian field of bases [p, p+s)
                lower = smer & cbmask
                upper = smer >> 2c
                scramble_key = (upper * mixer) & cbmask
                csmer = (lower ⊻ scramble_key) + 1

                bin = (csmer - 1) * m + idm # linear index in the (m, 4^c) layout
                CUDA.@atomic hist[2*bin-1] += real(ω)
                CUDA.@atomic hist[2*bin] += imag(ω)

                ω *= ω_x
                if w < end_w
                    sh += UInt32(2)
                    if sh == UInt32(32) # the window start crossed into the next word
                        sh = UInt32(0)
                        w0 = w1
                        wo += Int32(1)
                        w1 = dna_sh[wo+2]
                    end
                    v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))
                end
            end
        end
    end
    sync_threads()

    begin # G-homopolymer bin zeroing, the established `2*allones+1` convention
        if t == 1
            allones = UInt32(0)
            for i = 0:c-1
                allones += UInt32(4^i)
            end
            csmerG = 2 * allones + 1
            for idm = 1:m
                bin = (csmerG - 1) * m + idm
                hist[2*bin-1] = 0f0
                hist[2*bin] = 0f0
            end
        end
    end
    sync_threads()

    if normalize == 0 # one norm shared by all m copies
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                acc += abs2(ComplexF32(hist[2*bin-1], hist[2*bin]))
            end
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc # partial norms, as left by the original
        end
        sync_threads()
        if t == 1
            acc = sh_norms[1]
            for idm = 2:m
                acc += sh_norms[idm]
            end
            acc = sqrt(acc)
            acc += 1e-16
            embeds_norms[1, idr] = acc
            sh_norms[1] = acc
        end
        sync_threads()
        nrm = sh_norms[1]
        for i = t:B:(2*nbin)
            hist[i] /= nrm
        end
    elseif normalize == 1 # per-copy norms
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                acc += abs2(ComplexF32(hist[2*bin-1], hist[2*bin]))
            end
            acc = sqrt(acc)
            acc += 1f-7
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    elseif normalize == 2 # each s-mer treated equally
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                re_im = ComplexF32(hist[2*bin-1], hist[2*bin])
                re_im = ComplexF32(re_im / (abs(re_im) + eps()))
                hist[2*bin-1] = real(re_im)
                hist[2*bin] = imag(re_im)
                acc += abs2(re_im)
            end
            acc = sqrt(acc)
            acc += eps()
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    elseif normalize == 3 # use sqrt
        if t <= m
            acc = 0f0
            for bin = t:m:nbin
                re_im = ComplexF32(hist[2*bin-1], hist[2*bin])
                re_im = ComplexF32(re_im / sqrt(abs(re_im) + eps()))
                hist[2*bin-1] = real(re_im)
                hist[2*bin] = imag(re_im)
                acc += abs2(re_im)
            end
            acc = sqrt(acc)
            acc += eps()
            sh_norms[t] = acc
            embeds_norms[t, idr] = acc
        end
        sync_threads()
        if t <= m
            nrm = sh_norms[t]
            for bin = t:m:nbin
                hist[2*bin-1] /= nrm
                hist[2*bin] /= nrm
            end
        end
    end

    sync_threads()
    # the ONLY semantic change vs the complex-output kernel: the final
    # shared->global copy splits the interleaved histogram into the [Re; Im] halves
    for bin = t:B:nbin
        @inbounds embeds[idr, bin] = T(hist[2*bin-1])         # Re half
        @inbounds embeds[idr, bin+nbin] = T(hist[2*bin])      # Im half
    end

    return
end

"""
Real-output batch encoder: `dnas` is the flat fragment batch (nfrag * W words,
W = cld(k, 16)), `dest` is (nfrag, dim = 2*m*4^c) with eltype Float32 or
Float16 (the fp16 output option -- converted in-kernel), `dest_norms` is
(m, nfrag) Float32.
"""
function encode_frag_real_batch!(
    dest::CuArray{T,2},
    dest_norms::CuArray{Float32,2},
    re::RopeEncoder,
    dnas::CuVector{UInt32};
    normalize::Int = 0
) where {T<:Union{Float16,Float32}}
    @assert re.s <= 16 "kernel_rope_frag_real keeps the s-mer in a 32-bit funnel window, so it requires s <= 16"
    @assert re.c <= re.s "the compact c-mer must not exceed the s-mer (c <= s)"
    n = size(dest, 1)
    @assert size(dest, 2) == 2 * re.m * 4^re.c "dest must be (n, 2*m*4^c) -- the [Re; Im] concatenation"
    @assert size(dest_norms) == (re.m, n)
    W = cld(re.k, 16)
    @assert length(dnas) == n * W "dnas must hold n * cld(k, 16) words"

    nbin = re.m * 4^re.c
    shmem_bytes = (2 * nbin + re.m) * sizeof(Float32) + (W + 1) * sizeof(UInt32)
    @assert shmem_bytes <= 48 * 1024 "fragment too long for the shared-memory fast path (k = $(re.k) needs $(shmem_bytes) B; k <= ~163k fits at m=4, c=4)"

    n == 0 && return dest, dest_norms
    threads = 512
    # register cap as in kernel_best_v3: the funnel-shift state would otherwise
    # push past 65536 regs/SM and the launch would fail
    @cuda blocks = n threads = threads shmem = shmem_bytes maxregs = 128 kernel_rope_frag_real(
        dest, dest_norms, dnas, re.s, re.m, re.c, re.k, W, n, normalize
    )
    return dest, dest_norms
end

"""Allocating variant: `dnas` holds nfrag * cld(k, 16) words -> (embeds, norms)."""
function encode_frag_real_batch(::Type{T}, re::RopeEncoder, dnas::CuVector{UInt32};
                                normalize::Int = 0) where {T<:Union{Float16,Float32}}
    n = length(dnas) ÷ cld(re.k, 16)
    dest = CUDA.zeros(T, n, 2 * re.m * 4^re.c)
    dest_norms = CUDA.zeros(Float32, re.m, n)
    encode_frag_real_batch!(dest, dest_norms, re, dnas; normalize)
    return dest, dest_norms
end

# ==============================================================================
# Kernel self-check against the CPU golden reference (reference.jl).
# ==============================================================================

# GPU batch of random fragments vs the CPU reference (values + norms +, for
# normalize = 0, the unit-energy invariant).  Modes 2/3 divide by |bin| +
# eps, so bins with heavy collision cancellation amplify the float32-vs-
# float64 epsilon difference -- they get a looser tolerance, as they only
# check convention faithfulness, not precision.  Float16 output gets its own
# (still loose) tolerances: fp16 rounding is ~5e-4 relative on top of the
# float32 accumulation, the norms stay Float32 (v3's tolerance).
function _check_frag_kernel_real(re; k = re.k, nfrag = 64, normalize::Int = 0,
                                 T::Type{<:Union{Float16,Float32}} = Float32,
                                 seed = 1234)
    Random.seed!(seed)
    W = cld(k, 16)
    words = Vector{UInt32}(undef, W * nfrag)
    rand!(words)
    if (r = k & 15) > 0 # zero-pad the tail bits, as the fasta fragment emitter does
        mask = (UInt32(1) << (2 * r)) - 1
        for f in 1:nfrag
            words[(f - 1) * W + W] &= mask
        end
    end
    dest, dnorms = encode_frag_real_batch(T, re, cu(words); normalize)
    embeds = Array(dest)
    norms = Array(dnorms)
    D = re.m * 4^re.c
    heavy = normalize in (2, 3)
    rtol = 1e-2
    atol = T == Float16 ? (heavy ? 2e-2 : 2e-3) : (heavy ? 1e-2 : 2e-4)
    ertol = T == Float16 ? 1e-2 : 1e-3  # unit-energy tolerance
    nrmrtol = heavy ? 1e-2 : 1e-3       # norms are Float32 regardless of T
    nrm_rows = normalize == 0 ? (1:1) : (1:re.m) # mode 0: only row 1 is the norm
    for f in 1:nfrag
        codes = _frag_codes(@view(words[(f - 1) * W + 1:f * W]), k)
        href, nrmref = _ref_rope_frag_real(codes, re; normalize)
        @assert isapprox(@view(embeds[f, :]), href; rtol, atol) "real kernel vs CPU reference mismatch (frag $f, normalize=$normalize, T=$T, s=$(re.s), m=$(re.m), c=$(re.c), k=$k)"
        for idm in nrm_rows
            @assert isapprox(norms[idm, f], nrmref[idm]; rtol = nrmrtol) "norms mismatch (frag $f, copy $idm, normalize=$normalize)"
        end
        # split layout: |x|^2 = |Re|^2 + |Im|^2 == the complex unit energy
        if normalize == 0
            e = sum(abs2, @view(embeds[f, 1:D]); init = 0.0) +
                sum(abs2, @view(embeds[f, D+1:2D]); init = 0.0)
            @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (frag $f)"
        else
            # copy idm's bins sit at stride m WITHIN each half (bin =
            # (csmer-1)*m + idm -> positions idm, idm+m, ... in 1:D and D+1:2D)
            for idm in 1:re.m
                e = sum(abs2, @view(embeds[f, idm:re.m:D]); init = 0.0) +
                    sum(abs2, @view(embeds[f, D+idm:re.m:2D]); init = 0.0)
                @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (frag $f, copy $idm)"
            end
        end
    end
    return nothing
end
