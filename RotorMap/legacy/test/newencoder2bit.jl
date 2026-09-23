using Revise

include(joinpath(@__DIR__, "newencoder.jl")) # reuse kernel_best_v2 (and the original) as the reference

using Random
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders

# ==============================================================================
# 2-bit packed input version of `encode_batch_cuda_all_best_v2` (newencoder.jl).
#
# v2 keeps the read bases as one UInt8 (8 bits) per letter: the staged-bases
# region of dynamic shared memory costs maxlen bytes (~20 KB of the ~28 KB budget
# for 20000bp reads) and the hot loop fetches one shared byte per window.
#
# Here the batch is a flat **2-bit packed** bitstream of UInt32 words (base g,
# 0-based absolute, lives in global bits 2g..2g+1); reads are still identified by
# `starts`/`stops` in base units, so a read may start at any bit offset and no
# per-read word padding is needed. Consequences:
#   * the input is 4x smaller (device memory footprint and any host->device
#     transfer);
#   * staging into shared memory moves 4x less data (coalesced UInt32 loads,
#     funnel-shifted once for the sub-word alignment of the read start) and the
#     staged-bases region shrinks 4x (~5 KB instead of ~20 KB for 20000bp reads),
#     raising the shared-memory fast-path threshold from ~40k to ~163k bases per
#     read;
#   * an s-mer is now a contiguous 2s-bit field (requires s <= 16), so a thread
#     keeps its chunk's window in a REGISTER word pair + bit offset (a software
#     funnel shift) and slides it with pure register math; shared memory is only
#     touched once per 16 windows to rotate the word pair, where v2 loaded one
#     shared byte per window;
#   * phases, scramble, shared-memory atomics, G-bin zeroing and normalize modes
#     0-3 replicate v2 exactly.
#
# Endianness: a contiguous bit field extracted from a little-endian bitstream is
# itself little-endian (first base in the low bits), while the original kernel
# builds s-mers big-endian (`smer = (smer << 2) + base`), so the scramble's
# lower/upper split would differ. Instead of a per-window bit-reversal, the
# bases of each read are stored in REVERSE order: the little-endian field of
# reversed window w (1-based) then equals the original big-endian s-mer of the
# original window p = nw - w exactly (verified against v2 and by exhaustive CPU
# simulation). The only kernel change is the phase mapping: window w must get
# ω_x^p = ω_x^(nw-w), so the chunk starts at cis(θ·(nw-start_w)) and steps with
# conj(ω_x) per window.
#
# Because a read can start at any bit offset, the last bitstream word of a read
# can be shared with the first word of the next read; the pack kernel therefore
# ORs its (disjoint) bits in with a global atomic (the output is pre-zeroed).
#
# `pack_reads` packs host-side reads (Vector{Vector{UInt8}}, codes 0:3) into the
# bitstream; `pack_reads_cuda` packs a device-side byte vector (the production
# path: one block per read, reversed byte reads, atomic-OR word writes).
# ==============================================================================

"""
One block per read: packs the read's bytes (codes 0:3) into the 2-bit bitstream
in REVERSE base order (see the endianness note above). Words shared with a
neighbouring read (unaligned lengths) are merged by an atomic OR over the
pre-zeroed output; each read only ever ORs the bits of its own bases.
"""
function kernel_pack_reads(bytes, out, ranges_start, ranges_stop)
    idr = blockIdx().x # one block per read
    idr > length(ranges_start) && return

    rs = Int32(ranges_start[idr])
    L = Int32(ranges_stop[idr]) - rs + Int32(1)
    g0 = rs - Int32(1) # absolute 0-based index of the read's first base
    nwords = cld(L, 16)

    t = threadIdx().x
    B = blockDim().x
    for wo = t:B:nwords # 1-based output word within the read
        acc = UInt32(0)
        # reversed stream position q = 16(wo-1)+j holds the read's base L-1-q
        @inbounds for j in 0:15
            q = 16 * (wo - Int32(1)) + j
            q < L || break
            acc |= UInt32(bytes[g0 + (L - q)] & UInt8(3)) << (2 * j)
        end
        CUDA.@atomic out[(g0 >> 4) + wo] |= acc
    end
    return
end

"""
    pack_reads_cuda(bytes::CuArray{UInt8}, starts, stops) -> CuArray{UInt32}

Pack a device-side flat byte stream (codes 0:3, per-read ranges `starts`/`stops`)
into the 2-bit bitstream, appending one zero guard word (never unmasked into
s-mers, but read by the funnel-shift word pairs).
"""
function pack_reads_cuda(bytes::CuArray{UInt8}, starts::CuArray, stops::CuArray)
    total = Int(maximum(stops)) # reductions are fine, scalar indexing is not
    nwords = ((total - 1) >> 4) + 2 # +1 zero guard word
    out = CUDA.zeros(UInt32, nwords)
    threads = 512
    @cuda blocks = length(starts) threads = threads kernel_pack_reads(bytes, out, starts, stops)
    return out
end

"""Host-side equivalent of `pack_reads_cuda` for `Vector{Vector{UInt8}}` reads."""
function pack_reads(reads)
    starts = Vector{Int32}(undef, length(reads))
    stops = Vector{Int32}(undef, length(reads))
    total = 0
    for (r, read) in enumerate(reads)
        starts[r] = total + 1
        total += length(read)
        stops[r] = total
    end
    words = zeros(UInt32, ((total - 1) >> 4) + 2)
    acc = UInt64(0)
    nbits = 0
    wi = 1
    for read in reads, b in reverse(read) # reversed base order, see the header
        acc |= UInt64(b & 3) << nbits
        nbits += 2
        if nbits == 32
            words[wi] = acc % UInt32
            wi += 1
            acc >>= 32
            nbits = 0
        end
    end
    words[wi] = acc % UInt32 # trailing partial word
    return words, starts, stops
end

"""
Inverse of the packing: flat 2-bit bitstream -> flat byte vector (codes 0:3)
in the original base order (undoing the per-read reversal). Only used on the
fallback path of the v3 wrapper.
"""
function kernel_unpack(dnas, out, ranges_start, ranges_stop)
    idr = blockIdx().x # one block per read
    idr > length(ranges_start) && return

    rs = Int32(ranges_start[idr])
    L = Int32(ranges_stop[idr]) - rs + Int32(1)
    g0 = rs - Int32(1) # absolute 0-based index of the read's first base

    t = threadIdx().x
    B = blockDim().x
    for i = t:B:L # 1-based base position within the read
        g = g0 + (L - i) # absolute 0-based reversed-stream position of base i
        @inbounds out[rs + i - 1] = (dnas[(g >> 4) + 1] >> (UInt32(g & 15) << 1)) & UInt32(3)
    end
    return
end

function unpack_packed_cuda(dnas::CuArray{UInt32}, starts::CuArray, stops::CuArray)
    total = Int(maximum(stops))
    out = CUDA.zeros(UInt8, total)
    threads = 512
    @cuda blocks = length(starts) threads = threads kernel_unpack(dnas, out, starts, stops)
    return out
end

"""
One CUDA block per read, 2-bit packed input version of `kernel_best_v2`.

Dynamic shared memory layout (offsets in bytes):
  [ interleaved (re, im) histogram: 2 * m*4^c Float32 ][ m Float32 norm workspace ][ staged packed bases: cld(L,16)+1 UInt32 ]

The staged words are a funnel-shifted copy of the global bitstream (which
stores each read's bases in reverse order, see the file header), so read base b
(0-based) sits at bit 2b of the staged region regardless of the read's global
bit alignment. Requires s <= 16 (an s-mer must fit in a 32-bit funnel window).
"""
function kernel_best_v3(embeds, embeds_norms, dnas, s, m, c, k, ranges_start, ranges_stop, normalize, maxlen)
    idr = blockIdx().x # one block per read
    if idr > length(ranges_start)
        return
    end

    t = threadIdx().x
    B = blockDim().x

    # 32-bit read offsets
    rs = Int32(ranges_start[idr])
    L = Int32(ranges_stop[idr]) - rs + Int32(1)
    nw = L - Int32(s) + Int32(1) # number of sliding s-mer windows (0 for degenerate reads)
    nbin = m * 4^c # total number of bins over all m copies

    hist = CuDynamicSharedArray(Float32, 2 * nbin)
    sh_norms = CuDynamicSharedArray(Float32, m, 2 * nbin * sizeof(Float32))
    nwords = cld(L, 16) # staged UInt32 words covering the read's bases
    dna_sh = CuDynamicSharedArray(UInt32, nwords + 1, (2 * nbin + m) * sizeof(Float32))

    # stage the read into the shared bitstream (funnel shift for the sub-word
    # alignment of the read start) and zero the histogram
    gw0 = (rs - Int32(1)) >> 4 # global word containing the read's first base (0-based)
    sh0 = UInt32((rs - Int32(1)) & 15) << 1 # bit offset of the read start in that word
    last_word = Int32(length(dnas))
    for wo = t:B:nwords
        G = gw0 + wo - Int32(1) # 0-based global word of this staged word's first base
        if sh0 == 0
            dna_sh[wo] = dnas[G + 1]
        else
            w0 = dnas[G + 1]
            w1 = dnas[min(G + 2, last_word)] # clamp at the end of the stream; the
            # spilled high bits are never unmasked into an s-mer
            dna_sh[wo] = (w0 >> sh0) | (w1 << (UInt32(32) - sh0))
        end
    end
    if t == 1
        dna_sh[nwords + 1] = UInt32(0) # guard word for funnel word pairs at the read end
    end
    for i = t:B:(2*nbin)
        hist[i] = 0f0
    end
    sync_threads()

    bmask = UInt32(4^s - 1) # requires s <= 16
    cbmask = UInt32(4^c - 1)
    mixer = 0x9E3779B1 # based on the golden ratio

    # thread t accumulates a consecutive chunk of REVERSED windows; reversed
    # window w (1-based) corresponds to the original window p = nw - w, so it
    # gets phase ω_x^p = ω_x^(nw-w) and the per-window step is conj(ω_x)
    # (the phase math uses float32 trigonometry on pre-reduced angles, as in v2)
    # the chunk length is made odd so that lanes of a warp, reading the staged
    # words at stride W, touch distinct banks instead of conflicting
    W = cld(nw, B) # chunk length
    iszero(W & 1) && (W += 1)
    for idm = 1:m
        θ = (m == 1 ? 1f0 : Float32(2 * (idm - 1) / (m - 1) + 1)) * Float32(2π / k)
        ω_x = ComplexF32(cos(θ), -sin(θ)) # conj(exp(factor * 2π*im/k)) of the original

        start_w = (t - Int32(1)) * W + Int32(1) # the thread's first (reversed) window
        if start_w <= nw
            # software funnel shift: the staged word pair (w0, w1) + bit offset sh
            # keeps the window's 32-bit neighborhood; the s-mer is v & bmask
            p = start_w - Int32(1) # 0-based base position of the window
            wo = p >> 4 # staged word containing the window start
            sh = UInt32((p & 15) << 1) # bit offset of the window start in that word
            w0 = dna_sh[wo + 1]
            w1 = dna_sh[wo + 2]
            v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))

            a = θ * (nw - start_w) # phase of the thread's first window: ω_x^(nw-start_w)
            ω = ComplexF32(cos(a), sin(a))

            end_w = min(start_w + W - Int32(1), nw)
            for w = start_w:end_w
                smer = v & bmask
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
                        w1 = dna_sh[wo + 2]
                    end
                    v = sh == 0 ? w0 : (w0 >> sh) | (w1 << (UInt32(32) - sh))
                end
            end
        end
    end
    sync_threads()

    begin # ignore G-homopolymer bins (as in the original kernel)
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
    for bin = t:B:nbin
        embeds[bin, 1, idr] = ComplexF32(hist[2*bin-1], hist[2*bin])
    end

    return
end

"""
Drop-in replacement for `encode_batch_cuda_all_best_v2` with 2-bit packed input.
`dnas` is the flat packed bitstream (see `pack_reads` / `pack_reads_cuda`),
`starts`/`stops` are the per-read base ranges (1-based, inclusive), as in v2.
"""
function encode_batch_cuda_all_best_v3(
    re::RopeEncoder,
    dnas::CuArray{UInt32}
    ;
    starts::CuArray,
    stops::CuArray,
    normalize = 1
)
    n = length(starts)

    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, n)
    dest_norms = CUDA.zeros(Float32, re.m, n)

    encode_batch_cuda_all_best_v3!(
        dest,
        dest_norms,
        re,
        dnas
        ;
        starts = starts,
        stops = stops,
        normalize = normalize
    )
    return dest, dest_norms
end

function encode_batch_cuda_all_best_v3!(
    dest::CuArray{ComplexF32},
    dest_norms::CuArray{Float32},
    re::RopeEncoder,
    dnas::CuArray{UInt32}
    ;
    starts::CuArray,
    stops::CuArray,
    normalize = 1
)
    @assert re.s <= 16 "kernel_best_v3 keeps the s-mer in a 32-bit funnel window, so it requires s <= 16"

    n = length(starts)
    nbin = re.m * 4^re.c
    maxlen = Int(maximum(stops .- starts .+ 1))

    shmem_bytes = (2 * nbin + re.m) * sizeof(Float32) + (cld(maxlen, 16) + 1) * sizeof(UInt32)
    if shmem_bytes > 48 * 1024
        # read too long for the shared-memory budget: unpack to bytes and let the
        # v2 wrapper decide between its shared-memory kernel and the original
        bytes = unpack_packed_cuda(dnas, starts, stops)
        return encode_batch_cuda_all_best_v2!(dest, dest_norms, re, bytes; starts = starts, stops = stops, normalize = normalize)
    end

    threads = 512
    blocks = n # one block per read

    # cap registers at 128 = 65536 regs/SM ÷ 512 threads: without the cap the
    # funnel-shift state pushes the kernel to ~140 regs and the launch fails;
    # the cap spills only in the cold normalize section, not in the scan loop
    @cuda blocks = blocks threads = threads shmem = shmem_bytes maxregs = 128 kernel_best_v3(
        dest,
        dest_norms,
        dnas,
        re.s,
        re.m,
        re.c,
        re.k,
        starts,
        stops,
        normalize,
        maxlen
    )
    return dest, dest_norms
end

# ==============================================================================

function check_pair3(re, dnas_packed, dnas_bytes, starts, stops, normalize; rtol = 1e-3, atol = 2e-4)
    # note: the shared-memory atomics make the summation order (and thus the exact
    # float32 rounding) nondeterministic between runs; both kernels use the same
    # float32 phase math, so agreement is limited only by the atomic order
    # (normalize=2 amplifies per-bin rounding for small bins, as in newencoder.jl)
    cropes2, norms2 = encode_batch_cuda_all_best_v2(re, dnas_bytes; starts = starts, stops = stops, normalize = normalize)
    cropes3, norms3 = encode_batch_cuda_all_best_v3(re, dnas_packed; starts = starts, stops = stops, normalize = normalize)

    err = maximum(abs.(cropes2 .- cropes3))
    nerr = maximum(abs.(norms2 .- norms3))
    @info "  normalize=$normalize" max_diff = err norms_max_diff = nerr
    @assert isapprox(cropes2, cropes3; rtol = rtol, atol = atol) "v3 encoding mismatch for normalize=$normalize"
end

function test()
    @show CUDA.name(device())

    n_reads = 2^13
    L = 20_000
    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)

    @info "Generating $n_reads random DNA reads of length $L"
    Random.seed!(1234)
    reads = [rand(UInt8(0):UInt8(3), L) for _ in 1:n_reads]
    @info "Packing the reads into the 2-bit bitstream"
    words, starts_h, stops_h = pack_reads(reads)
    @show (sizeof(words), sum(length, reads)) # 41MB vs 164MB: 4x smaller input
    dnas_packed = cu(words)
    starts = cu(starts_h)
    stops = cu(stops_h)
    jreads = jagged_array(reads)
    dnas_bytes = jreads.vect

    @info "Checking v3 against v2 for all normalize modes"
    for normalize in 0:3
        check_pair3(re, dnas_packed, dnas_bytes, starts, stops, normalize)
    end

    # also check the m=1 special case of the phase formula
    @info "Checking the m=1 configuration (s=5, c=4)"
    re1 = RopeEncoder(k = L, s = 5, m = 1, c = 4)
    for normalize in 0:3
        check_pair3(re1, dnas_packed, dnas_bytes, starts, stops, normalize)
    end

    # the funnel-shift edge case: the s-mer is exactly 32 bits wide
    @info "Checking the s=16 configuration (m=2, c=4)"
    re16 = RopeEncoder(k = L, s = 16, m = 2, c = 4)
    for normalize in 0:3
        check_pair3(re16, dnas_packed, dnas_bytes, starts, stops, normalize)
    end

    # jagged reads of random lengths (the read starts become unaligned to word
    # borders, exercising the funnel-shift staging path)
    @info "Checking jagged reads of random lengths"
    reads_jag = [rand(UInt8(0):UInt8(3), L - rand(0:100)) for _ in 1:n_reads]
    words_jag, starts_jag_h, stops_jag_h = pack_reads(reads_jag)
    dnas_packed_jag = cu(words_jag)
    starts_jag = cu(starts_jag_h)
    stops_jag = cu(stops_jag_h)
    jreads_jag = jagged_array(reads_jag)
    dnas_bytes_jag = jreads_jag.vect
    for normalize in (0, 1)
        check_pair3(re, dnas_packed_jag, dnas_bytes_jag, starts_jag, stops_jag, normalize)
    end

    # timing (all kernels are compiled by now)
    normalize = 0
    @info "Timing the original encode_batch_cuda_all_best"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, dnas_bytes; starts = starts, stops = stops, normalize = normalize)
    end
    @info "Timing encode_batch_cuda_all_best_v2 (UInt8 input)"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best_v2(re, dnas_bytes; starts = starts, stops = stops, normalize = normalize)
    end
    @info "Timing encode_batch_cuda_all_best_v3 (2-bit packed input)"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best_v3(re, dnas_packed; starts = starts, stops = stops, normalize = normalize)
    end
    @info "Timing the device-side pack kernel (one-time preprocessing)"
    dnas_packed2 = pack_reads_cuda(dnas_bytes, starts, stops) # compile first
    CUDA.@time dnas_packed2 = pack_reads_cuda(dnas_bytes, starts, stops)
    @assert all(dnas_packed2 .== dnas_packed) "pack_reads_cuda mismatch"

    # long reads: v2 exceeds its 48KB shared-memory budget and falls back to the
    # original kernel, while v3 still fits (the staged bases are 4x smaller)
    L2 = 150_000
    n2 = 2^11
    @info "Checking + timing $n2 reads of length $L2 (v2 falls back, v3 stays on the fast path)"
    reads2 = [rand(UInt8(0):UInt8(3), L2) for _ in 1:n2]
    words2, starts2_h, stops2_h = pack_reads(reads2)
    dnas_packed2L = cu(words2)
    starts2 = cu(starts2_h)
    stops2 = cu(stops2_h)
    jreads2 = jagged_array(reads2)
    dnas_bytes2 = jreads2.vect
    for normalize in (0, 1)
        check_pair3(re, dnas_packed2L, dnas_bytes2, starts2, stops2, normalize)
    end
    @info "Timing v2 (falls back to the original kernel)"
    CUDA.@time cropes, _ = encode_batch_cuda_all_best_v2(re, dnas_bytes2; starts = starts2, stops = stops2, normalize = 0)
    @info "Timing v3 (2-bit packed input)"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best_v3(re, dnas_packed2L; starts = starts2, stops = stops2, normalize = 0)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
