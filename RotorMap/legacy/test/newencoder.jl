using Revise

using Random
using LinearAlgebra
using CUDA
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders

# ==============================================================================
# A more efficient version of `encode_batch_cuda_all_best` (mode = :best).
#
# The original kernel assigns one *thread* per read, which forces a scattered
# global read-modify-write `embeds[idm, csmer, idy] += ω_i` for every s-mer
# position (~64 bytes of L2 traffic per position, i.e. >1MB per 20000bp read),
# while the launch configuration covers 128 reads per block only (so for small
# batches most SMs sit idle). The sliding s-mer update is serially dependent,
# which is what prevents using more threads per read in that scheme.
#
# kernel_best_v2 assigns one *block* per read instead:
#   * the read bases are staged into shared memory once (coalesced global reads);
#   * each thread handles a consecutive chunk of windows, so after building the
#     chunk's first s-mer from the shared bases (O(s) work), the s-mer is just
#     slid O(1) per window instead of the fully serial sliding update, letting
#     the windows be parallelized over the threads of the block;
#   * thread t handles the chunk starting at window (t-1)*W+1 (W = ceil(nw/B));
#     window w (1-based) must get phase ω_x^w, so the thread starts from
#     cis(θ*(start_w-1)) and advances by ω_x per window (θ replicates the
#     original ω_x formula, including the m == 1 special case); the phases are
#     computed with float32 trigonometry only, keeping the register footprint
#     low and the occupancy high;
#   * the complex contributions are accumulated into a shared-memory histogram
#     of m*4^c bins via shared-memory atomics, so per read only the final m*4^c
#     values are written to global memory (coalesced): ~8KB per read instead of
#     ~1.3MB of scattered traffic for a 20000bp read;
#   * the G-homopolymer bin is zeroed and the normalize modes 0-3 replicate the
#     original kernel (same epsilons, same dest_norms semantics).
#
# Reads that do not fit into the 48KB dynamic shared memory budget fall back to
# the original implementation.
# ==============================================================================

"""
One CUDA block per read cooperative version of the original `kernel` (mode=:best).

Dynamic shared memory layout (offsets in bytes):
  [ interleaved (re, im) histogram: 2 * m*4^c Float32 ][ m Float32 norm workspace ][ staged bases: maxlen UInt8 ]
"""
function kernel_best_v2(embeds, embeds_norms, dnas, s, m, c, k, ranges_start, ranges_stop, normalize, maxlen)
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
    dna_sh = CuDynamicSharedArray(UInt8, maxlen, (2 * nbin + m) * sizeof(Float32))

    # stage the read bases and zero the histogram
    for i = t:B:L
        dna_sh[i] = dnas[rs + i - 1]
    end
    for i = t:B:(2*nbin)
        hist[i] = 0f0
    end
    sync_threads()

    bmask = UInt32(4^s - 1)
    cbmask = UInt32(4^c - 1)
    mixer = 0x9E3779B1 # based on the golden ratio

    # thread t accumulates a consecutive chunk of windows; window w gets phase ω_x^w
    # (the phase math uses float32 trigonometry on pre-reduced angles; θ ≤ ~1e-3
    # for k=20000, so the float32 rounding stays within ~1e-6 for any window)
    # the chunk length is made odd so that lanes of a warp, reading shared memory at
    # stride W, touch distinct banks instead of conflicting
    W = cld(nw, B) # chunk length
    iszero(W & 1) && (W += 1)
    for idm = 1:m
        # all-float32 phase math; θ = factor*2π/k ≤ ~1e-3 for k=20000, so the
        # float32 rounding of the angle stays within ~1e-6 for any window
        θ = (m == 1 ? 1f0 : Float32(2 * (idm - 1) / (m - 1) + 1)) * Float32(2π / k)
        ω_x = ComplexF32(cos(θ), sin(θ)) # matches exp(factor * 2π*im/k) of the original

        start_w = (t - Int32(1)) * W + Int32(1) # the thread's first window
        if start_w <= nw
            a = θ * (start_w - Int32(1))
            ω = ComplexF32(cos(a), sin(a)) # phase of the thread's first window

            # build the chunk's first s-mer from the staged bases, then just slide it
            smer = UInt32(0)
            @inbounds for j = 0:s-1
                smer = (smer << 2) + dna_sh[start_w+j]
            end

            end_w = min(start_w + W - 1, nw)
            for w = start_w:end_w
                lower = smer & cbmask
                upper = smer >> 2c
                scramble_key = (upper * mixer) & cbmask
                csmer = (lower ⊻ scramble_key) + 1

                bin = (csmer - 1) * m + idm # linear index in the (m, 4^c) layout
                CUDA.@atomic hist[2*bin-1] += real(ω)
                CUDA.@atomic hist[2*bin] += imag(ω)

                ω *= ω_x
                if w < end_w
                    @inbounds smer = ((smer << 2) & bmask) + dna_sh[w+s]
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
Drop-in replacement for `encode_batch_cuda_all_best` (mode=:best), see kernel_best_v2.
"""
function encode_batch_cuda_all_best_v2(
    re::RopeEncoder,
    dnas::CuArray
    ;
    starts::CuArray,
    stops::CuArray,
    normalize = 1
)
    n = length(starts)

    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, n)
    dest_norms = CUDA.zeros(Float32, re.m, n)

    encode_batch_cuda_all_best_v2!(
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

function encode_batch_cuda_all_best_v2!(
    dest::CuArray{ComplexF32},
    dest_norms::CuArray{Float32},
    re::RopeEncoder,
    dnas::CuArray
    ;
    starts::CuArray,
    stops::CuArray,
    normalize = 1
)
    n = length(starts)
    nbin = re.m * 4^re.c
    maxlen = Int(maximum(stops .- starts .+ 1))

    shmem_bytes = (2 * nbin + re.m) * sizeof(Float32) + maxlen
    if shmem_bytes > 48 * 1024
        # read too long for the shared-memory histogram, fall back to the original
        return encode_batch_cuda_all_best!(dest, dest_norms, re, dnas; starts = starts, stops = stops, normalize = normalize)
    end

    threads = 512
    blocks = n # one block per read

    @cuda blocks = blocks threads = threads shmem = shmem_bytes kernel_best_v2(
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

function check_pair(re, dnas, starts, stops, normalize; rtol = 1e-3, atol = 2e-4)
    # note: the shared-memory atomics make the summation order (and thus the exact
    # float32 rounding) nondeterministic between runs; normalize=2 additionally
    # amplifies per-bin rounding for small bins, hence the loose absolute tolerance
    cropes1, norms1 = encode_batch_cuda_all_best(re, dnas; starts = starts, stops = stops, normalize = normalize)
    cropes2, norms2 = encode_batch_cuda_all_best_v2(re, dnas; starts = starts, stops = stops, normalize = normalize)

    err = maximum(abs.(cropes1 .- cropes2))
    nerr = maximum(abs.(norms1 .- norms2))
    @info "  normalize=$normalize" max_diff = err norms_max_diff = nerr
    @assert isapprox(cropes1, cropes2; rtol = rtol, atol = atol) "v2 encoding mismatch for normalize=$normalize"
end

function test()
    @show CUDA.name(device())

    n_reads = 2^13
    L = 20_000
    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)

    @info "Generating $n_reads random DNA reads of length $L"
    Random.seed!(1234)
    reads = [rand(UInt8(0):UInt8(3), L) for _ in 1:n_reads]
    jreads = jagged_array(reads)
    dnas = jreads.vect
    starts = cu(jreads.starts)
    stops = cu(jreads.stops)

    @info "Checking v2 against the original for all normalize modes"
    for normalize in 0:3
        check_pair(re, dnas, starts, stops, normalize)
    end

    # also check the m=1 special case of the phase formula
    @info "Checking the m=1 configuration (s=5, c=4)"
    re1 = RopeEncoder(k = L, s = 5, m = 1, c = 4)
    for normalize in 0:3
        check_pair(re1, dnas, starts, stops, normalize)
    end

    # jagged reads of random lengths
    @info "Checking jagged reads of random lengths"
    reads_jag = [rand(UInt8(0):UInt8(3), L - rand(0:100)) for _ in 1:n_reads]
    jreads_jag = jagged_array(reads_jag)
    dnas_jag = jreads_jag.vect
    starts_jag = cu(jreads_jag.starts)
    stops_jag = cu(jreads_jag.stops)
    for normalize in (0, 1)
        check_pair(re, dnas_jag, starts_jag, stops_jag, normalize)
    end

    # timing (all kernels are compiled by now)
    normalize = 0
    @info "Timing the original encode_batch_cuda_all_best"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, dnas; starts = starts, stops = stops, normalize = normalize)
    end
    @info "Timing encode_batch_cuda_all_best_v2"
    for _ in 1:3
        CUDA.@time cropes, _ = encode_batch_cuda_all_best_v2(re, dnas; starts = starts, stops = stops, normalize = normalize)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
