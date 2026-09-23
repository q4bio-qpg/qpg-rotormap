# =============================================================================
# pack.jl -- 2-bit DNA packing primitives (from legacy newfasta.jl +
# newencoder2bit.jl, verbatim):
#   _PACK_LUT           byte -> 2-bit code LUT (ACGT/acgt -> 0:3; N/n -> G as in
#                       load_fasta_mmap_fixed; everything else -> 0xff = drop)
#   kernel_pack_reads   GPU: one block per read, packs codes into the 2-bit
#                       bitstream in REVERSE base order (v3 kernel convention),
#                       atomic-OR word writes into a pre-zeroed stream
#   pack_reads_cuda     pack_reads on the GPU (jagged ranges)
#   pack_reads          CPU reference packer (same stream layout)
#   kernel_unpack       GPU: unpack a packed 2-bit stream back to codes
#   unpack_packed_cuda  unpack entry point
# Consumers derive per-byte-code tables from _PACK_LUT at include time
# (fasta/loader.jl's _PAIR_LUT; fastareads_v3's _LUT3/_PAIR3 are independent
# copies).  NO dependencies beyond CUDA.
# =============================================================================
using CUDA # kernel_pack_reads's CUDA.@atomic expands at definition time

const _PACK_LUT = let
    lut = fill(0xff, 256)
    lut[Int('A')+1] = 0x00; lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02; lut[Int('T')+1] = 0x03
    lut[Int('a')+1] = 0x00; lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02; lut[Int('t')+1] = 0x03
    lut[Int('N')+1] = 0x02 # as in load_fasta_mmap_fixed: N/n are treated as G
    lut[Int('n')+1] = 0x02
    lut
end

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
