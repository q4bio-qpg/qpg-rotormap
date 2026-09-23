# ==============================================================================
# humanref_v2.jl -- stream the human reference genome (GRCh38.p14 primary
# assembly, 68 records, ~3.1 Gb) through the v2 loader (FastaBatchesV2, the
# newflow_v2.jl parse engine), rope-encode every full k-window fragment
# (fwd + rc, kernel_best_v3, normalize = 0) on the GPU, and save the ROPE
# ENCODINGS -- the set of complex vectors -- as a Julia-serialized .bin
# (Utils.save) next to the FASTA. Run on kau with 16 cores:
#
#   julia --project=RotorMap -t 16 RotorMap/test/humanref_v2.jl
#
# `... humanref_v2.jl bench` runs the same pipeline JIT-excluded: one cold
# rep (labeled "incl. JIT") + steady-state reps of parse-only and of the full
# pipeline (parse + cut + encode + D2H), no textual checks, nothing written.
#
# Difference to humanref.jl: that script saves the PACKED DNA bitstream
# (<fasta>.bin) and only reports rope statistics; this script saves the
# encodings themselves to <fasta dir>/humanref.bin. It is a true streamed
# flow (the newflow_v2.jl shape): batches are record-aligned, so every
# record's k-window fragments lie entirely inside one batch's packed
# bitstream -- no whole-genome stream concatenation (humanref.jl's seam
# merging) and no packed .bin are needed; only one batch of packed words is
# alive at a time. Per batch:
#
#   cut floor(len/k) full k-windows per record (batch-local Int32 bounds)
#   -> MIRROR every window [a,b] within its record [s,e] to [s+e-b, s+e-a]
#      (GOTCHA, see below)
#   -> upload words + rwords + kernel bounds (H2D): MIRRORED bounds for the
#      fwd stream, RAW genomic bounds for the rc stream (GOTCHA below)
#   -> one kernel_best_v3 encode fwd + rc (normalize = 0)
#   -> CUDA.synchronize -> D2H copyto! into preallocated PINNED host arrays
#      at linear offsets (the newflow_v2.md collecting-sink recipe)
#
# GOTCHA -- kernel addressing of sub-record windows: the v3 packed layout
# stores each read's bases in REVERSE order (newencoder2bit.jl endianness
# note): base i of the read [start, stop] lives at 0-based base-slot
# g = stop - i (slot g = global bits 2g..2g+1). That convention is only
# valid for a window that is itself a packed read; a k-window [a, b] INSIDE
# a record [s, e] of a record-packed stream must be handed to the kernel
# MIRRORED within the record, as [s+e-b, s+e-a] -- then its slots e'-i =
# s+e-a-i hold exactly the fragment's bases in forward order and the phase
# mapping folds back to the fragment's own windows. Passing [a, b] directly
# makes the kernel read a different (reflected) window with folded phases
# (humanref.jl step [5] did exactly that; statistics-only there, and
# statistically indistinguishable -- this script's textual and re-encode
# checks catch it).
# The rc stream needs the OPPOSITE bounds: it holds each record's reverse
# complement packed the same way, so the fragment's revcomp sits at the
# fragment's OWN address [a, b] -- verified over random records/windows:
#   kernel(words, [s+e-b, s+e-a]) = fragment bases (forward order)
#   kernel(rwords, [a, b])        = fragment bases (reverse-complement order)
#   rwords[a..b] base i == 3 - words[ks..ke] base (k+1-i)   (per-window)
# The saved frag_starts/frag_stops are the TRUE genomic windows (they double
# as the rc-stream kernel bounds); pack_starts/pack_stops are the fwd-stream
# (mirrored) bounds to hand to the kernel when re-encoding from a
# record-packed bitstream.
#
# The saved NamedTuple (format :rope_ref_v1):
#   format, convention, source, k, s, m, c, normalize,
#   n_seqs, nbases, n_frags,
#   frag_seq    (Int32, record index per fragment, 1-based into heads),
#   frag_starts / frag_stops (Int64, global 1-based base offsets of the
#                genomic k-windows, over the records concatenated in file
#                order),
#   pack_starts / pack_stops (Int64, the same windows mirrored within their
#                records -- the bounds to pass to kernel_best_v3 for the FWD
#                stream of a record-packed bitstream, see GOTCHA; the rc
#                stream takes frag_starts/frag_stops directly),
#   heads       (record header lines, with '>'),
#   cropes / norms         (ComplexF32 (m,4^c,n_frags) / Float32 (m,n_frags)),
#   cropes_rc / norms_rc   (reverse complements)
# Encodings are stored raw (normalize = 0) together with their norms, so
# consumers can normalize downstream. The file holds only encodings +
# fragment provenance, not the bases -- pair it with humanref.jl's packed
# .bin (same parse convention) when the sequences themselves are needed.
#
# Verification:
#   [a] per-batch fragment arithmetic asserts (count / size / coverage);
#   [b] textual spot-checks: sampled fragments (every batch's first fragment,
#       the last fragment-bearing record's first fragment, and each new
#       longest record's LAST full fragment) are decoded from the batch's
#       packed words and compared against the raw FASTA text (memmem of the
#       header + a base-counting walk; the text is folded to the stream
#       convention: upper case, N/n and IUPAC B/K/M/R/S/W/Y -> G), plus the
#       revcomp invariant on the samples;
#   [c] independent re-encode of every sample: the k bases are re-extracted
#       from the raw text, re-packed with pack_reads and re-encoded through
#       the v3 wrapper -- must match the accumulated encodings up to
#       atomic-order rounding;
#   [d] save/reload: Utils.load round-trip, all fields bitwise ==.
#
# Config (env overrides): HUMANREF_FASTA, HUMANREF_OUT, HUMANREF_K,
# HUMANREF_BATCH (records per parse batch).
#
# RESULTS (kau, RTX 5090, -t 16, batch_reads = 8, k = 20,000, 2026):
# the FASTA (3,147,051,578 bytes) streams in 9 batches (batch 1 = chr1 + 7
# more, 12,477 frags) in ~5.0 s wall (loader stages: parse ~75%, merge ~21%,
# scan ~4%); 68 sequences, 3,095,453,524 packed bases (N/n + 99 IUPAC
# B/K/M/R/S/W/Y -> G, positions preserved). 67/68 records reach k (only the
# 16,569 bp mt genome falls short): 154,737 full k-windows -- 3,094,740,000
# bases covered, 713,524 leftover in record tails; step = k/10 sliding
# windows would give 1,547,081 entries. Encode fwd+rc + D2H: 0.70-0.74 s =
# ~210-220k frags/s, device pool peak 0.26 GiB used; textual sampling of 19
# fragments (every batch's first, each batch's last record-bearing first,
# chr1's LAST full fragment) 0.6 s incl. one ~250 M-base walk. Independent
# re-encode of all 19 samples from the raw text (pack_reads -> v3 wrapper)
# matches the accumulated encodings, max diff 9.8e-7 (atomic-order class);
# this check is what caught the sub-record-window addressing GOTCHA above
# (and the same latent bug in humanref.jl step [5]). Utils.save writes
# humanref.bin (2,545,741,007 bytes = 154,737 x 1,024 x ComplexF32 fwd+rc +
# norms + provenance) in ~0.4 s; Utils.load reloads bitwise == in ~0.5 s;
# mean norm per fragment 2.396e5 (fwd = rc). Total ~5.9 s + julia/CUDA
# startup.
#
# BENCH (JIT-excluded, `... humanref_v2.jl bench`, 4 reps, same hardware):
# parse-only steady state 1.59 s = 1.98 GB/s (rep 1 incl. JIT: 3.32 s =
# 0.95 GB/s); full pipeline (parse + cut + encode fwd+rc + D2H) steady state
# 2.24 s = 1.40 GB/s, encode+d2h 0.46 s = ~335k frags/s (rep 1: 2.64 s).
# Identical counts every rep (68 seqs, 3,095,453,524 bases, 154,737 frags).
# ==============================================================================

include(joinpath(@__DIR__, "newflow_v2.jl")) # FastaBatchesV2 (the E1/E2 parse
# engine) + the v3 rope encode kernels + pack_reads (+ transitively: newflow.jl
# -> newfasta.jl -> newencoder2bit.jl; RotorMap.Utils save/load)

# ==============================================================================
# IUPAC substitution (same deviation from the loaders' default as
# humanref.jl): the shared pack LUT accepts only ACGTN/acgtn (N/n -> G) and
# silently DROPS every other letter (LUT 0xff), shifting positions by the
# dropped letters before them. GRCh38.p14 contains 99 such letters
# (B/K/M/R/S/W/Y); substitute them to G -- the same treatment as N/n -- so
# every raw sequence character keeps its position. Both parse paths must see
# the substitution: _PACK_LUT is read per byte by the scalar fallback, and
# _PAIR_LUT (derived from it at include time) gates the SWAR fast path.
# ==============================================================================
const _IUPAC = (UInt8('B'), UInt8('K'), UInt8('M'), UInt8('R'),
                UInt8('S'), UInt8('W'), UInt8('Y'))
for c in _IUPAC
    _PACK_LUT[Int(c) + 1] = 0x02  # upper case -> G, like N/n
    _PACK_LUT[Int(c) + 33] = 0x02 # lower case ('a' = 'A' + 32; index = code + 1)
end
let # rebuild every _PAIR_LUT entry from the patched _PACK_LUT (same entry
    # layout as newfasta_v2.jl: bits 0-1/2-3 = the two base codes, bit 4 =
    # both bytes are bases, bit 5 = the pair contains N/n, 0x00 = invalid)
    N1 = UInt8('N'); N2 = UInt8('n')
    for x in 0:65535
        b0 = UInt8(x & 0xff); b1 = UInt8(x >> 8)
        v0 = _PACK_LUT[Int(b0) + 1]; v1 = _PACK_LUT[Int(b1) + 1]
        _PAIR_LUT[x + 1] = v0 != 0xff && v1 != 0xff ?
                           v0 | (v1 << 2) | 0x10 |
                           ((b0 == N1 || b0 == N2 || b1 == N1 || b1 == N2) ? 0x20 : 0x00) :
                           0x00
    end
end

using CUDA
using Printf
using Mmap
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders
using Base.Threads

const FASTA = get(ENV, "HUMANREF_FASTA",
                  "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
const OUTFILE = get(ENV, "HUMANREF_OUT", joinpath(dirname(FASTA), "humanref.bin"))
const K = parse(Int, get(ENV, "HUMANREF_K", "20000"))          # rope fragment length
const BATCH_READS = parse(Int, get(ENV, "HUMANREF_BATCH", "8")) # batch < 2^31 bases

# ==============================================================================
# raw-text helpers (verification against the FASTA bytes, as in humanref.jl)
# ==============================================================================

function _find_pattern(raw, needle::AbstractString, from::Int)
    nb = codeunits(needle)
    from > length(raw) && return 0
    p = ccall(:memmem, Ptr{UInt8}, (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t),
              pointer(raw, from), length(raw) - from + 1, pointer(nb), length(nb))
    return p == C_NULL ? 0 : Int(p - pointer(raw)) + 1
end

# walk `want` sequence bytes from byte `from` in direction `dir` (+1/-1),
# skipping newlines
function _collect_bases(raw, from::Int, want::Int, dir::Int)
    out = Vector{UInt8}(undef, want)
    i = from
    k = 0
    while k < want
        c = raw[i]
        i += dir
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        k += 1
        out[k] = c
    end
    return out
end

"byte position of the `target`-th sequence base (newlines skipped) at/after `from`"
function _seq_pos(raw, from::Int, target::Int)
    cnt = 0
    i = from
    @inbounds while true
        @assert i <= length(raw) "sequence truncated in the raw text"
        c = raw[i]
        if c != UInt8('\n') && c != UInt8('\r')
            cnt += 1
            cnt == target && return i
        end
        i += 1
    end
end

"code of base i (1-based within the [gstart, gstart+L-1] window) of a packed stream"
function _code(words, gstart::Int, L::Int, i::Int)
    bit = 2 * (gstart - 1) + 2 * (L - i)
    return Int((words[(bit >> 5) + 1] >> (bit & 31)) & 3)
end

_decode_str(words, gstart::Int, L::Int, idxs) =
    String(["ACGT"[_code(words, gstart, L, i) + 1] for i in idxs])

"fold raw FASTA text to the stream convention (upper case; N/n and the IUPAC
letters B/K/M/R/S/W/Y -> G, exactly what the packed stream holds)"
function _fold(txt::AbstractVector{UInt8})
    out = Vector{UInt8}(undef, length(txt))
    for (j, c) in enumerate(txt)
        c in UInt8('a'):UInt8('z') && (c -= 0x20)
        @assert c in (UInt8('A'), UInt8('C'), UInt8('G'), UInt8('T'), UInt8('N')) ||
                c in _IUPAC "unexpected letter $(Char(c))"
        out[j] = (c == UInt8('N') || c in _IUPAC) ? UInt8('G') : c
    end
    return String(out)
end

"raw FASTA bases -> 2-bit codes (0=A, 1=C, 2=G, 3=T; N/n and IUPAC -> G)"
function _fold_codes(txt::AbstractVector{UInt8})
    out = Vector{UInt8}(undef, length(txt))
    for (j, c) in enumerate(txt)
        c in UInt8('a'):UInt8('z') && (c -= 0x20)
        @assert c in (UInt8('A'), UInt8('C'), UInt8('G'), UInt8('T'), UInt8('N')) ||
                c in _IUPAC "unexpected letter $(Char(c))"
        out[j] = c == UInt8('A') ? 0x00 : c == UInt8('C') ? 0x01 :
                 c == UInt8('T') ? 0x03 : 0x02
    end
    return out
end

"verify one sampled fragment against the raw FASTA text: decode its head/tail
bases from the packed batch words (via the window MIRRORED within its record,
`kstart` -- see the GOTCHA in the header -- whose standard read addressing
yields the fragment's bases in forward order), compare with the folded text at
the walked positions, and check the revcomp invariant against the rc stream
read at the fragment's RAW bounds `fstart` (the rc stream is the revcomp
genome packed the same way, so revcomp(frag) sits at the fragment's own
address). Returns the fragment's first-base text byte (for the independent
re-encode pass). `t0` is the text byte just after the record's header line."
function _text_check!(raw, words, rwords, kstart::Int, fstart::Int, k::Int,
                      t0::Int, off_in_rec::Int, tag::String)
    p1 = _seq_pos(raw, t0, off_in_rec + 1)
    p2 = _seq_pos(raw, t0, off_in_rec + k)
    n = 40
    @assert _fold(_collect_bases(raw, p1, n, +1)) == _decode_str(words, kstart, k, 1:n) "fragment head bases mismatch ($tag)"
    @assert _fold(reverse!(_collect_bases(raw, p2, n, -1))) == _decode_str(words, kstart, k, k-n+1:k) "fragment tail bases mismatch ($tag)"
    for i in vcat(1:n, k-n+1:k) # rc stream at RAW bounds = revcomp of the fragment
        @assert _code(rwords, fstart, k, i) == 3 - _code(words, kstart, k, k + 1 - i) "revcomp invariant broken ($tag, base $i)"
    end
    return p1
end

"re-extract each sampled fragment's k bases from the raw text, re-pack with
pack_reads and re-encode (fwd + rc) through the v3 wrapper, and compare with
the accumulated encodings; returns the max abs diff"
function _reencode_check(re, raw, samples, Afwd, Afrc, k::Int)
    maxerr = 0.0
    for (gidx, head, off, p1) in samples
        codes = _fold_codes(_collect_bases(raw, p1, k, +1))
        @assert length(codes) == k "sample $gidx: extracted $(length(codes)) of $k bases"
        cw, cs, ce = pack_reads([codes])
        rw, _, _ = pack_reads([UInt8(3) .- reverse(codes)]) # rc read = complemented, forward order
        dest, _ = encode_batch_cuda_all_best_v3(re, CuArray(cw); starts = CuArray(cs),
                                                stops = CuArray(ce), normalize = 0)
        rdest, _ = encode_batch_cuda_all_best_v3(re, CuArray(rw); starts = CuArray(cs),
                                                 stops = CuArray(ce), normalize = 0)
        af = view(Afwd, :, :, gidx)
        ar = view(Afrc, :, :, gidx)
        maxerr = max(maxerr, maximum(abs.(Array(dest) .- af)), maximum(abs.(Array(rdest) .- ar)))
        @assert isapprox(Array(dest), af; rtol = 1e-3, atol = 2e-4) "fwd re-encode mismatch (fragment $gidx: $head)"
        @assert isapprox(Array(rdest), ar; rtol = 1e-3, atol = 2e-4) "rc re-encode mismatch (fragment $gidx: $head)"
    end
    return maxerr
end

# compile the v3 kernels outside the timed region (tiny synthetic read)
function _warm_encode(re)
    dw = CuArray(fill(UInt32(0x1b1b1b1b), cld(2 * re.k, 32) + 1))
    ds = CuArray(Int32[1])
    de = CuArray(Int32[re.k])
    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, 1)
    dn = CUDA.zeros(Float32, re.m, 1)
    for _ in 1:2
        encode_batch_cuda_all_best_v3!(dest, dn, re, dw; starts = ds, stops = de, normalize = 0)
    end
    CUDA.synchronize()
    return nothing
end

# ==============================================================================
# main
# ==============================================================================

function _comma(n::Integer)
    s = string(n)
    out = IOBuffer()
    for (i, c) in enumerate(s)
        print(out, c)
        from_end = length(s) - i
        (from_end > 0 && from_end % 3 == 0) && print(out, ',')
    end
    return String(take!(out))
end

function main()
    @printf("== humanref_v2.jl: %s -> rope encodings -> %s ==\n", basename(FASTA), OUTFILE)
    @printf("source: %s (%s bytes)\n", FASTA, _comma(filesize(FASTA)))
    @printf("threads: %d   gpu: %s   k: %d   batch_reads: %d   encoder: RopeEncoder(s=%d, m=%d, c=%d)\n\n",
            nthreads(), CUDA.name(device()), K, BATCH_READS, 8, 4, 4)
    @assert isfile(FASTA) "FASTA not found: $FASTA"

    re = RopeEncoder(k = K, s = 8, m = 4, c = 4)
    nb_el = re.m * 4^re.c
    nfrag_ub = Int(filesize(FASTA) ÷ K) + 1 # every base costs >= 1 file byte

    # ---- [1/6] host accumulation (upper bound), raw text view, warm-up -------
    @printf("[1/6] preallocating host accumulation: upper bound %s fragments, 2 x (%s x %s) ComplexF32 = %.2f GiB + pinning\n",
            _comma(nfrag_ub), _comma(nb_el), _comma(nfrag_ub), 2 * nb_el * nfrag_ub * 8 / 2^30)
    Afwd = Array{ComplexF32}(undef, re.m, 4^re.c, nfrag_ub) # (m, 4^c, nfrag):
    Afrc = similar(Afwd)                          # linear layout == per-frag blocks,
    Nfwd = Array{Float32}(undef, re.m, nfrag_ub)  # so the D2H copyto! below is a
    Nfrc = similar(Nfwd)                          # plain contiguous DMA per batch
    CUDA.pin(Afwd) # cudaHostRegister, one-time: pinned-dest D2H at ~34 GB/s
    CUDA.pin(Afrc)
    raw = open(FASTA, "r") do io
        Mmap.mmap(io)
    end
    _warm_encode(re)

    # ---- [2/6] stream: parse batch -> cut fragments -> encode -> D2H ---------
    @printf("\n[2/6] streaming (FastaBatchesV2) + fragment encode fwd+rc (kernel_best_v3, normalize=0)\n")
    gfs = Int64[]             # fragment genomic global starts (Int64: > 2^31 bases)
    gfe = Int64[]             # fragment genomic global stops
    gks = Int64[]             # fragment kernel bounds, mirrored in-record (global)
    gke = Int64[]
    gfseq = Int32[]           # fragment -> record index (1-based into heads)
    heads_all = String[]
    rec_lens = Int64[]
    samples = Tuple{Int,String,Int,Int}[] # (global frag idx, head, off in record, text byte)
    acc = Int64(0)            # global packed bases before the current batch
    nseq = 0
    nfrag = 0
    nbatches = 0
    maxlen = 0                # longest record seen so far (deep-sample trigger)
    cursor = 1                # raw-text memmem cursor (records appear in file order)
    cur_head = ""             # last located header (same-record samples reuse it)
    t_enc = 0.0               # upload + encode + D2H
    t_chk = 0.0               # textual sampling
    _v2time_reset!()
    t_stream = @elapsed begin
        it = FastaBatchesV2(FASTA; batch_reads = BATCH_READS, revcomp = true, keep_heads = true)
        cur = iterate(it)
        while cur !== nothing
            nt, hi = cur
            nbatches += 1
            n = length(nt.starts)
            nseq0 = nseq
            nfrag0 = nfrag
            nb = 0
            t_b = 0.0
            @assert nt.rwords !== nothing "revcomp batch stream missing (batch $nbatches)"
            if n > 0
                bi = Int(nt.stops[end])
                @assert bi < (Int64(1) << 31) "batch $nbatches exceeds the 2^31-base Int32 offset limit: lower batch_reads"
                @assert length(nt.words) == cld(2 * bi, 32) + 1 == length(nt.rwords) "unexpected batch stream layout (batch $nbatches)"
                append!(rec_lens, Int64.(nt.stops) .- Int64.(nt.starts) .+ 1)

                # [a] cut floor(len/k) full k-windows per record; batches are
                # record-aligned, so fragments never cross a batch seam. Each
                # genomic window [a, b] is mirrored within its record [s, e]
                # to [s+e-b, s+e-a] for the kernel (reversed-per-read layout,
                # GOTCHA in the header); Int64 intermediate: s+e overflows Int32
                fs = Int32[]; fe = Int32[]; ks = Int32[]; ke = Int32[]; fseq = Int32[]
                rec_first = zeros(Int, n); rec_last = zeros(Int, n)
                for i in 1:n
                    L = Int(nt.stops[i]) - Int(nt.starts[i]) + 1
                    se = Int(nt.starts[i]) + Int(nt.stops[i])
                    for f in 1:(L ÷ K)
                        a = nt.starts[i] + (f - 1) * K
                        b = a + K - 1
                        push!(fs, a)
                        push!(fe, b)
                        push!(ks, Int32(se - Int(b)))
                        push!(ke, Int32(se - Int(a)))
                        push!(fseq, nseq0 + i)
                    end
                    nf = L ÷ K
                    if nf > 0
                        rec_first[i] = length(fs) - nf + 1
                        rec_last[i] = length(fs)
                    end
                end
                nb = length(fs)
                @assert nb == sum((Int(nt.stops[i]) - Int(nt.starts[i]) + 1) ÷ K for i in 1:n) "fragment count mismatch (batch $nbatches)"
                @assert all(j -> Int(fe[j]) - Int(fs[j]) == K - 1 && Int(ke[j]) - Int(ks[j]) == K - 1, 1:nb) "fragment size mismatch (batch $nbatches)"
                @assert nb == 0 || (Int(fs[1]) >= Int(nt.starts[1]) && Int(fe[end]) <= Int(nt.stops[end])) "fragment bounds outside the batch (batch $nbatches)"

                if nb > 0
                    # upload + encode fwd/rc + D2H into the pinned accumulators
                    t_b = @elapsed begin
                        dw = CuArray(nt.words)
                        drw = CuArray(nt.rwords)
                        ds = CuArray(ks) # FWD stream: MIRRORED kernel bounds
                        de = CuArray(ke)
                        dsr = CuArray(fs) # RC stream: RAW genomic bounds
                        der = CuArray(fe)
                        dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, nb)
                        dn = CUDA.zeros(Float32, re.m, nb)
                        rdest = CUDA.zeros(ComplexF32, re.m, 4^re.c, nb)
                        rdn = CUDA.zeros(Float32, re.m, nb)
                        encode_batch_cuda_all_best_v3!(dest, dn, re, dw; starts = ds, stops = de, normalize = 0)
                        encode_batch_cuda_all_best_v3!(rdest, rdn, re, drw; starts = dsr, stops = der, normalize = 0)
                        CUDA.synchronize()
                        copyto!(Afwd, nfrag0 * nb_el + 1, dest, 1, nb * nb_el)
                        copyto!(Afrc, nfrag0 * nb_el + 1, rdest, 1, nb * nb_el)
                        copyto!(Nfwd, nfrag0 * re.m + 1, dn, 1, nb * re.m)
                        copyto!(Nfrc, nfrag0 * re.m + 1, rdn, 1, nb * re.m)
                        @assert minimum(view(Nfwd, :, nfrag0+1:nfrag0+nb)) > 0 "non-finite fwd norms (batch $nbatches)"
                        @assert minimum(view(Nfrc, :, nfrag0+1:nfrag0+nb)) > 0 "non-finite rc norms (batch $nbatches)"
                        dw = drw = ds = de = dsr = der = dest = dn = rdest = rdn = nothing
                    end
                    t_enc += t_b

                    append!(gfs, Int64.(fs) .+ acc)
                    append!(gfe, Int64.(fe) .+ acc)
                    append!(gks, Int64.(ks) .+ acc) # mirroring is in-record, so
                    append!(gke, Int64.(ke) .+ acc) # global = batch offset + local
                    append!(gfseq, fseq)
                    nfrag = nfrag0 + nb

                    # [b] textual samples: the batch's first fragment, the first
                    # fragment of its last fragment-bearing record, and each new
                    # longest record's LAST full fragment (deep stream position)
                    pend = Tuple{Int,Int,Int}[] # (local record, frag idx in batch, off in record)
                    push!(pend, (fseq[1] - nseq0, 1, Int(fs[1]) - Int(nt.starts[fseq[1] - nseq0])))
                    il = findlast(>(0), rec_last)
                    if il !== nothing && rec_first[il] != 1
                        push!(pend, (il, rec_first[il], Int(fs[rec_first[il]]) - Int(nt.starts[il])))
                    end
                    Ls = Int.(nt.stops) .- Int.(nt.starts) .+ 1
                    imax = argmax(Ls)
                    if Ls[imax] > maxlen
                        maxlen = Ls[imax]
                        if rec_last[imax] > 0
                            j = rec_last[imax]
                            push!(pend, (imax, j, Int(fs[j]) - Int(nt.starts[imax])))
                        end
                    end
                    sort!(pend; by = p -> p[1]) # file order for the memmem cursor
                    t_cb = @elapsed begin
                        for (lr, j, off) in pend
                            h = nt.heads[lr]
                            if h != cur_head
                                hpos = _find_pattern(raw, h, cursor)
                                @assert hpos > 0 "head not found in the raw file (from $cursor): $h"
                                cursor = hpos + sizeof(h) # text byte after the header line
                                cur_head = h
                            end
                            p1 = _text_check!(raw, nt.words, nt.rwords, Int(ks[j]), Int(fs[j]), K,
                                              cursor, off, "frag $(nfrag0 + j): $h")
                            push!(samples, (nfrag0 + j, h, off, p1))
                        end
                    end
                    t_chk += t_cb
                end
                acc += bi
                nseq += n
                append!(heads_all, nt.heads)
            end
            @printf("      batch %d: %s records, %s fragments%s\n", nbatches, _comma(n),
                    _comma(nb), nb > 0 ? @sprintf("  encode+d2h %.2f s", t_b) : "")
            cur = iterate(it, hi)
        end
    end
    _v2time_print()

    # ---- [3/6] reference statistics ------------------------------------------
    @printf("\n[3/6] reference statistics\n")
    @assert nfrag == length(gfs) == length(gfe) == length(gks) ==
            length(gke) == length(gfseq) "fragment bookkeeping mismatch"
    @assert nfrag <= nfrag_ub "fragment count exceeds the preallocated upper bound"
    covered = Int64(nfrag) * K
    nrec_k = count(>=(K), rec_lens)
    stp = K ÷ 10
    nslide = sum((L - K) ÷ stp + 1 for L in rec_lens if L >= K; init = 0)
    @printf("      sequences: %s   bases: %s (N/n and IUPAC B/K/M/R/S/W/Y are substituted to G)\n",
            _comma(nseq), _comma(acc))
    @printf("      records >= k: %d/%d (min %s, max %s, mean %s bases)\n",
            nrec_k, nseq, _comma(Int(minimum(rec_lens))), _comma(Int(maximum(rec_lens))),
            _comma(round(Int, sum(rec_lens) / nseq)))
    @printf("      fragments: %s full k-windows (%s bases covered, %s leftover in record tails)\n",
            _comma(nfrag), _comma(covered), _comma(acc - covered))
    @printf("      (info: sliding windows with step = k/10 would give %s entries)\n", _comma(nslide))
    if nfrag > 0
        @printf("      encode+d2h: %.2f s = %s frags/s; textual sampling %.2f s; stream total %.2f s\n",
                t_enc, _comma(round(Int, nfrag / t_enc)), t_chk, t_stream)
    end
    @printf("      device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)

    # ---- [4/6] independent re-encode of the textual samples -------------------
    @printf("\n[4/6] independent re-encode of %d textual samples (raw text -> pack_reads -> v3)\n",
            length(samples))
    maxerr = NaN
    t_re = 0.0
    if !isempty(samples)
        t_re = @elapsed maxerr = _reencode_check(re, raw, samples, Afwd, Afrc, K)
        @printf("      %d/%d fragments match the accumulated encodings: PASS (max diff %.2e, %.2f s)\n",
                length(samples), length(samples), maxerr, t_re)
    else
        @printf("      no fragments -- nothing to check\n")
    end

    # ---- [5/6] save the encodings --------------------------------------------
    @printf("\n[5/6] saving %s (Serialization, Utils.save)\n", OUTFILE)
    GC.gc()
    cropes = Afwd[:, :, 1:nfrag]; Afwd = nothing; GC.gc()
    norms = Nfwd[:, 1:nfrag]; Nfwd = nothing
    cropes_rc = Afrc[:, :, 1:nfrag]; Afrc = nothing; GC.gc()
    norms_rc = Nfrc[:, 1:nfrag]; Nfrc = nothing
    data = (format = :rope_ref_v1,
            convention = "2-bit packed ACGT, reversed base order (kernel_best_v3 layout); N/n and IUPAC B/K/M/R/S/W/Y -> G",
            source = basename(FASTA),
            k = K,
            s = re.s,
            m = re.m,
            c = re.c,
            normalize = 0,
            n_seqs = nseq,
            nbases = acc,
            n_frags = nfrag,
            frag_seq = gfseq,      # Int32 record index per fragment (1-based into heads)
            frag_starts = gfs,     # Int64 global 1-based base offsets of the genomic
            frag_stops = gfe,      # k-windows (records concatenated in file order)
            pack_starts = gks,     # the same windows mirrored within their records:
            pack_stops = gke,      # kernel_best_v3 bounds for the FWD stream of a
                                   # record-packed bitstream (rc: frag_starts/stops)
            heads = heads_all,     # FASTA header lines (with '>')
            cropes = cropes,       # ComplexF32 (m, 4^c, n_frags), forward strands
            norms = norms,         # Float32 (m, n_frags), per-m norm sums (normalize = 0)
            cropes_rc = cropes_rc, # reverse complements
            norms_rc = norms_rc)
    t_save = @elapsed Utils.save(data, OUTFILE)
    @printf("      %s bytes (%s frags x %s bins x ComplexF32, fwd+rc) in %.2f s (%.0f MB/s)\n",
            _comma(filesize(OUTFILE)), _comma(nfrag), _comma(nb_el), t_save,
            filesize(OUTFILE) / t_save / 1e6)

    # ---- [6/6] verify the round-trip -----------------------------------------
    @printf("\n[6/6] verifying the .bin\n")
    t_load = @elapsed data2 = Utils.load(OUTFILE)
    ok = data2.format === :rope_ref_v1 && data2.convention == data.convention &&
         data2.source == basename(FASTA) && data2.k == K && data2.s == re.s &&
         data2.m == re.m && data2.c == re.c && data2.normalize == 0 &&
         data2.n_seqs == nseq && data2.nbases == acc && data2.n_frags == nfrag &&
         data2.frag_seq == gfseq && data2.frag_starts == gfs && data2.frag_stops == gfe &&
         data2.pack_starts == gks && data2.pack_stops == gke &&
         data2.heads == heads_all && data2.cropes == cropes && data2.norms == norms &&
         data2.cropes_rc == cropes_rc && data2.norms_rc == norms_rc
    @printf("      reload: %.2f s   fields bitwise ==: %s\n", t_load, ok ? "PASS" : "FAIL")
    @assert ok "reloaded .bin differs"
    @printf("      mean norm per fragment: fwd %.3e   rc %.3e\n",
            sum(norms) / (re.m * nfrag), sum(norms_rc) / (re.m * nfrag))

    @printf("\nALL DONE in %.1f s (stream %.2f [encode+d2h %.2f, textual %.2f] + re-encode %.2f + save %.2f + reload %.2f)\n",
            t_stream + t_re + t_save + t_load, t_stream, t_enc, t_chk, t_re, t_save, t_load)
    return nothing
end

# ==============================================================================
# bench mode: steady-state (JIT-excluded) timings -- `... humanref_v2.jl bench`
#
# Rep 1 of each measurement compiles the per-batch paths (labeled "incl.
# JIT"); reps 2+ are JIT-free steady state in the same process. The full-
# pipeline pass mirrors main()'s per-batch work (cut -> mirrored/raw kernel
# bounds -> upload -> encode fwd+rc -> D2H into the pinned accumulators)
# minus the textual checks and the save/reload. Nothing is written.
# ==============================================================================

function _parse_pass(; batch_reads::Int = BATCH_READS)
    nseq = 0
    nbases = Int64(0)
    nwords = 0
    it = FastaBatchesV2(FASTA; batch_reads = batch_reads, revcomp = true, keep_heads = false)
    cur = iterate(it)
    while cur !== nothing
        nt, hi = cur
        n = length(nt.starts)
        if n > 0
            nseq += n
            nbases += Int64(nt.stops[end])
            nwords += length(nt.words) + length(nt.rwords)
        end
        cur = iterate(it, hi)
    end
    return nseq, nbases, nwords
end

"one parse + fragment-cut + encode fwd/rc + D2H pass into the pinned accumulators"
function _pipeline_pass(re, Afwd, Afrc, Nfwd, Nfrc, nb_el; batch_reads::Int = BATCH_READS)
    nfrag = 0
    nseq = 0
    nbases = Int64(0)
    t_enc = 0.0
    it = FastaBatchesV2(FASTA; batch_reads = batch_reads, revcomp = true, keep_heads = false)
    cur = iterate(it)
    while cur !== nothing
        nt, hi = cur
        n = length(nt.starts)
        if n > 0
            fs = Int32[]; fe = Int32[]; ks = Int32[]; ke = Int32[]
            for i in 1:n
                L = Int(nt.stops[i]) - Int(nt.starts[i]) + 1
                se = Int(nt.starts[i]) + Int(nt.stops[i])
                for f in 1:(L ÷ K)
                    a = nt.starts[i] + (f - 1) * K
                    b = a + K - 1
                    push!(fs, a)
                    push!(fe, b)
                    push!(ks, Int32(se - Int(b)))
                    push!(ke, Int32(se - Int(a)))
                end
            end
            nb = length(fs)
            if nb > 0
                t_b = @elapsed begin
                    dw = CuArray(nt.words)
                    drw = CuArray(nt.rwords)
                    ds = CuArray(ks) # FWD stream: MIRRORED kernel bounds
                    de = CuArray(ke)
                    dsr = CuArray(fs) # RC stream: RAW genomic bounds
                    der = CuArray(fe)
                    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, nb)
                    dn = CUDA.zeros(Float32, re.m, nb)
                    rdest = CUDA.zeros(ComplexF32, re.m, 4^re.c, nb)
                    rdn = CUDA.zeros(Float32, re.m, nb)
                    encode_batch_cuda_all_best_v3!(dest, dn, re, dw; starts = ds, stops = de, normalize = 0)
                    encode_batch_cuda_all_best_v3!(rdest, rdn, re, drw; starts = dsr, stops = der, normalize = 0)
                    CUDA.synchronize()
                    copyto!(Afwd, nfrag * nb_el + 1, dest, 1, nb * nb_el)
                    copyto!(Afrc, nfrag * nb_el + 1, rdest, 1, nb * nb_el)
                    copyto!(Nfwd, nfrag * re.m + 1, dn, 1, nb * re.m)
                    copyto!(Nfrc, nfrag * re.m + 1, rdn, 1, nb * re.m)
                    dw = drw = ds = de = dsr = der = dest = dn = rdest = rdn = nothing
                end
                t_enc += t_b
                nfrag += nb
            end
            nseq += n
            nbases += Int64(nt.stops[end])
        end
        cur = iterate(it, hi)
    end
    return nfrag, nseq, nbases, t_enc
end

function bench(; reps::Int = 4)
    @printf("== humanref_v2.jl bench: JIT-excluded steady-state timings ==\n")
    @printf("source: %s (%s bytes)\n", FASTA, _comma(filesize(FASTA)))
    @printf("threads: %d   gpu: %s   k: %d   batch_reads: %d   reps: %d\n\n",
            nthreads(), CUDA.name(device()), K, BATCH_READS, reps)
    @assert isfile(FASTA) "FASTA not found: $FASTA"
    @assert reps >= 2 "need at least one warm + one steady rep"
    re = RopeEncoder(k = K, s = 8, m = 4, c = 4)
    nb_el = re.m * 4^re.c
    nfrag_ub = Int(filesize(FASTA) ÷ K) + 1
    Afwd = Array{ComplexF32}(undef, re.m, 4^re.c, nfrag_ub)
    Afrc = similar(Afwd)
    Nfwd = Array{Float32}(undef, re.m, nfrag_ub)
    Nfrc = similar(Nfwd)
    CUDA.pin(Afwd)
    CUDA.pin(Afrc)
    _warm_encode(re)
    bytes = Float64(filesize(FASTA))

    # ---- [1] parse-only (record scan + parse + 2-bit pack fwd/rc) ------------
    @printf("[1] parse-only (FastaBatchesV2, batch_reads = %d, fwd+rc pack)\n", BATCH_READS)
    tp = Float64[]
    nseq = 0; nbases = Int64(0); nwords = 0
    for r in 1:reps
        local nseq_r, nbases_r, nwords_r
        t = @elapsed ((nseq_r, nbases_r, nwords_r) = _parse_pass())
        nseq, nbases, nwords = nseq_r, nbases_r, nwords_r
        push!(tp, t)
        @printf("      rep %d: %.2f s  %.2f GB/s%s\n", r, t, bytes / t / 1e9,
                r == 1 ? "  (incl. JIT)" : "")
    end
    @printf("      steady state: %.2f s  %.2f GB/s   (%s seqs, %s bases, %s words fwd+rc)\n",
            minimum(tp[2:end]), bytes / minimum(tp[2:end]) / 1e9,
            _comma(nseq), _comma(nbases), _comma(nwords))

    # ---- [2] full pipeline (parse + cut + encode fwd+rc + D2H) ---------------
    @printf("\n[2] full pipeline (parse + cut + encode fwd+rc + D2H, no checks/save)\n")
    tf = Float64[]; te = Float64[]
    nfrag = 0
    for r in 1:reps
        local nfrag_r, nseq_r, nbases_r, t_enc
        t = @elapsed ((nfrag_r, nseq_r, nbases_r, t_enc) = _pipeline_pass(re, Afwd, Afrc, Nfwd, Nfrc, nb_el))
        nfrag, nseq, nbases = nfrag_r, nseq_r, nbases_r
        push!(tf, t); push!(te, t_enc)
        @printf("      rep %d: %.2f s  %.2f GB/s  (%s frags, encode+d2h %.2f s = %s frags/s)%s\n",
                r, t, bytes / t / 1e9, _comma(nfrag), t_enc,
                _comma(round(Int, nfrag / t_enc)), r == 1 ? "  (incl. JIT)" : "")
    end
    @printf("      steady state: %.2f s  %.2f GB/s  (encode+d2h %.2f s = %s frags/s)\n",
            minimum(tf[2:end]), bytes / minimum(tf[2:end]) / 1e9,
            minimum(te[2:end]), _comma(round(Int, nfrag / minimum(te[2:end]))))
    @printf("      sanity: %s seqs, %s bases, %s fragments\n",
            _comma(nseq), _comma(nbases), _comma(nfrag))
    @printf("      device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "run" : ARGS[1]
    mode == "run" && main()
    mode == "bench" && bench()
    mode in ("run", "bench") || error("unknown mode $mode (use run|bench)")
end
