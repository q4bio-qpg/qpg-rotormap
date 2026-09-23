# ==============================================================================
# humanref.jl -- apply the v2 streamed FASTA loader/encoder (newflow_v2.jl ->
# newfasta_v2.jl) to the human reference genome (GRCh38.p14 primary assembly,
# 68 records, ~3.1 Gb -- ABOVE the 2^31-base Int32 offset limit), save the
# 2-bit packed DNA as a Julia-serialized .bin next to the FASTA (Utils.save /
# Utils.load, the Serialization format of Utils.jl's save_dnas/load_dnas), and
# print conversion statistics. Run on kau with 16 cores:
#
#   julia --project=RotorMap -t 16 RotorMap/test/humanref.jl
#
# Pipeline:
#   [1] stream FastaBatchesV2 (revcomp, swar) over the FASTA in record-aligned
#       batches and concatenate the per-batch packed streams into ONE
#       continuous whole-genome bitstream. Two things need care at this scale:
#       * batch-local offsets are Int32, so a batch must stay below 2^31 bases
#         (batch_reads = 8: the largest 8-record window of this file is ~1 Gb);
#       * a batch may end mid-word, and its trailing zero padding must not
#         enter the stream: the next batch's payload is funneled into the
#         pending tail and its guard word is stripped (one fresh guard word
#         closes the final stream). The result is bitwise the whole-file
#         `load_fasta_packed` layout: base i of record r lives at global bit
#         2*(starts[r]-1) + 2*(len_r - i) (reversed base order, the v3
#         endianness), the rc stream holds complemented bases in forward order.
#       Record offsets are rebased to global Int64 (1:3,095,453,524).
#   [2] save a NamedTuple (format, source, k, n_seqs, nbases, words, rwords,
#       starts, stops, heads) to <fasta>.bin via Serialization.
#   [3] verify the .bin: reload it, assert bitwise equality of every field,
#       decode head/tail bases of sampled records from the packed stream and
#       string-compare them against the raw FASTA text (memmem lookup of the
#       record headers; the text is folded to the stream's convention: upper
#       case, N/n and IUPAC B/K/M/R/S/W/Y -> G), and check the revcomp
#       invariant (rwords[r,i] == 3 - words[r, len+1-i] per record region).
#   [4] cut rope fragments: floor(len/k) full k-windows per record
#       (k = 20,000, the rope encoder length used across the test suite),
#       count them and report the leftover bases + the sliding-window count
#       (step = k/10) a RopeIndexer would produce.
#   [5] actually rope-encode ALL fragments (fwd + rc, kernel_best_v3,
#       normalize = 0) in groups of 2^15 from the RELOADED .bin: fragments are
#       k-windows of the packed bitstream, so no repacking is needed -- only
#       derived starts/stops (rebased Int32 per group) into a word-slice
#       upload. The embeddings are consumed on device (statistics only).
#
#       ERRATUM (found by humanref_v2.jl): the v3 packed layout stores each
#       read's bases REVERSED, so kernel addressing is only valid for windows
#       that are themselves packed reads; a sub-record k-window [a,b] of a
#       record [s,e] must be MIRRORED to [s+e-b, s+e-a] for the fwd stream
#       (the rc stream takes [a,b]). Step [5] passes the raw [a,b], so its
#       encodings -- and the mean |encoding| statistics -- correspond to the
#       REFLECTED windows, not the claimed fragments (statistically
#       indistinguishable, which is why it went unnoticed). See the GOTCHA
#       in humanref_v2.jl.
#
# Config (env overrides): HUMANREF_FASTA, HUMANREF_BIN, HUMANREF_K,
# HUMANREF_BATCH (records per parse batch).
#
# RESULTS (kau, RTX 5090, -t 16, batch_reads = 8, k = 20,000, 2026):
# the FASTA (3,147,051,578 bytes) parses in 9 batches in ~4.2-4.9 s wall
# (first-run compile included; loader task-time shares: parse ~74%, merge
# ~21%, batch scan ~5%, prefault ~1%) into 193,465,846 words fwd + rc (0.72
# GiB each); 68 sequences, 3,095,453,524 packed bases -- exactly the file's
# raw sequence character count (awk-checked): the IUPAC substitution above
# keeps every character at its position instead of dropping the 99
# B/K/M/R/S/W/Y letters. Utils.save writes the 1,547,735,411 byte .bin in
# 0.3-1.0 s; the reload is bitwise identical, and textual head/tail
# spot-checks of sampled records against the raw FASTA (folded to the same
# convention) plus the revcomp invariant PASS -- these validate the
# continuous-stream seam merging end to end. Rope statistics at k = 20,000:
# 67/68 records reach k (only the 16,569 bp mt genome falls short), giving
# 154,737 full k-fragments (3,094,740,000 bases, 713,524 leftover in record
# tails); step = k/10 sliding windows would give 1,547,081 index entries. All
# 154,737 fragments rope-encode fwd+rc from the RELOADED .bin in ~1.1-1.2 s
# (~130-140k frags/s, 5 groups of 2^15, derived starts/stops into word-slice
# uploads, no repacking; mean |encoding| 2.392e5 fwd / 2.396e5 rc, peak
# device pool ~0.9 GiB). Total ~7 s plus julia/CUDA startup.
# ==============================================================================

include(joinpath(@__DIR__, "newflow_v2.jl")) # FastaBatchesV2 (the E1/E2 parse
# engine) + the v3 rope encode kernels (+ transitively: newflow.jl ->
# newfasta.jl -> newencoder2bit.jl; RotorMap.Utils save/load)

# ==============================================================================
# IUPAC substitution (this script's deviation from the loaders' default): the
# shared pack LUT accepts only ACGTN/acgtn (N/n -> G) and silently DROPS every
# other letter (LUT 0xff), so positions shift by the dropped letters before
# them. GRCh38.p14 contains 99 such letters (B/K/M/R/S/W/Y); this script
# substitutes them to G instead -- the same treatment as N/n -- so every raw
# sequence character keeps its position (packed bases == raw sequence
# characters). Both parse paths must see the substitution: _PACK_LUT is read
# per byte by the scalar fallback, and _PAIR_LUT (derived from it at include
# time) gates the SWAR fast path -- patch the former, rebuild the latter.
# ==============================================================================
const _IUPAC = (UInt8('B'), UInt8('K'), UInt8('M'), UInt8('R'),
                UInt8('S'), UInt8('W'), UInt8('Y'))
for c in _IUPAC
    _PACK_LUT[Int(c) + 1] = 0x02  # upper case -> G, like N/n
    _PACK_LUT[Int(c) + 33] = 0x02 # lower case ('a' = 'A' + 32; index = code + 1)
end
let # rebuild every _PAIR_LUT entry from the patched _PACK_LUT (same entry
    # layout as newfasta_v2.jl: bits 0-1/2-3 = the two base codes, bit 4 =
    # both bytes are bases, bit 5 = the pair contains N/n, 0x00 = invalid;
    # an invalid pair would only force the byte-exact fallback, but keep the
    # LUTs consistent so the fast path covers the letters too)
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
using Serialization
using Mmap
using ProgressMeter
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders
using Base.Threads

const FASTA = get(ENV, "HUMANREF_FASTA",
                  "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
const BINFILE = get(ENV, "HUMANREF_BIN", string(splitext(FASTA)[1], ".bin"))
const K = parse(Int, get(ENV, "HUMANREF_K", "20000"))       # rope fragment length
const BATCH_READS = parse(Int, get(ENV, "HUMANREF_BATCH", "8")) # batch < 2^31 bases
const GROUP = 2^15                                          # fragments per encode group

# ==============================================================================
# continuous-stream concatenation
# ==============================================================================

"""
    pend = _append_bits!(words, payload, p, vlast)

Append a batch's packed payload (trailing guard word already stripped) to a
continuous bitstream whose last word holds `p` valid low bits. The payload's
last word contributes only its `vlast` valid low bits (the rest is the loader's
zero padding and must not enter the stream); every completed word is followed
by a fresh zero tail word, so the stream always ends with `pend < 32` valid
low bits (`pend` is returned). Record bits stay contiguous across batch seams
exactly as in a whole-file `load_fasta_packed`.
"""
function _append_bits!(words::Vector{UInt32}, payload, p::Int, vlast::Int)
    isempty(payload) && return p
    pend = p
    n = length(payload)
    for j in 1:n
        q = UInt64(payload[j])
        nb = j == n ? vlast : 32 # valid bits of this word
        while nb > 0
            take = min(32 - pend, nb)
            words[end] = (words[end] | ((q & ((UInt64(1) << take) - 1)) << pend)) % UInt32
            q >>= take
            nb -= take
            pend += take
            if pend == 32 # completed word stays intact; open a fresh tail
                push!(words, UInt32(0))
                pend = 0
            end
        end
    end
    return pend
end

# ==============================================================================
# verification helpers (decode from the packed stream + raw FASTA text)
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

"code of base i (1-based within the record) from the reversed-order packed stream"
function _code(words, gstart::Int64, L::Int, i::Int)
    bit = 2 * (gstart - 1) + 2 * (L - i)
    return Int((words[(bit >> 5) + 1] >> (bit & 31)) & 3)
end

_decode_str(words, gstart::Int64, L::Int, idxs) =
    String(["ACGT"[_code(words, gstart, L, i) + 1] for i in idxs])

"fold raw FASTA text to this script's convention (upper case; N/n and the
IUPAC letters B/K/M/R/S/W/Y -> G, exactly what the packed stream holds)"
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

"decode sampled records from the packed streams and compare with the raw text"
function _spot_check(raw, starts64, stops64, heads, words, rwords, samples)
    for r in samples
        g = starts64[r]
        L = Int(stops64[r] - g + 1)
        @assert L >= 60 "sampled record $r too short for the textual check"
        head = heads[r]
        hpos = _find_pattern(raw, head, 1)
        @assert hpos > 0 "head not found in the raw file: $head"
        txt_head = _collect_bases(raw, hpos + sizeof(head), 60, +1)
        npos = _find_pattern(raw, "\n>", hpos + sizeof(head)) # next record start
        txt_tail = _collect_bases(raw, (npos == 0 ? length(raw) + 1 : npos) - 1, 60, -1)
        n = 60
        @assert _fold(txt_head[1:n]) == _decode_str(words, g, L, 1:n) "head bases mismatch (record $r)"
        @assert _fold(reverse!(txt_tail[1:n])) == _decode_str(words, g, L, L-n+1:L) "tail bases mismatch (record $r)"
        for i in vcat(1:n, L-n+1:L) # rwords = complemented words, reversed per record
            @assert _code(rwords, g, L, L + 1 - i) == 3 - _code(words, g, L, i) "revcomp invariant broken (record $r, base $i)"
        end
    end
    return nothing
end

# ==============================================================================
# rope fragments + encode
# ==============================================================================

"full k-windows per record of the global bitstream (derived starts/stops, no repack)"
function _build_fragments(starts64, stops64, k::Int)
    fs = Int64[]
    fe = Int64[]
    nrec = 0
    for r in eachindex(starts64)
        L = Int(stops64[r] - starts64[r] + 1)
        L < k && continue
        nrec += 1
        s = starts64[r]
        for f in 1:(L ÷ k)
            push!(fs, s + (f - 1) * k)
            push!(fe, s + f * k - 1)
        end
    end
    return fs, fe, nrec
end

# ERRATUM: `fs`/`fe` are passed to the kernel as-is, but sub-record k-windows
# must be MIRRORED within their record for the v3 reversed-per-read layout
# (fwd: [s+e-b, s+e-a]; rc: [a,b] -- see humanref_v2.jl). The encodings
# produced here correspond to the reflected windows, not `fs`/`fe`; fine for
# the magnitude statistics this script reports, wrong for downstream use.
"rope-encode fragments [g0:g1] (fwd + rc, normalize = 0); returns the norm sums"
function _encode_group(re, words, rwords, fs, fe, g0::Int, g1::Int)
    sgb = Int(fs[g0])
    geb = Int(fe[g1])
    wlo = ((2 * (sgb - 1)) >> 5) + 1        # word holding the group's first bit
    whi = min(((2 * (geb - 1)) >> 5) + 2, length(words)) # + one guard word
    off16 = 16 * (wlo - 1)                  # bases the slice start is shifted by
    ls = Int32.(view(fs, g0:g1) .- off16)   # group-local Int32 offsets
    le = Int32.(view(fe, g0:g1) .- off16)
    ng = length(ls)
    dw = CuArray(view(words, wlo:whi))
    drw = CuArray(view(rwords, wlo:whi))
    ds = CuArray(ls)
    de = CuArray(le)
    dest = CUDA.zeros(ComplexF32, re.m, 4^re.c, ng)
    dn = CUDA.zeros(Float32, re.m, ng)
    encode_batch_cuda_all_best_v3!(dest, dn, re, dw; starts = ds, stops = de, normalize = 0)
    rdest = CUDA.zeros(ComplexF32, re.m, 4^re.c, ng)
    rdn = CUDA.zeros(Float32, re.m, ng)
    encode_batch_cuda_all_best_v3!(rdest, rdn, re, drw; starts = ds, stops = de, normalize = 0)
    CUDA.synchronize()
    sf = Float64(sum(Array(dn)))
    sr = Float64(sum(Array(rdn)))
    @assert isfinite(sf) && isfinite(sr) && sf > 0 && sr > 0 "non-finite encodings in group"
    return sf, sr
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
    @printf("== humanref.jl: %s -> packed .bin ==\n", basename(FASTA))
    @printf("source: %s (%s bytes)\n", FASTA, _comma(filesize(FASTA)))
    @printf("threads: %d   gpu: %s   k: %d   batch_reads: %d\n\n",
            nthreads(), CUDA.name(device()), K, BATCH_READS)
    @assert isfile(FASTA) "FASTA not found: $FASTA"

    # ---- [1] stream + 2-bit pack (FastaBatchesV2) ----------------------------
    @printf("[1/5] streaming + 2-bit packing (FastaBatchesV2, revcomp, swar)\n")
    words = UInt32[0] # seed tail word (absorbed by the first funnel merge)
    rwords = UInt32[0]
    sizehint!(words, filesize(FASTA) ÷ 4 + 16) # bases <= file bytes: no growth copies
    sizehint!(rwords, filesize(FASTA) ÷ 4 + 16)
    gstarts = Int64[]
    gstops = Int64[]
    heads = String[]
    acc = Int64(0)
    pend_w = 0 # pending valid bits in the last word, PER STREAM (words / rwords)
    pend_r = 0
    nbatches = 0
    it = FastaBatchesV2(FASTA; batch_reads = BATCH_READS, revcomp = true)
    prog = ProgressUnknown(desc = "records packed: ", dt = 0.5)
    _v2time_reset!()
    t_parse = @elapsed begin
        cur = iterate(it)
        while cur !== nothing
            nt, hi = cur
            nbatches += 1
            n = length(nt.starts)
            if n > 0
                bi = Int(nt.stops[end])
                @assert bi < (Int64(1) << 31) "batch $nbatches exceeds the 2^31-base Int32 offset limit: lower batch_reads"
                m = length(nt.words)
                @assert m == cld(2 * bi, 32) + 1 && length(nt.rwords) == m "unexpected batch stream layout (batch $nbatches)"
                vlast = ((2 * bi - 1) & 31) + 1 # valid bits of the payload's last word
                pend_w = _append_bits!(words, @view(nt.words[1:m-1]), pend_w, vlast)
                pend_r = _append_bits!(rwords, @view(nt.rwords[1:m-1]), pend_r, vlast)
                append!(gstarts, Int64.(nt.starts) .+ acc)
                append!(gstops, Int64.(nt.stops) .+ acc)
                append!(heads, nt.heads)
                acc += bi
                @assert pend_w == (2 * acc) % 32 && pend_r == pend_w "pend drift (batch $nbatches): pend_w=$pend_w pend_r=$pend_r expected $((2 * acc) % 32), bi=$bi m=$m vlast=$vlast"
                @assert length(words) == length(rwords) "stream divergence (batch $nbatches): fwd $(length(words)) rc $(length(rwords))"
            end
            update!(prog, length(gstarts))
            cur = iterate(it, hi)
        end
    end
    finish!(prog)
    pend_w > 0 && push!(words, UInt32(0)) # one fresh guard word closes each stream
    pend_r > 0 && push!(rwords, UInt32(0))
    nseqs = length(gstarts)
    nbases = acc
    @assert length(words) == cld(2 * nbases, 32) + 1 && length(rwords) == length(words) "stream size mismatch"
    @assert (isempty(gstarts) || (gstarts[1] == 1 && all(gstarts[2:end] .== gstops[1:end-1] .+ 1))) "records are not bit-contiguous"
    @printf("      %d batches in %.2f s (%.2f GB/s of FASTA)\n", nbatches, t_parse, filesize(FASTA) / t_parse / 1e9)
    @printf("      sequences: %s   bases: %s (N/n and IUPAC B/K/M/R/S/W/Y are substituted to G)\n",
            _comma(nseqs), _comma(nbases))
    @printf("      stream check: %d words fwd / %d rc, expected %d (bits %s, pend %d/%d)\n",
            length(words), length(rwords), cld(2 * nbases, 32) + 1, _comma(2 * nbases), pend_w, pend_r)
    @printf("      packed stream: %s words fwd + rc (%.2f GiB each), max record %s bases\n",
            _comma(length(words) - 1), length(words) * 4 / 2^30, _comma(Int(maximum(gstops .- gstarts .+ 1))))
    _v2time_print()

    # ---- [2] save .bin (Serialization, the Utils.jl save/load format) --------
    @printf("\n[2/5] saving %s (Serialization, Utils.save)\n", BINFILE)
    data = (format = :packed_dna_v1,
            source = basename(FASTA),
            k = K,
            n_seqs = nseqs,
            nbases = nbases,
            words = words,      # 2-bit packed, reversed base order (v3 layout)
            rwords = rwords,    # reverse complement, forward order
            starts = gstarts,   # global 1-based Int64 base offsets
            stops = gstops,
            heads = heads)      # FASTA header lines (with '>')
    t_save = @elapsed Utils.save(data, BINFILE)
    @printf("      %s bytes in %.2f s (%.0f MB/s)\n",
            _comma(filesize(BINFILE)), t_save, filesize(BINFILE) / t_save / 1e6)

    # ---- [3] verify ----------------------------------------------------------
    @printf("\n[3/5] verifying the .bin\n")
    t_load = @elapsed data2 = Utils.load(BINFILE)
    ok_fields = data2.format === :packed_dna_v1 && data2.words == words &&
                data2.rwords == rwords && data2.starts == gstarts &&
                data2.stops == gstops && data2.heads == heads
    @printf("      reload: %.2f s   fields bitwise ==: %s\n", t_load, ok_fields ? "PASS" : "FAIL")
    @assert ok_fields "reloaded .bin differs"
    data = words = rwords = gstarts = gstops = heads = nothing # free the originals
    GC.gc()
    raw = open(FASTA, "r") do io
        Mmap.mmap(io)
    end
    samples = sort!(unique!(filter(r -> 1 <= r <= data2.n_seqs,
                                   vcat(1, 1 .+ BATCH_READS .* [1, 2, 4, 6], data2.n_seqs))))
    t_check = @elapsed _spot_check(raw, data2.starts, data2.stops, data2.heads, data2.words, data2.rwords, samples)
    @printf("      textual head/tail spot-check + revcomp invariant on %d sampled records: PASS (%.2f s)\n",
            length(samples), t_check)
    raw = nothing

    # ---- [4] rope fragments ---------------------------------------------------
    @printf("\n[4/5] rope fragments (k = %s, non-overlapping full k-windows per record)\n", _comma(K))
    fs, fe, nrec_k = _build_fragments(data2.starts, data2.stops, K)
    nfrag = length(fs)
    lens = data2.stops .- data2.starts .+ 1
    covered = nfrag * K
    stp = K ÷ 10
    nslide = sum((L - K) ÷ stp + 1 for L in lens if L >= K; init = 0)
    @printf("      records >= k: %d/%d (min %s, max %s, mean %s bases)\n",
            nrec_k, data2.n_seqs, _comma(Int(minimum(lens))), _comma(Int(maximum(lens))),
            _comma(round(Int, sum(lens) / data2.n_seqs)))
    @printf("      fragments: %s (%s bases covered, %s leftover in record tails)\n",
            _comma(nfrag), _comma(covered), _comma(data2.nbases - covered))
    @printf("      (info: sliding windows with step = k/10 would give %s index entries)\n", _comma(nslide))

    # ---- [5] rope encode fwd + rc from the RELOADED .bin ----------------------
    @printf("\n[5/5] rope encode fwd+rc (RopeEncoder(k=%d, s=8, m=4, c=4), normalize=0, groups of %s)\n",
            K, _comma(GROUP))
    re = RopeEncoder(k = K, s = 8, m = 4, c = 4)
    _warm_encode(re)
    t_enc = 0.0
    sf_all = 0.0
    sr_all = 0.0
    g = 0
    for g0 in 1:GROUP:nfrag
        g1 = min(g0 + GROUP - 1, nfrag)
        g += 1
        t = @elapsed (sf, sr) = _encode_group(re, data2.words, data2.rwords, fs, fe, g0, g1)
        t_enc += t
        sf_all += sf
        sr_all += sr
        @printf("      group %d: %s fragments  %.3f s\n", g, _comma(g1 - g0 + 1), t)
    end
    @printf("      encoded %s fragments (x2 for rc) in %.2f s = %s frags/s\n",
            _comma(nfrag), t_enc, _comma(round(Int, nfrag / t_enc)))
    @printf("      mean |encoding| per fragment: fwd %.3e   rc %.3e\n", sf_all / (re.m * nfrag), sr_all / (re.m * nfrag))
    @printf("      device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)

    @printf("\nALL DONE in %.1f s (parse %.2f + save %.2f + verify %.2f + fragments/encode %.2f)\n",
            t_parse + t_save + t_load + t_check + t_enc, t_parse, t_save, t_load + t_check, t_enc)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
