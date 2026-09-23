# =============================================================================
# build.jl -- the INDEX flow: rope-encode a REFERENCE fasta into a matrix of
# REAL [Re; Im] vectors saved as a Julia-serialized .bin next to the fasta
# (from legacy test/indexflowreal.jl, verbatim).  One thread parses the whole
# file sequentially (records are huge -- human chromosomes) and slides a
# k-window with step kstep over every record; fwd + revcomp windows are
# encoded with the production kernel and saved fp16 and/or as transposed e4m3
# fp8 database columns (the production mapping layout).
#
# CHANGES vs the legacy file: `using RotorMap*` lines dropped (the package
# moved to legacy/); `save(/load` are now bare save/load from
# common/dna.jl; the synthetic test fasta builder lives in common/testref.jl
# (same names: _ensure_test_fasta / _if_test_fasta_path).
#
# REQUIRES (include beforehand, canonical order): common/dna.jl,
#   common/testref.jl, common/util.jl, fasta/pack.jl + fasta/loader.jl +
#   fasta/reader.jl, encode/ropeencoder.jl + encode/reference.jl (_frag_codes)
#   + encode/kernel.jl + encode/stream.jl, gemm/fp8_convert.jl.
# ENTRY POINT: `julia --project=RotorMap RotorMap/index/build.jl [mode]` with
#   modes gen|test|save|verify|bench|all (dispatch at the bottom, kept).
# =============================================================================
# ==============================================================================
# indexreal.jl -- the INDEX flow: rope-encode a REFERENCE fasta (the human
# genome) into a matrix of REAL [Re; Im] vectors saved as a Julia-serialized
# .bin next to the fasta.  Unlike ropeflowreal (which consumes short reads
# through the multi-part parallel fasta_reads), here ONE thread parses the
# whole file sequentially -- records are huge (human chromosomes), so the
# file is never split into byte ranges -- and slides a k-window with step
# kstep over every record.
#
# SPEC (prompts/indexflowreal.md + user clarifications):
#   * single thread = no file splitting / no parallel parsing; the GPU encode
#     still runs in its own pipeline task (t = 16 threads are fine for that);
#   * k (default 20,000) is the fragment length; kstep (default k/10) the
#     sliding-window step; the LAST WINDOW IS FULLY INSIDE the record
#     (offsets 1, 1+kstep, ... while start + k - 1 <= len; a record shorter
#     than k contributes nothing);
#   * every fragment carries its LOCATION in the meta: the record's verbatim
#     header and the 1-based window start in the record's sequence-character
#     space (every byte except \n/\r is a position; junk/IUPAC -> G but keeps
#     its position -- the fastareads_v3 convention);
#   * the REVERSE COMPLEMENT of every window is encoded too ("complementary
#     reversed DNA fragments") and appended to the SAME output matrix (x2
#     rows).  The rc fragment words are materialized on the GPU (one tiny
#     funnel-shift kernel, `kernel_revcomp_words!`) and rope-encoded by the
#     UNCHANGED kernel_rope_frag_real -- i.e. an rc fragment is encoded like
#     any fragment, phases running from its own (reversed) start; this is the
#     same rc convention as the v2 rc stream (newfasta_v2 / humanref).  Rows
#     per batch: 1:nb = fwd, nb+1:2nb = revcomp of the same fragments; the
#     `strand` meta vector (0 = fwd, 1 = rc) makes rows self-describing;
#   * the embeddings are collected into ONE matrix and saved as a
#     Julia-serialized NamedTuple next to the fasta.  TWO save layouts
#     (INDEX_FP8, default TRUE = the production mapping layout):
#     fp8  -> format "indexreal.fp8.v1", file <fasta>.indexreal_fp8.bin:
#             embeds8 (2*m*4^c, 2*n_frag) Float8_E4M3FN -- the database
#             COLUMNS the fp8 mapping flow (e2ehuman.jl) resolves against,
#             quantized from the fp16 stream at save time (the exact
#             F8.(Float32.(.)) conversion the GPU-side index_to_f8 applied,
#             so the mapping results are bitwise identical).  A load is a
#             pure H2D: no host-side transpose (12.7 GiB -> 6.3 GiB), which
#             matters double on a PCIe Gen1 x16 link (~4 GB/s); a GPU
#             transpose + H2D of the fp16 rows costs ~9 s per mapping run.
#     fp16 -> format "indexreal.v1", file <fasta>.indexreal.bin:
#             embeds (2*n_frag, 2*m*4^c) {Float16,Float32} rows -- the
#             classic layout (e2ehuman16.jl's fp16 mapping flow consumes it;
#             INDEX_FP8=0 selects it for save)
#     both carry norms (m, 2*n_frag) Float32, heads (verbatim '>' headers),
#     starts (1-based), strand (0/1) -- order does not matter, the
#     descriptions are saved along.
#
# PIPELINE (two channel stages; the reader replaces ropeflowreal's
# fasta_reads + gather, the encode stage is its twin plus the rc transform):
#
#   single-threaded reader task   walk records (memchr line-start '>'), SWAR-
#     | Channel{(W,nb) words,     pack each record ONCE into forward 2-bit
#     | heads, starts}            words, slice k-windows by funnel shifts into
#     v                           pinned staging buffers (batch_size columns)
#   encode task                   H2D -> GPU revcomp words -> kernel_rope_frag_
#     | Channel{RefRopeRealBatch} real on BOTH word sets -> D2H -> batch
#     v                           (embeds (2nb, 2m4^c), norms, heads, starts,
#   consumer                      strand, first)
#
# Backpressure, buffer recycling, err_out/tasks_out and the early-close
# teardown contract are ropeflowreal's verbatim; the kernel itself is shared
# unchanged (this file includes ropeflowreal.jl).
#
# Run modes (ARGV[1]; ARGS[2] optionally overrides the fasta):
#   gen    (re)generate the synthetic edge-case reference fasta (cached)
#   test   correctness: window extraction vs an independent naive serial
#          parser (k = 50/2001/20000, several ksteps), GPU revcomp words vs
#          the codes reference (tail residues), end-to-end stream vs the CPU
#          rope reference (fwd + rc rows, fp32/fp16, batch layouts, meta
#          equality), save/load .bin roundtrips (fp16, fp32 AND the fp8
#          mapping layout), early close, empty/header-only files, plus the
#          human fasta PREFIX (first record) end-to-end
#   save   production run on the human reference: build the matrices and save
#          the index -- fp8 transposed-e4m3 mapping layout by DEFAULT
#          (INDEX_FP8=0 for the classic fp16 row layout; INDEX_FP16=0
#          for an fp32 stream)
#   verify reload the .bin and check it against the fasta independently:
#          counts, full structural meta check (strand-block layout, per-window
#          starts), sampled rows vs the float64 CPU rope reference (both
#          formats; --bin8 targets the fp8 mapping file)
#   bench  timings on the human fasta (count pass, reader drain, full stream)
#   all    gen + test + bench (save/verify are explicit production commands)
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16, GRCh38.p14 = 3.1 GB,
# 68 records, 3,095,453,524 seq chars, k = 20,000, kstep = 2,000,
# s = 8, m = 4, c = 4 -> row dim 2*m*4^c = 2048, batch_size = 2^13):
#
#   count pass (records + windows)                       0.292 s   10.76 GB/s
#   reader drain (walk + pack + slice, single thread)    2.178 s    1.44 GB/s
#   full stream, no-op consumer (H2D+rc+2xkernel+D2H)   18.430 s    0.17 GB/s
#
# `save` on the full human reference: 1,547,081 windows (x2 rows = 3,094,162)
# -- exactly the count humanref.jl predicted for step k/10 -- encoded in
# ~17.7 s, the 11.88 GiB fp16 .bin serialized in 2.3 s.  `verify` reloads it
# and passes the independent count pass, the full structural meta check and
# 24 sampled rows vs the float64 CPU rope reference.
#
# fp8 save (executed on kau, the new default): same stream (36.4 s including
# the per-batch fused transpose + fp16->e4m3 quantize), the 5.98 GiB
# indexreal_fp8.bin written in 1.1 s.  `verify --bin8` passes (count,
# structural meta, 24 dequantized rows vs the CPU reference with e4m3
# tolerances); `test` passes the fp8 roundtrip + fp16/fp32/fp8 agreement and
# the human-prefix fp8 rows.  The mapping flow (e2ehuman.jl) loads it with a
# pure H2D: database build 8.9 s -> 0.6 s, and the mapping numbers are
# bitwise the load-time-quantized run's (124,486/131,072 at err = 0.05).
#
# Correctness (all on kau, julia -t 16): A) window extraction vs an
# independent naive serial parser (k = 50/2001/20000 x kstep 1..k, exact word
# equality); B) revcomp words: host funnel twin + GPU kernel vs the codes
# reference (tail residues r = k mod 16, fragments and batches); C) end-to-end
# stream on the synthetic edge-case file (batch_size 1/3/2^13/10^6, fp32+fp16,
# normalize 0/1, kstep = k, a second encoder config s=5/m=1/c=4; deterministic
# single-reader order, unit energy, sampled rows vs the CPU reference for BOTH
# strands); D) build_index_matrix + .bin roundtrip (bitwise fields, fp16 and
# fp32); E) early-close teardown + empty/header-only files; F) the human
# chr1 prefix (124,469 windows x2) end-to-end with sampled independent checks.
# ALL PASS.
#
# GOTCHAS hit on the way:
#   * a FULL bit-reversal also mirrors the bits WITHIN each 2-bit lane
#     (codes 1<->2 swap!) -- the stream reversal must be the 2-bit swap
#     network MINUS the 1-bit round (_lanerev32);
#   * the window tail word must be masked to the zero-padded _emit3!
#     convention (the funnel otherwise carries the record's next bases);
#   * `copyto!(dest, offset, src, 1, n)` with mismatched column-major heights
#     SCRAMBLES the layout silently (the ropeflow_v3/ropeflowreal bench sinks
#     are scrambled -- timing-only); download device batches into
#     matching-shape host matrices and vcat/hcat instead.
# ==============================================================================

# (legacy include of ropeflowreal.jl replaced by the entry script's canonical
# include order: encode/{ropeencoder,reference,kernel,stream}.jl provide
# kernel_rope_frag_real, encode_frag_real_batch!, _ref_rope_frag_real,
# _frag_codes; fasta/reader.jl provides _next_rec3/_LUT3; common/util.jl
# provides _timed_min/_warm_cache/_reps; common/testref.jl provides
# _v3_data_dir)

using Random
using CUDA
using Printf
using Serialization
using Mmap
using ProgressMeter
using DLFP8Types: Float8_E4M3FN # the fp8 save layout's element type (F8 in
# the gemmtopkfp8/flowtopkfp8 scripts)
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides; ARGS[2] can override the fasta per mode)
# ------------------------------------------------------------------------------
const IF_FASTA = get(ENV, "INDEX_FASTA",
                     "/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.fna")
_if_binfile(fasta) = get(ENV, "INDEX_BIN",
                         string(splitext(fasta)[1], ".indexreal.bin"))
_if_fp8file(fasta) = get(ENV, "INDEX_FP8BIN",
                         string(splitext(fasta)[1], ".indexreal_fp8.bin"))
const IF_K = parse(Int, get(ENV, "INDEX_K", "20000"))          # fragment length
const IF_KSTEP = parse(Int, get(ENV, "INDEX_KSTEP", string(IF_K ÷ 10)))
const IF_FP16 = parse(Bool, get(ENV, "INDEX_FP16", "true"))    # stream eltype
const IF_FP8 = parse(Bool, get(ENV, "INDEX_FP8", "true"))      # default save:
# the transposed-e4m3 mapping layout (false = the classic fp16 row layout)
const IF_NORM = parse(Int, get(ENV, "INDEX_NORM", "0"))        # normalize mode
const IF_BATCH = parse(Int, get(ENV, "INDEX_BATCH", "8192"))   # frag/batch
# the RopeEncoder config for save/bench (e2ehuman reads s/m/c back from the
# .bin, so the whole mapping flow follows whatever was saved here)
const IF_S = parse(Int, get(ENV, "INDEX_S", "5"))              # s-mer length
const IF_M = parse(Int, get(ENV, "INDEX_M", "1"))              # rotations per strand
const IF_C = parse(Int, get(ENV, "INDEX_C", string(IF_S)))     # c-mer length (<= s;
#                                                               defaults to s)
const IF_TEST_RECORDS = parse(Int, get(ENV, "INDEX_TEST_RECORDS", "1"))
const IF_REPS = parse(Int, get(ENV, "INDEX_REPS", "1"))
const IF_FORMAT = "indexreal.v1"
const IF_FP8_FORMAT = "indexreal.fp8.v1"

# ==============================================================================
# The single-threaded reference reader: record walk -> record pack -> windows
# ==============================================================================

"""
Pack ONE record's sequence bytes [seq_lo, seq_hi] into forward 2-bit words
(the fastareads_v3 convention: base j -> word (j-1)>>4 + 1, bit 2*((j-1)&15);
every byte except \\n/\\r is a base via _LUT3, junk/IUPAC -> G keeping its
position).  The record is SWAR-collected into `sc` (grown as needed, reused
across records) and the groups folded into `cld(nbases, 16) + 1` words -- the
extra ZERO GUARD word makes funnel-shift window reads safe.  Returns
(words, nbases).
"""
function _pack_record!(sc::Vector{UInt16}, raw::Vector{UInt8}, seq_lo::Int, seq_hi::Int)
    seq_lo > seq_hi && return (zeros(UInt32, 1), 0)
    span = seq_hi - seq_lo + 1
    need = (span >> 3) + 2 # max groups: 8 codes each, + room for the pending fold
    length(sc) < need && resize!(sc, need)
    # k = span can never be reached early (<= span non-newline bytes exist)
    (sc, ng, pending, pnbits, nbases) =
        _collect3!(sc, 0, UInt64(0), 0, 0, pointer(raw, seq_lo), span, span)
    words = zeros(UInt32, cld(nbases, 16) + 1) # zero tail + guard word
    _groups_to_words!(words, sc, ng, pending, pnbits, nbases)
    return (words, nbases)
end

# Fold _collect3!'s 16-bit groups (+ pending sub-group tail) into forward
# UInt32 words; the same packing math as fastareads_v3's _emit3! for an
# arbitrary kept length `nbases`.
function _groups_to_words!(words::Vector{UInt32}, sc::Vector{UInt16}, ng::Int,
                           pending::UInt64, pnbits::Int, nbases::Int)
    npart = pnbits >> 1
    if npart > 0 # fold the sub-group tail into one extra group
        sc[ng+1] = pending % UInt16 # capacity guaranteed by _pack_record!'s pre-size
    end
    npair = nbases >> 4 # complete words = two consecutive 8-code groups
    @inbounds for j in 1:npair
        words[j] = UInt32(sc[2*j-1]) | UInt32(sc[2*j]) << 16
    end
    r = nbases & 15
    if r > 0
        gi = 2 * npair + 1
        rf = r >> 3 # complete 8-code groups in the partial word (0 or 1)
        rt = r & 7  # remaining codes (low bits of the next group)
        w = UInt32(0)
        @inbounds for t in 0:rf-1
            w |= UInt32(sc[gi+t]) << (16 * t)
        end
        if rt > 0
            # NB: Julia's << binds TIGHTER than *, hence the explicit parens
            w |= (UInt32(sc[gi+rf]) & ((UInt32(1) << (2 * rt)) - 1)) << (16 * rf)
        end
        words[npair+1] = w
    end
    return nothing
end

"""
Slice the k-window starting at 0-based base offset `off0` out of a packed
record (with trailing zero guard word) into `dst` (cld(k, 16) words): the
window is a plain funnel shift of the record's bit stream; the last word is
masked to the window's tail so the result is bitwise the forward packing of
the window's bases with a ZERO-PADDED tail (the `_emit3!` / kernel input
convention; the rope kernel itself never reads the padding bits).
"""
function _window_words!(dst::Vector{UInt32}, recwords::Vector{UInt32}, off0::Int, k::Int)
    W = cld(k, 16)
    length(dst) == W || throw(ArgumentError("dst must hold cld(k, 16) words"))
    r = k & 15
    tailmask = r == 0 ? typemax(UInt32) : (UInt32(1) << (2 * r)) - UInt32(1)
    @inbounds for w in 0:W-1
        b = 2 * off0 + 32 * w    # the window word's first bit in the record
        q = (b >> 5) + 1         # record word holding it
        sh = b & 31              # bit offset inside that word
        v = sh == 0 ? recwords[q] :
            (recwords[q] >>> sh) | (recwords[q+1] << (UInt32(32) - UInt32(sh)))
        dst[w+1] = w == W - 1 ? v & tailmask : v
    end
    return nothing
end

"""
    _each_ref_window!(f, raw; k, kstep, sc, wnd, max_records)

Walk the fasta bytes `raw` record by record (SINGLE thread, no file splitting:
memchr for line-start '>' record bounds, verbatim header lines) and call
`f(header, start)` for every k-window at offsets 1, 1+kstep, ... that is FULLY
INSIDE the record (`wnd` then holds the window's forward-packed words; it is
reused between calls).  Records shorter than k contribute nothing; `start` is
1-based in the record's sequence-character space.  Returns the number of
records seen.  This is the production reader's engine -- and the direct CPU
test path.
"""
function _each_ref_window!(f, raw::Vector{UInt8}; k::Int, kstep::Int,
                           sc::Vector{UInt16} = Vector{UInt16}(undef, 4096),
                           wnd::Vector{UInt32} = Vector{UInt32}(undef, cld(k, 16)),
                           max_records::Int = typemax(Int))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    kstep >= 1 || throw(ArgumentError("kstep must be >= 1"))
    length(wnd) == cld(k, 16) || throw(ArgumentError("wnd must hold cld(k, 16) words"))
    n = length(raw)
    nrec = 0
    s = _next_rec3(raw, 1, n) # junk before the first record is discarded
    while s <= n
        nrec += 1
        nrec > max_records && break
        h = s # header line: [s, h) up to the first newline (CR tolerated)
        while h <= n && raw[h] != UInt8('\n')
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == UInt8('\r')) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n) # next line-start '>' or n+1 (EOF)
        (words, nbases) = _pack_record!(sc, raw, h + 1, e - 1)
        if nbases >= k
            header = String(raw[s:hend])
            nwin = (nbases - k) ÷ kstep + 1 # last window fully inside the record
            for wi in 0:nwin-1
                off0 = wi * kstep
                _window_words!(wnd, words, off0, k)
                f(header, off0 + 1)
            end
        end
        s = e
    end
    return nrec
end

# count sequence characters (non-newline bytes) in [lo, hi]
function _count_bases(raw::Vector{UInt8}, lo::Int, hi::Int)
    cnt = 0
    @inbounds for p in lo:hi
        c = raw[p]
        (c == UInt8('\n') || c == UInt8('\r')) || (cnt += 1)
    end
    return cnt
end

"""
    _count_ref_windows(file; k, kstep, max_records) -> (nwin, nrec, nkept)

First pass of the production flow: walk the file's records and count the
sliding windows WITHOUT packing anything (cheap text scan) -- the collected
matrices can then be preallocated to their exact size.  Applies the same
window rule as `_each_ref_window!`.
"""
function _count_ref_windows(file::String; k::Int, kstep::Int,
                            max_records::Int = typemax(Int))
    raw = filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end
    n = length(raw)
    nwin = nrec = nkept = 0
    s = _next_rec3(raw, 1, n)
    while s <= n
        nrec += 1
        nrec > max_records && break
        h = s
        while h <= n && raw[h] != UInt8('\n')
            h += 1
        end
        e = _next_rec3(raw, s + 1, n)
        nbases = _count_bases(raw, h + 1, e - 1)
        if nbases >= k
            nkept += 1
            nwin += (nbases - k) ÷ kstep + 1
        end
        s = e
    end
    return (nwin, nrec, nkept)
end

"""
    _record_spans(raw; max_records) -> (spans::Dict{String,(lo,hi)}, order)

Map every record's verbatim header to its sequence byte span -- used by the
verification paths to locate windows independently of the production walk.
"""
function _record_spans(raw::Vector{UInt8}; max_records::Int = typemax(Int))
    spans = Dict{String,Tuple{Int,Int}}()
    order = String[]
    n = length(raw)
    nrec = 0
    s = _next_rec3(raw, 1, n)
    while s <= n
        nrec += 1
        nrec > max_records && break
        h = s
        while h <= n && raw[h] != UInt8('\n')
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == UInt8('\r')) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n)
        header = String(raw[s:hend])
        spans[header] = (h + 1, e - 1)
        push!(order, header)
        s = e
    end
    return (spans, order)
end

# ==============================================================================
# Reverse-complement fragment words (complement codes + reversed base order)
# ==============================================================================

# reverse the sixteen 2-bit code lanes WITHOUT touching the bits inside each
# lane (a full bit-reversal would also mirror every code: 1<->2!) -- the 2-bit
# swap network minus the 1-bit round
_lanerev32(x::UInt32) = let
    x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333)
    x = ((x & 0x0f0f0f0f) << 4) | ((x >> 4) & 0x0f0f0f0f)
    x = ((x & 0x00ff00ff) << 8) | ((x >> 8) & 0x00ff00ff)
    (x << 16) | (x >> 16)
end

"""Codes-based revcomp reference: complement + reverse the base codes, pack."""
function _revcomp_words_ref(words::AbstractVector{UInt32}, k::Int)
    codes = _frag_codes(words, k)
    rc = Vector{UInt8}(undef, k)
    @inbounds for j in 1:k
        rc[j] = 0x03 - codes[k+1-j]
    end
    return _ref_forward_pack(rc, k)
end

"""Host twin of `kernel_revcomp_words!` (the same funnel + lane-reverse math)."""
function _revcomp_words_funnel(words::Vector{UInt32}, k::Int)
    W = cld(k, 16)
    sh = 32 * W - 2 * k # left shift aligning the k bases to the word end
    cw = [~w for w in words] # complement every 2-bit lane (c -> 3-c = ~c)
    rc = Vector{UInt32}(undef, W)
    for p in 1:W
        i = W + 1 - p # rc word p reads the aligned stream at word W+1-p
        x = cw[i]
        v = sh == 0 ? x :
            (x << sh) | (i > 1 ? (cw[i-1] >>> (32 - sh)) : UInt32(0))
        rc[p] = _lanerev32(v)
    end
    return rc
end

"""
One thread per output word: fragment f's revcomp words from its forward words.
`shbits = 32*W - 2k` (0..30, even) left-aligns the k complemented bases to the
word-array end; reversing that aligned stream = reverse the word order and
reverse the sixteen 2-bit lanes of each word (`_lanerev32`, lane contents
preserved) -- the tail lanes stay zero (the alignment pushed the complemented
zero padding out of the top).  See `_revcomp_words_funnel` for the host twin.
"""
function kernel_revcomp_words!(rcw, dnas, W::Int32, n::Int32, shbits::UInt32)
    g = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x # 1 .. W*n
    g > W * n && return
    f = (g - Int32(1)) ÷ W + Int32(1) # fragment
    p = (g - Int32(1)) % W + Int32(1) # output word position
    i = W + Int32(1) - p              # source (aligned) word
    x = ~dnas[(f - Int32(1)) * W + i] # complement every 2-bit lane (c -> 3-c)
    v = shbits == UInt32(0) ? x :
        (x << shbits) | ((i > Int32(1) ?
                          ~dnas[(f - Int32(1)) * W + i - Int32(1)] :
                          UInt32(0)) >>> (UInt32(32) - shbits))
    @inbounds rcw[g] = _lanerev32(v)
    return
end

"""`dnas` (n * cld(k,16) forward words) -> `drc` (n * cld(k,16) revcomp words)."""
function revcomp_frag_words!(drc::CuVector{UInt32}, dnas::CuVector{UInt32},
                             k::Int, n::Int)
    W = cld(k, 16)
    @assert length(dnas) == n * W && length(drc) == n * W
    total = W * n
    total == 0 && return drc
    shbits = UInt32(32 * W - 2 * k)
    threads = 256
    @cuda blocks = cld(total, threads) threads = threads kernel_revcomp_words!(
        drc, dnas, Int32(W), Int32(n), shbits
    )
    return drc
end

# ==============================================================================
# The streamed flow (reader -> encode -> consumer), ropeflowreal's topology
# ==============================================================================

"""
One streamed batch of reference-window rope encodings.  Row layout (2*nb rows
for nb fragments): rows 1:nb are the FORWARD windows of fragments
`first .. first+nb-1`, rows nb+1:2nb their REVERSE COMPLEMENTS (`strand[r]`:
0 = fwd, 1 = revcomp).  `embeds[r, :]` is the FIXED split layout [Re; Im]
(2*m*4^c columns, as ropeflowreal).  `norms[:, r]` the kernel's per-copy
norms.  `heads[r]`/`starts[r]`: the fragment's record header (verbatim, with
'>') and 1-based window start in the record's sequence-character space -- the
LOCATION meta that makes the saved matrix order-independent.  `first` is the
1-based global index of the batch's first fragment (pair).  Fresh host
matrices -- retain freely.
"""
struct RefRopeRealBatch{T<:Union{Float16,Float32}}
    embeds::Matrix{T}          # (2*nb, 2*m*4^c), [Re; Im] split layout
    norms::Matrix{Float32}     # (m, 2*nb)
    heads::Vector{String}      # 2*nb
    starts::Vector{Int}        # 2*nb
    strand::Vector{UInt8}      # 2*nb: 0 = fwd, 1 = revcomp
    first::Int
end

# task-level error router (ropeflowreal's contract, own log prefix)
function _indexflow_error(err, err_out::Ref{Any}, chs...; who::String)
    if err isa Base.InvalidStateException
        return true
    end
    println(stderr, "indexreal: $(who) failed: ",
            sprint(showerror, err, catch_backtrace()))
    flush(stderr)
    err_out[] === nothing && (err_out[] = err)
    for ch in chs
        try
            close(ch)
        catch
        end
    end
    return false
end

"""
    ref_rope_real_stream(re::RopeEncoder, file::String; k = re.k,
                         kstep = max(k ÷ 10, 1), batch_size = 2^13,
                         normalize = 0, fp16 = true, in_cap = 2, out_cap = 2,
                         progress = false, max_records = typemax(Int),
                         err_out = Ref{Any}(nothing),
                         tasks_out = Ref{Vector{Task}}(Task[]))
                         -> Channel{RefRopeRealBatch}

Stream-rope-encode every sliding k-window (step `kstep`) of every record of
the REFERENCE fasta `file`, plus its reverse complements.  ONE thread walks
the file sequentially (records are never split -- they can be entire
chromosomes), packs each record once into forward 2-bit words and slices
k-windows into pinned staging buffers; the encode task uploads each batch,
materializes the revcomp words on the GPU (`revcomp_frag_words!`), runs
`kernel_rope_frag_real` on both word sets and emits
`RefRopeRealBatch(embeds (2nb, 2*m*4^c), norms (m, 2nb), heads, starts,
strand, first)` with the last batch partial.  `k` must equal the encoder
baseline `re.k`; `kstep >= 1`; the last window of a record is fully inside
it.  `fp16 = true` (default) makes `embeds::Matrix{Float16}` (converted
in-kernel; norms stay Float32).  `max_records` bounds the walk (tests on file
prefixes).  Consume to the end or `close(ch)` early: the pipeline tears down
quietly; real failures land in `err_out[]`.  The stream order is
DETERMINISTIC (file order: records sequentially, windows per record in
sliding order; fwd/rc interleaving per batch as documented on the struct).
"""
function ref_rope_real_stream(re::RopeEncoder, file::String;
                              k::Int = re.k,
                              kstep::Int = max(k ÷ 10, 1),
                              batch_size::Int = 2^13,
                              normalize::Int = 0,
                              fp16::Bool = true,
                              in_cap::Int = 2,
                              out_cap::Int = 2,
                              progress::Bool = false,
                              max_records::Int = typemax(Int),
                              err_out::Ref{Any} = Ref{Any}(nothing),
                              tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    k >= re.s || throw(ArgumentError("k = $k < s = $(re.s): no s-mer windows"))
    k == re.k || throw(ArgumentError("k = $k must equal the encoder baseline re.k = $(re.k)"))
    kstep >= 1 || throw(ArgumentError("kstep must be >= 1"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1"))
    normalize in (0, 1, 2, 3) || throw(ArgumentError("normalize must be 0..3"))
    T = fp16 ? Float16 : Float32
    rdim = 2 * re.m * 4^re.c # the [Re; Im] row dimension
    W = cld(k, 16)

    raw = filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end
    free = Channel{Matrix{UInt32}}(in_cap + 1) # recycled pinned staging buffers
    gathered = Channel{Tuple{Matrix{UInt32},Int,Vector{String},Vector{Int}}}(in_cap)
    out = Channel{RefRopeRealBatch}(out_cap)

    for _ in 1:(in_cap + 1)
        buf = Matrix{UInt32}(undef, W, batch_size)
        CUDA.pin(buf) # one-time page-lock, reused by every batch (fast H2D)
        put!(free, buf)
    end

    # ---- single-threaded reader: records -> windows -> pinned word batches --
    t_read = Threads.@spawn begin
        sc = Vector{UInt16}(undef, 4096)  # SWAR group scratch (grown per record)
        wnd = Vector{UInt32}(undef, W)    # one window's words (reused)
        buf = take!(free)
        nb = 0
        heads = String[]
        starts = Int[]
        try
            _each_ref_window!(raw; k, kstep, sc, wnd, max_records) do header, wstart
                copyto!(buf, nb * W + 1, wnd, 1, W) # linear fill, column-major
                push!(heads, header)
                push!(starts, wstart)
                nb += 1
                if nb == batch_size
                    put!(gathered, (buf, nb, heads, starts))
                    buf = take!(free)
                    nb = 0
                    heads = String[]
                    starts = Int[]
                end
            end
            if err_out[] === nothing # healthy end: flush the (partial) tail
                if nb > 0
                    put!(gathered, (buf, nb, heads, starts))
                else
                    put!(free, buf)
                end
            end # on a reader error the tail is dropped; the stream ends short
        catch err
            # benign teardown: encode (the only closer of gathered/free) died
            # and closed them; the extra close is belt-and-braces idempotence
            _indexflow_error(err, err_out, gathered; who = "read")
        finally
            close(gathered)
        end
    end

    # ---- encode: H2D -> GPU revcomp -> 2x kernel -> D2H -> RefRopeRealBatch -
    t_encode = Threads.@spawn begin
        prog = progress ? ProgressUnknown(desc = "Reference windows encoded ", dt = 1.0) : nothing
        total = 0
        try
            for (buf, nb, heads, starts) in gathered
                first_idx = total + 1
                dwords = CuArray{UInt32}(undef, W * nb)
                copyto!(dwords, 1, buf, 1, W * nb) # stream-ordered H2D (pinned src)
                drc = CuArray{UInt32}(undef, W * nb) # revcomp words, on GPU
                revcomp_frag_words!(drc, dwords, k, nb)
                dest_f = CUDA.zeros(T, nb, rdim)
                dest_r = CUDA.zeros(T, nb, rdim)
                dn_f = CUDA.zeros(Float32, re.m, nb)
                dn_r = CUDA.zeros(Float32, re.m, nb)
                encode_frag_real_batch!(dest_f, dn_f, re, dwords; normalize)
                encode_frag_real_batch!(dest_r, dn_r, re, drc; normalize)
                # NB: contiguous device->host downloads into matching-shape
                # matrices, then host-side vcat/hcat into the batch layout
                # (a linear-offset copyto! would scramble the column-major
                # layout; a copyto! into a row-block view from a CuArray is
                # not GPU-supported)
                embeds_f = Matrix{T}(undef, nb, rdim)
                embeds_r = Matrix{T}(undef, nb, rdim)
                copyto!(embeds_f, dest_f)          # rows 1:nb = fwd
                copyto!(embeds_r, dest_r)          # rows nb+1:2nb = revcomp
                norms_f = Matrix{Float32}(undef, re.m, nb)
                norms_r = Matrix{Float32}(undef, re.m, nb)
                copyto!(norms_f, dn_f)
                copyto!(norms_r, dn_r)
                embeds = vcat(embeds_f, embeds_r)
                norms = hcat(norms_f, norms_r)
                CUDA.synchronize() # batch done: staging buffer and device pool free
                put!(free, buf)
                total += nb
                put!(out, RefRopeRealBatch(embeds, norms,
                                           vcat(heads, heads), vcat(starts, starts),
                                           vcat(fill(UInt8(0), nb), fill(UInt8(1), nb)),
                                           first_idx))
                prog === nothing || update!(prog, total)
            end
        catch err
            if _indexflow_error(err, err_out, out, gathered, free; who = "encode")
                # benign teardown (the consumer closed `out`): stop upstream --
                # read may be blocked in put!(gathered)/take!(free); idempotent
                close(gathered)
                close(free)
            end
        finally
            close(out) # also the early-close path (close on closed is a no-op)
            prog === nothing || finish!(prog)
        end
    end

    append!(tasks_out[], [t_read, t_encode]) # consumers may wait these
    return out
end

"""
    build_index_matrix(re, file; k = re.k, kstep = k ÷ 10, batch_size = 2^13,
                       normalize = 0, fp16 = true, progress = false,
                       max_records = typemax(Int), err_out, tasks_out)

The production collection step: a cheap first pass counts the windows (so the
matrices are allocated ONCE at their exact size), then the stream fills

    embeds (2*n_frag, 2*m*4^c) {Float16,Float32}   rows: per batch fwd then rc
    norms  (m, 2*n_frag) Float32
    heads  (2*n_frag) String   -- record headers, verbatim with '>'
    starts (2*n_frag) Int      -- 1-based window start in the record
    strand (2*n_frag) UInt8    -- 0 = fwd, 1 = revcomp

Returns them as a NamedTuple with `n_frag` (the fragment/window count; rows =
2 * n_frag).  This is exactly what `run_save` serializes to the .bin.
"""
function build_index_matrix(re::RopeEncoder, file::String;
                            k::Int = re.k,
                            kstep::Int = max(k ÷ 10, 1),
                            batch_size::Int = 2^13,
                            normalize::Int = 0,
                            fp16::Bool = true,
                            fp8::Bool = IF_FP8,
                            progress::Bool = false,
                            max_records::Int = typemax(Int),
                            err_out::Ref{Any} = Ref{Any}(nothing),
                            tasks_out::Ref{Vector{Task}} = Ref{Vector{Task}}(Task[]))
    t_count = @elapsed (nwin, nrec, nkept) = _count_ref_windows(file; k, kstep, max_records)
    @info "count pass: $nrec records, $nkept with >= k bases -> $nwin windows (x2 with revcomp) in $(round(t_count; digits = 2)) s"
    rdim = 2 * re.m * 4^re.c
    if fp8
        # the production MAPPING layout: database columns (rdim, 2*nwin) in
        # e4m3 -- the fp8 flow's resident B verbatim (loaded with a pure H2D;
        # quantization happens here, once, instead of at every mapping run)
        embeds8 = Matrix{Float8_E4M3FN}(undef, rdim, 2 * nwin)
    else
        T = fp16 ? Float16 : Float32
        embeds = Matrix{T}(undef, 2 * nwin, rdim)
    end
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    heads = Vector{String}(undef, 2 * nwin)
    starts = Vector{Int}(undef, 2 * nwin)
    strand = fill(UInt8(0), 2 * nwin) # rc rows set below
    row = 0
    t_stream = @elapsed for nt in ref_rope_real_stream(re, file; k, kstep, batch_size,
                                                       normalize, fp16, progress,
                                                       max_records, err_out, tasks_out)
        n2 = size(nt.embeds, 1)
        if fp8
            # fused transpose + fp16 -> fp32 -> e4m3 quantize into the column
            # block: the exact conversion the mapping flow's GPU-side
            # index_to_f8 applied to the saved fp16 rows (bitwise-equal B)
            @views embeds8[:, row+1:row+n2] .= Float8_E4M3FN.(Float32.(permutedims(nt.embeds)))
        else
            copyto!(view(embeds, row+1:row+n2, :), nt.embeds) # row-block elementwise
        end
        copyto!(view(norms, :, row+1:row+n2), nt.norms)
        copyto!(heads, row + 1, nt.heads, 1, n2)
        copyto!(starts, row + 1, nt.starts, 1, n2)
        copyto!(strand, row + 1, nt.strand, 1, n2)
        row += n2
    end
    err_out[] === nothing || error("stream failed: $(err_out[])")
    row == 2 * nwin || error("stream produced $row rows, expected $(2 * nwin)")
    @info "stream pass: $row rows in $(round(t_stream; digits = 2)) s"
    if fp8
        return (embeds8 = embeds8, norms = norms, heads = heads, starts = starts,
                strand = strand, n_frag = nwin)
    end
    return (embeds = embeds, norms = norms, heads = heads, starts = starts,
            strand = strand, n_frag = nwin)
end

# ==============================================================================
# Production modes: save / verify
# ==============================================================================

function _index_data_nt(source::String, re::RopeEncoder, k::Int, kstep::Int,
                        normalize::Int, fp16::Bool, data::NamedTuple;
                        fp8::Bool = false)
    if fp8
        return (; format = IF_FP8_FORMAT, source = abspath(source), k, kstep,
                s = re.s, m = re.m, c = re.c, normalize, fp8, n_frag = data.n_frag,
                embeds8 = data.embeds8, norms = data.norms, heads = data.heads,
                starts = data.starts, strand = data.strand)
    end
    (; format = IF_FORMAT, source = abspath(source), k, kstep,
     s = re.s, m = re.m, c = re.c, normalize, fp16, n_frag = data.n_frag,
     embeds = data.embeds, norms = data.norms, heads = data.heads,
     starts = data.starts, strand = data.strand)
end

"""
    run_save(; fasta, binfile, k, kstep, normalize, fp16, fp8, batch_size)

Build the index matrices for the reference `fasta` and serialize them next to
it.  `fp8 = true` (the DEFAULT, env INDEX_FP8) writes the production
mapping layout: transposed e4m3 database columns (`embeds8`, format
"indexreal.fp8.v1") to `<fasta>.indexreal_fp8.bin` -- e2ehuman.jl's
resident B verbatim.  `fp8 = false` writes the classic fp16/fp32 row layout
(format "indexreal.v1") to `<fasta>.indexreal.bin` (e2ehuman16.jl's
input); `fp16` selects the stream/output eltype in both cases.
"""
function run_save(; fasta::String = IF_FASTA,
                  binfile::Union{String,Nothing} = nothing,
                  k::Int = IF_K, kstep::Int = IF_KSTEP, normalize::Int = IF_NORM,
                  fp16::Bool = IF_FP16, fp8::Bool = IF_FP8, batch_size::Int = IF_BATCH)
    binfile === nothing && (binfile = fp8 ? _if_fp8file(fasta) : _if_binfile(fasta))
    re = RopeEncoder(k = k, s = IF_S, m = IF_M, c = IF_C)
    @info "indexreal save" fasta k kstep normalize fp16 fp8 batch_size s = IF_S m = IF_M c = IF_C rdim = 2 * IF_M * 4^IF_C layout = (fp8 ? "transposed e4m3 columns (mapping db)" : (fp16 ? "fp16 rows" : "fp32 rows")) rowdim = 2 * re.m * 4^re.c
    data = build_index_matrix(re, fasta; k, kstep, batch_size, normalize, fp16,
                              fp8, progress = true)
    nt = _index_data_nt(fasta, re, k, kstep, normalize, fp16, data; fp8)
    t = @elapsed save(nt, binfile)
    gbytes = filesize(binfile) / 2^30
    @info "saved $binfile ($(round(gbytes; digits = 2)) GiB) in $(round(t; digits = 1)) s"
    return nt
end

# independent codes for the window (head, start): scalar scan from the record's
# sequence span -- deliberately NOT the production SWAR path (no shared code)
function _window_codes_indep(raw::Vector{UInt8}, seq_lo::Int, seq_hi::Int,
                             start::Int, k::Int)
    codes = Vector{UInt8}(undef, k)
    pos = 0
    got = 0
    i = seq_lo
    @inbounds while i <= seq_hi
        c = raw[i]
        i += 1
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        pos += 1
        pos < start && continue
        got += 1
        codes[got] = _LUT3[Int(c) + 1]
        got == k && break
    end
    got == k || error("window (start=$start, k=$k) not inside the record")
    return codes
end

# sampled rows vs the independent chain: raw bytes -> LUT codes -> (rc) ->
# float64 CPU rope reference.  fp8 rows are DEQUANTIZED first: the tolerance
# then covers e4m3 rounding (half-spacing up to 2^-4 relative for normals,
# 2^-10 absolute in the subnormal/zero tail) on top of the fp16 kernel noise
# -- loose enough not to false-positive, far tighter than any real bug (a
# wrong window/meta gives O(1)-typical entry discrepancies).
function _check_rows_indep!(d::NamedTuple, raw::Vector{UInt8},
                            spans::Dict{String,Tuple{Int,Int}}, re::RopeEncoder;
                            rows::AbstractVector{Int})
    normalize = d.normalize
    fp8 = get(d, :format, nothing) == IF_FP8_FORMAT
    fp16 = get(d, :fp16, true)
    heavy = normalize in (2, 3)
    rtol = fp8 ? 1.5e-1 : 1e-2
    atol = fp8 ? 8.0e-3 : (fp16 ? (heavy ? 2e-2 : 2e-3) : (heavy ? 1e-2 : 2e-4))
    nrm_rows = normalize == 0 ? (1:1) : (1:re.m)
    for si in rows
        (head, start, strand) = (d.heads[si], d.starts[si], d.strand[si])
        haskey(spans, head) || error("row $si header not found in the fasta: $head")
        (lo, hi) = spans[head]
        codes = _window_codes_indep(raw, lo, hi, start, d.k)
        if strand == 0x01 # revcomp row: complement + reverse the codes
            codes = UInt8[0x03 - codes[d.k+1-j] for j in 1:d.k]
        else
            @assert strand == 0x00 "bad strand byte at row $si: $(strand)"
        end
        href, nrmref = _ref_rope_frag_real(codes, re; normalize)
        row = fp8 ? Float32.(d.embeds8[:, si]) : view(d.embeds, si, :)
        @assert isapprox(row, href; rtol, atol) "row $si ($(head) start=$start strand=$strand) vs CPU rope reference"
        for idm in nrm_rows
            @assert isapprox(d.norms[idm, si], nrmref[idm]; rtol = 1e-3) "norms row $si (copy $idm)"
        end
    end
    return nothing
end

"""
    run_verify(; fasta, binfile, rows = 24, seed = 1234)

Reload the .bin and validate it against the fasta INDEPENDENTLY: format
fields, eltype, window count vs a fresh count pass, a full structural meta
check (strand-block layout; per-window heads/starts rebuilt from a fresh
record walk in stream order), and `rows` sampled embeddings vs the float64
CPU rope reference.
"""
function run_verify(; fasta::String = IF_FASTA, binfile::String = _if_binfile(fasta),
                    rows::Int = 24, seed::Int = 1234)
    @info "indexreal verify" fasta binfile
    isfile(fasta) || error("fasta not found: $fasta")
    d = load(binfile)
    fmt = get(d, :format, nothing)
    fp8 = fmt == IF_FP8_FORMAT
    @assert fmt in (IF_FORMAT, IF_FP8_FORMAT) "not an $IF_FORMAT/$IF_FP8_FORMAT file"
    @assert d.k == IF_K && d.kstep == IF_KSTEP "config mismatch: k=$(d.k) kstep=$(d.kstep)"
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    rdim = 2 * d.m * 4^d.c
    if fp8
        @assert d.fp8 == true "fp8 flag mismatch"
        @assert eltype(d.embeds8) == Float8_E4M3FN "eltype mismatch"
        N = size(d.embeds8, 2)
        @assert size(d.embeds8, 1) == rdim "embeds8 width mismatch"
    else
        T = d.fp16 ? Float16 : Float32
        @assert eltype(d.embeds) == T "eltype mismatch"
        N = size(d.embeds, 1)
        @assert size(d.embeds, 2) == rdim "embeds width mismatch"
    end
    @assert size(d.norms) == (d.m, N) "norms shape mismatch"
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"

    abspath(d.source) == abspath(fasta) ||
        @warn "saved source differs from the given fasta" saved = d.source given = abspath(fasta)

    # independent count pass + record spans (one walk for the sampled checks)
    raw = open(fasta, "r") do io
        Mmap.mmap(io)
    end
    (nwin, nrec, nkept) = _count_ref_windows(fasta; k = d.k, kstep = d.kstep)
    @assert N == 2 * nwin == 2 * d.n_frag "row count mismatch: $N vs 2 x $nwin"
    @assert count(==(0x00), d.strand) == nwin && count(==(0x01), d.strand) == nwin "strand split mismatch"
    @info "count OK: $nrec records, $nkept kept, $nwin windows (x2 rows = $N)"

    # full structural check: rebuild the (head, start) list in file order and
    # compare against the saved meta's [fwd-block; rc-block] batch layout
    meta = Vector{Tuple{String,Int}}()
    n = length(raw)
    s = _next_rec3(raw, 1, n)
    while s <= n
        h = s
        while h <= n && raw[h] != UInt8('\n')
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == UInt8('\r')) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n)
        nbases = _count_bases(raw, h + 1, e - 1)
        if nbases >= d.k
            header = String(raw[s:hend])
            for wi in 0:((nbases - d.k) ÷ d.kstep)
                push!(meta, (header, wi * d.kstep + 1))
            end
        end
        s = e
    end
    length(meta) == nwin || error("meta walk count mismatch")
    i = 1
    g = 0
    while i <= N
        @assert d.strand[i] == 0x00 "expected a fwd block at row $i"
        j = i
        while j <= N && d.strand[j] == 0x00
            j += 1
        end
        j2 = j
        while j2 <= N && d.strand[j2] == 0x01
            j2 += 1
        end
        B = j - i
        @assert j2 - j == B "fwd/rc block sizes differ at row $j"
        for t in 0:B-1
            g += 1
            (h_, st_) = meta[g]
            @assert d.heads[i+t] == h_ && d.starts[i+t] == st_ "fwd meta mismatch at row $(i+t)"
            @assert d.heads[j+t] == h_ && d.starts[j+t] == st_ "rc meta mismatch at row $(j+t)"
        end
        i = j2
    end
    @assert g == nwin "structural check missed windows ($g vs $nwin)"
    @info "structural meta OK ($nwin fwd/rc pairs)"

    # sampled rows vs the independent CPU chain
    (spans, _) = _record_spans(raw)
    rs = MersenneTwister(seed)
    _check_rows_indep!(d, raw, spans, re; rows = rand(rs, 1:N, min(rows, N)))
    @info "sampled rows OK ($(min(rows, N)) vs the float64 CPU reference)"
    @info "INDEXFLOWREAL VERIFY PASSED"
    return d
end

# ==============================================================================
# Data generation for the tests: a synthetic edge-case reference fasta
# ==============================================================================

# _reference_fragments walk): the full (header, start, words) window list in
# file order -- scalar byte collection, _ref_forward_pack per window.
function _ref_windows_naive(raw::Vector{UInt8}, k::Int, kstep::Int)
    out = Vector{Tuple{String,Int,Vector{UInt32}}}()
    NL = UInt8('\n'); CR = UInt8('\r'); GT = UInt8('>')
    n = length(raw)
    i = 1
    while i <= n # junk before the first record is discarded
        raw[i] == GT && (i == 1 || raw[i-1] == NL || raw[i-1] == CR) && break
        i += 1
    end
    while i <= n
        s = i
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = n + 1
        j = min(h + 1, n + 1)
        while j <= n
            if raw[j] == GT && (raw[j-1] == NL || raw[j-1] == CR)
                e = j
                break
            end
            j += 1
        end
        codes = UInt8[]
        for t in h+1:e-1
            c = raw[t]
            (c == NL || c == CR) && continue
            push!(codes, _LUT3[Int(c) + 1])
        end
        i = e
        len = length(codes)
        if len >= k
            header = String(raw[s:hend])
            for p in 0:kstep:(len - k)
                push!(out, (header, p + 1, _ref_forward_pack(codes[p+1:p+k], k)))
            end
        end
    end
    return out
end

# the production CPU path (engine + funnel windows); NOTE: `wnd`/`sc` must be
# passed explicitly -- the engine fills ITS argument vectors, the closure reads
# the caller's
function _prod_windows(raw::Vector{UInt8}, k::Int, kstep::Int)
    out = Vector{Tuple{String,Int,Vector{UInt32}}}()
    wnd = Vector{UInt32}(undef, cld(k, 16))
    sc = Vector{UInt16}(undef, 4096)
    _each_ref_window!(raw; k, kstep, sc, wnd) do h, st
        push!(out, (h, st, copy(wnd)))
    end
    return out
end

# A. window extraction: production vs the independent naive parser (exact)
function _check_windows(path::String; k::Int, ksteps)
    raw = read(path)
    for kstep in ksteps
        want = _ref_windows_naive(raw, k, kstep)
        got = _prod_windows(raw, k, kstep)
        @assert length(got) == length(want) "window count mismatch (k=$k, kstep=$kstep): $(length(got)) vs $(length(want))"
        for i in eachindex(want)
            @assert got[i][1] == want[i][1] && got[i][2] == want[i][2] "window meta mismatch at $i (k=$k, kstep=$kstep): $((got[i][1], got[i][2])) vs $((want[i][1], want[i][2]))"
            @assert got[i][3] == want[i][3] "window words mismatch at $i (k=$k, kstep=$kstep)"
        end
        # the count pass must agree with both
        (nwin, _, _) = _count_ref_windows(path; k, kstep)
        @assert nwin == length(want) "count pass mismatch (k=$k, kstep=$kstep)"
    end
    return nothing
end

# B. revcomp: funnel twin vs codes reference (host); GPU kernel vs codes ref
function _check_revcomp()
    Random.seed!(99)
    for k in (17, 50, 2001, 16, 32, 20_000)
        W = cld(k, 16)
        words = Vector{UInt32}(undef, W)
        for rep in 1:8
            rand!(words)
            if (r = k & 15) > 0 # zero-padded tail, as the parser emits
                words[W] &= (UInt32(1) << (2 * r)) - 1
            end
            ref = _revcomp_words_ref(words, k)
            @assert _revcomp_words_funnel(words, k) == ref "host funnel revcomp mismatch (k=$k, rep=$rep)"
            if rep <= 2
                dwords = cu(words)
                drc = CUDA.zeros(UInt32, W)
                revcomp_frag_words!(drc, dwords, k, 1)
                @assert Array(drc) == ref "GPU revcomp mismatch (k=$k, rep=$rep)"
            end
        end
    end
    # a batch through the kernel, mixed tail residues
    for (k, nfrag) in ((2001, 64), (20_000, 8), (50, 130))
        W = cld(k, 16)
        words = Vector{UInt32}(undef, W * nfrag)
        rand!(words)
        if (r = k & 15) > 0
            mask = (UInt32(1) << (2 * r)) - 1
            for f in 1:nfrag
                words[f*W] &= mask
            end
        end
        drc = CUDA.zeros(UInt32, W * nfrag)
        revcomp_frag_words!(drc, cu(words), k, nfrag)
        drc_h = Array(drc)
        for f in 1:nfrag
            @assert drc_h[(f-1)*W+1:f*W] == _revcomp_words_ref(words[(f-1)*W+1:f*W], k) "GPU batch revcomp mismatch (k=$k, frag $f)"
        end
    end
    return nothing
end

# C/D. end-to-end stream on the synthetic file: meta (deterministic order!),
# unit energy, sampled rows vs the CPU reference (fwd AND rc rows).
function _check_ref_stream(re, path; k::Int, kstep::Int, normalize::Int,
                           batch_size::Int, fp16::Bool,
                           refw::Vector{Tuple{String,Int,Vector{UInt32}}},
                           sample::Int, seed = 7)
    rs = MersenneTwister(seed)
    err = Ref{Any}(nothing)
    nb = seen = 0
    D = re.m * 4^re.c
    rdim = 2 * D
    T = fp16 ? Float16 : Float32
    for nt in ref_rope_real_stream(re, path; k, kstep, batch_size, normalize, fp16,
                                   err_out = err)
        nb += 1
        n2 = size(nt.embeds, 1)
        nh = length(nt.heads)
        @assert nt isa RefRopeRealBatch && eltype(nt.embeds) == T "batch eltype mismatch (fp16 = $fp16)"
        @assert n2 == nh == length(nt.starts) == length(nt.strand) "batch meta length mismatch"
        @assert nt.first == seen + 1 "batch first-index mismatch"
        nfrag = n2 ÷ 2
        ertol = fp16 ? 1e-2 : 1e-3
        for r in 1:n2
            frag = nt.first + (r - 1) % nfrag # 1-based pair index within the stream
            want_strand = r <= nfrag ? UInt8(0) : UInt8(1)
            @assert nt.strand[r] == want_strand "strand mismatch at row $r"
            @assert nt.heads[r] == refw[frag][1] "head mismatch at row $r"
            @assert nt.starts[r] == refw[frag][2] "start mismatch at row $r"
            # unit energy (normalize 0: whole row; else per copy at stride m)
            e = if normalize == 0
                sum(abs2, @view(nt.embeds[r, 1:D]); init = 0.0) +
                sum(abs2, @view(nt.embeds[r, D+1:rdim]); init = 0.0)
            else
                maximum(sum(abs2, @view(nt.embeds[r, idm:re.m:D]); init = 0.0) +
                        sum(abs2, @view(nt.embeds[r, D+idm:re.m:rdim]); init = 0.0)
                        for idm in 1:re.m)
            end
            @assert isapprox(e, 1.0; rtol = ertol) "unit energy violated (row $r)"
        end
        # sampled rows vs the CPU reference
        heavy = normalize in (2, 3)
        rtol = 1e-2
        atol = fp16 ? (heavy ? 2e-2 : 2e-3) : (heavy ? 1e-2 : 2e-4)
        for r in rand(rs, 1:n2, min(sample, n2))
            frag = nt.first + (r - 1) % nfrag
            codes = _frag_codes(refw[frag][3], k)
            if nt.strand[r] == 0x01
                codes = UInt8[0x03 - codes[k+1-j] for j in 1:k]
            end
            href, nrmref = _ref_rope_frag_real(codes, re; normalize)
            @assert isapprox(@view(nt.embeds[r, :]), href; rtol, atol) "streamed row vs CPU reference mismatch ($(nt.heads[r]), start=$(nt.starts[r]), strand=$(nt.strand[r]))"
            nrm_cols = normalize == 0 ? (1:1) : (1:re.m)
            for idm in nrm_cols
                @assert isapprox(nt.norms[idm, r], nrmref[idm]; rtol = 1e-3) "streamed norms mismatch (row $r, copy $idm)"
            end
        end
        seen += nfrag
    end
    @assert err[] === nothing "stream error: $(err[])"
    @assert seen == length(refw) "fragment count mismatch: $seen vs $(length(refw))"
    @assert nb == cld(seen, batch_size) "batch count mismatch ($nb for batch_size=$batch_size)"
    return nb
end

# save/load roundtrip on `path`; returns the reloaded NamedTuple
function _check_roundtrip(re, path::String, dir::String;
                          k::Int, kstep::Int, batch_size::Int, normalize::Int,
                          fp16::Bool, fp8::Bool = false,
                          max_records::Int = typemax(Int))
    data = build_index_matrix(re, path; k, kstep, batch_size, normalize, fp16,
                              fp8, max_records)
    bin = joinpath(dir, fp8 ? "roundtrip_fp8.bin" : "roundtrip.bin")
    nt = _index_data_nt(path, re, k, kstep, normalize, fp16, data; fp8)
    save(nt, bin)
    d = load(bin)
    @assert get(d, :format, nothing) == (fp8 ? IF_FP8_FORMAT : IF_FORMAT)
    @assert d.k == k && d.kstep == kstep && d.normalize == normalize
    @assert (d.s, d.m, d.c) == (re.s, re.m, re.c)
    @assert d.n_frag == data.n_frag
    if fp8
        @assert d.fp8 == true
        @assert eltype(d.embeds8) == Float8_E4M3FN
        @assert size(d.embeds8) == (2 * re.m * 4^re.c, 2 * data.n_frag)
    else
        @assert d.fp16 == fp16
        @assert eltype(d.embeds) == (fp16 ? Float16 : Float32)
        @assert isequal(d.embeds, data.embeds)
    end
    @assert isequal(d.norms, data.norms)
    @assert d.heads == data.heads && d.starts == data.starts && d.strand == data.strand
    return d
end

function run_rope_index_test()
    @show CUDA.name(device())
    @show nthreads()

    dir = mktempdir(prefix = "indexreal_")
    small = _ensure_test_fasta() # records sized relative to k = 20,000
    re = RopeEncoder(k = 20_000, s = 8, m = 4, c = 4) # the workflow's config

    # ==========================================================================
    # A. window extraction vs the independent naive parser (exact, CPU only)
    # ==========================================================================
    @info "A. window extraction vs the naive reference"
    for (k, ksteps) in ((50, (1, 3, 7, 50, 51)), (2001, (1, 100, 2000)), (20_000, (20_000, 2_000)))
        _check_windows(small; k, ksteps)
        @info "  k=$k, ksteps=$(collect(ksteps)) OK"
    end

    # ==========================================================================
    # B. revcomp words: host funnel twin + GPU kernel vs the codes reference
    # ==========================================================================
    @info "B. revcomp words (host funnel + GPU kernel vs codes reference)"
    _check_revcomp()
    @info "  all tail residues + batches OK"

    # ==========================================================================
    # C. end-to-end stream on the synthetic file: batch layouts, fp32/fp16,
    #    normalize 0/1, meta equality (deterministic single-reader order),
    #    unit energy, sampled rows vs the CPU reference (fwd + rc)
    # ==========================================================================
    @info "C. ref_rope_real_stream on the synthetic file"
    refw = _ref_windows_naive(read(small), 20_000, 2_000)
    nwin = length(refw)
    @info "  reference: $nwin windows (k=20000, kstep=2000)"
    for batch_size in (1, 3, 2^13, 10^6) # single, tiny, default, oversized
        nb = _check_ref_stream(re, small; k = 20_000, kstep = 2_000, normalize = 0,
                               batch_size, fp16 = false, refw, sample = 8)
        @info "  normalize=0, fp32, batch_size=$batch_size OK ($nb batches)"
    end
    nb = _check_ref_stream(re, small; k = 20_000, kstep = 2_000, normalize = 1,
                           batch_size = 3, fp16 = false, refw, sample = 8)
    @info "  normalize=1, fp32, batch_size=3 OK ($nb batches)"
    for normalize in (0, 1) # the fp16 default path through the whole stream
        nb = _check_ref_stream(re, small; k = 20_000, kstep = 2_000, normalize,
                               batch_size = 3, fp16 = true, refw, sample = 8)
        @info "  normalize=$normalize, fp16, batch_size=3 OK ($nb batches)"
    end
    nb = _check_ref_stream(re, small; k = 20_000, kstep = 20_000, normalize = 0,
                           batch_size = 2, fp16 = true,
                           refw = _ref_windows_naive(read(small), 20_000, 20_000),
                           sample = 8)
    @info "  kstep=k (non-overlapping windows), fp16 OK ($nb batches)"
    re2 = RopeEncoder(k = 20_000, s = 5, m = 1, c = 4) # another encoder config
    nb = _check_ref_stream(re2, small; k = 20_000, kstep = 2_000, normalize = 0,
                           batch_size = 3, fp16 = true, refw, sample = 4)
    @info "  config (s=5, m=1, c=4) end-to-end OK ($nb batches)"

    # ==========================================================================
    # D. build_index_matrix + save/load roundtrip (bitwise field equality)
    # ==========================================================================
    @info "D. build_index_matrix + .bin roundtrip"
    d = _check_roundtrip(re, small, dir; k = 20_000, kstep = 2_000, batch_size = 3,
                         normalize = 0, fp16 = true)
    @assert d.n_frag == nwin
    d32 = _check_roundtrip(re, small, dir; k = 20_000, kstep = 2_000, batch_size = 3,
                           normalize = 0, fp16 = false)
    @assert eltype(d32.embeds) == Float32
    @assert isapprox(d32.embeds, Float32.(d.embeds); rtol = 1e-2, atol = 2e-3) "fp32 vs fp16 matrices disagree"
    d8 = _check_roundtrip(re, small, dir; k = 20_000, kstep = 2_000, batch_size = 3,
                          normalize = 0, fp16 = true, fp8 = true)
    # the e4m3 columns vs the fp16 rows: same values within the e4m3 grid
    # (half-spacing up to 2^-4 relative for normals, 2^-10 absolute subnormal)
    @assert isapprox(Float32.(permutedims(d8.embeds8)), Float32.(d.embeds);
                     rtol = 1.5e-1, atol = 8e-3) "fp8 columns vs fp16 rows disagree"
    @info "  roundtrip OK (fp16 + fp32 + fp8, n_frag=$nwin)"

    # ==========================================================================
    # E. teardown: early close unwinds quietly; degenerate files give 0 windows
    # ==========================================================================
    @info "E. early close + degenerate files"
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    ch = ref_rope_real_stream(re, small; k = 20_000, kstep = 2_000, batch_size = 3,
                              err_out = err, tasks_out = tks)
    got = 0
    for _ in ch
        got += 1
        got == 2 && break
    end
    close(ch)
    foreach(wait, tks[]) # deterministic: the whole pipeline has unwound
    @assert err[] === nothing "early close raised: $(err[])"
    @assert got == 2 && all(istaskdone, tks[])
    nb = _check_ref_stream(re, small; k = 20_000, kstep = 2_000, normalize = 0,
                           batch_size = 3, fp16 = false, refw, sample = 4)
    @info "  early close OK; pipeline reusable afterwards ($nb batches)"

    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    onlyheads = joinpath(dir, "onlyheads.fasta")
    write(onlyheads, ">r1\n>r2\n>r3 description only\n")
    for f in (empty_fasta, onlyheads)
        err = Ref{Any}(nothing)
        n = 0
        for _ in ref_rope_real_stream(re, f; k = 20_000, kstep = 2_000, err_out = err)
            n += 1
        end
        @assert n == 0 && err[] === nothing "degenerate file $f: $n batches"
    end
    (nw0, _, _) = _count_ref_windows(onlyheads; k = 20_000, kstep = 2_000)
    @assert nw0 == 0
    @info "  empty + header-only files OK (0 batches)"

    # ==========================================================================
    # F. the human reference PREFIX (first IF_TEST_RECORDS records) end-to-end:
    #    count formula, stream, .bin roundtrip, sampled rows vs the independent
    #    chain (raw bytes -> LUT -> rc -> float64 CPU rope reference)
    # ==========================================================================
    if isfile(IF_FASTA)
        nrec = IF_TEST_RECORDS
        @info "F. human reference prefix (first $nrec record(s)) end-to-end"
        (nwin_h, _, _) = _count_ref_windows(IF_FASTA; k = 20_000, kstep = 2_000,
                                            max_records = nrec)
        @info "  prefix windows: $nwin_h (x2 rows = $(2 * nwin_h))"
        d = _check_roundtrip(re, IF_FASTA, dir; k = 20_000, kstep = 2_000,
                             batch_size = 2^13, normalize = 0, fp16 = true,
                             max_records = nrec)
        @assert d.n_frag == nwin_h "prefix count mismatch"
        d8 = _check_roundtrip(re, IF_FASTA, dir; k = 20_000, kstep = 2_000,
                              batch_size = 2^13, normalize = 0, fp16 = true,
                              fp8 = true, max_records = nrec)
        @assert d8.n_frag == nwin_h "fp8 prefix count mismatch"
        raw = open(IF_FASTA, "r") do io
            Mmap.mmap(io)
        end
        (spans, _) = _record_spans(raw; max_records = nrec)
        rs = MersenneTwister(555)
        rows = rand(rs, 1:(2 * nwin_h), 8)
        _check_rows_indep!(d, raw, spans, re; rows)
        _check_rows_indep!(d8, raw, spans, re; rows)
        @info "  prefix roundtrip + sampled rows OK (fp16 + fp8, $(2 * nwin_h) rows)"
    else
        @warn "human fasta not found, skipping F" IF_FASTA
    end

    @info "ALL INDEXFLOWREAL CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks on the human reference (skip timings gracefully if it is absent)
# ==============================================================================
function bench_index(; fasta::String = IF_FASTA, k::Int = IF_K, kstep::Int = IF_KSTEP,
                     batch_size::Int = IF_BATCH, reps::Int = IF_REPS)
    isfile(fasta) || error("fasta not found: $fasta")
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4)
    _warm_cache(fasta)
    bytes = filesize(fasta)
    @info "indexreal benchmarks" julia_threads = nthreads() reps bytes

    _timed_min("count pass (records + windows)"; bytes, reps) do
        (nwin, _, _) = _count_ref_windows(fasta; k, kstep)
        nwin
    end

    _timed_min("reader drain (walk + pack + slice, no GPU)"; bytes, reps) do
        n = 0
        raw = open(fasta, "r") do io
            Mmap.mmap(io)
        end
        _each_ref_window!(raw; k, kstep) do _, _
            n += 1
        end
        n
    end

    _timed_min("full stream, no-op consumer (H2D+rc+2xkernel+D2H)"; bytes, reps) do
        n = 0
        for nt in ref_rope_real_stream(re, fasta; k, kstep, batch_size, fp16 = true)
            n += size(nt.embeds, 1)
        end
        n
    end

    println("  (GB/s = FASTA bytes consumed per second; the stream holds ~in_cap")
    println("   pinned word batches + out_cap embedding batches in flight; rows = 2x")
    println("   windows -- fwd + revcomp; fp16 output halves the embedding D2H)")
    @printf("  device pool: used %.2f GiB, cached-free %.2f GiB\n",
            CUDA.used_memory() / 2^30, CUDA.cached_memory() / 2^30)
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    posargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(posargs) ? "all" : posargs[1]
    fasta2 = length(posargs) > 1 ? posargs[2] : IF_FASTA
    mode == "gen" && _ensure_test_fasta()
    mode == "test" && run_rope_index_test()
    mode == "save" && run_save(fasta = fasta2)
    mode == "verify" && run_verify(fasta = fasta2,
                                   binfile = "--bin8" in ARGS ? _if_fp8file(fasta2) :
                                             _if_binfile(fasta2))
    mode == "bench" && bench_index(fasta = fasta2)
    mode == "all" && (_ensure_test_fasta(); run_rope_index_test(); isfile(IF_FASTA) && bench_index())
    mode in ("gen", "test", "save", "verify", "bench", "all") ||
        error("unknown mode $mode (use gen|test|save|verify|bench|all)")
end
