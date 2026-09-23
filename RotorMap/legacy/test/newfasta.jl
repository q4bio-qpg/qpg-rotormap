using Revise

include(joinpath(@__DIR__, "newencoder2bit.jl")) # pack_reads (the reference packer) + the v2/v3 kernels

using Random
using CUDA
using Mmap
using ProgressMeter
using RotorMap
using RotorMap.Utils
using RotorMap.RopeEncoders
using Base.Threads

# ==============================================================================
# Fused FASTA -> 2-bit packed bitstream reader for the v3 encoder
# (newencoder2bit.jl). It replaces the reading workflow of newindex2.jl
#
#   seqs, heads  = load_fasta_mmap(file)                       # 8 bits/base, per-read vectors
#   seqs2        = deepcopy(seqs)
#   append!(seqs2, [UInt8(3).-reverse(a) for a in seqs])       # + reverse complements
#   jreads       = jagged_array(seqs2)                         # flat bytes + starts/stops
#   words, ...   = pack_reads(seqs2)                           # 2-bit pack, REVERSED base order
#   cropes       = encode_batch_cuda_all_best(re, jreads.vect; starts=..., stops=...)
#
# with a single pass over the file that emits the packed v3 bitstream directly:
#
#   words, starts, stops, heads = load_fasta_packed(file; revcomp = true)
#   cropes = encode_batch_cuda_all_best_v3(re, cu(words); starts = cu(starts), stops = cu(stops))
#
# WHY THE REVERSAL CANNOT LEAVE THE PACK STAGE
#
# v3 stores each read's bases in reverse order so that a little-endian bit field
# extracted from the stream equals the original big-endian s-mer (see the
# newencoder2bit.jl header). Keeping a forward stream instead would require
# either a per-window digit-reversal of the 2s-bit field in the hot loop
# (~10-25 ALU ops/window, defeating the register funnel shift), or changing the
# scramble to hash the little-endian halves directly -- which changes the bin
# assignment and breaks compatibility with indexes built by the original
# encoder. So the pack stage must own the reversal; the question is only how
# cheaply a forward FASTA stream can produce a reversed packed stream.
#
# THE FUSED STREAM FLOW (one pass over the file)
#
#   * a read's length is known only at its end, so its codes are collected into
#     one REUSABLE buffer (no per-read allocations, unlike the
#     Vector{Vector{UInt8}} intermediate of the two-pass path);
#   * at each '>' the buffer is emitted in reverse into a running bitstream
#     accumulator (UInt64 acc + 2-bit counter, flushed at 32 bits) that is
#     never reset between reads -- bitwise identical to `pack_reads`, which
#     accumulates `for read in reads, b in reverse(read)` the same way;
#   * the reverse-COMPLEMENT batch that newindex2.jl encodes alongside the
#     forward one is, in packed form, exactly the FORWARD sequence of
#     complemented codes (packing comp(reverse(s)) reversed = comp(s) read
#     forward; comp of a 2-bit code is `x ⊻ 0b11`), i.e. a pure sequential
#     stream. It is emitted in the same pass almost for free, replacing the
#     `[UInt8(3).-reverse(a)...]` + deepcopy + second jagged_array of the old
#     workflow (the two batches share starts/stops and headers);
#   * the produced stream is 4x smaller than the byte path at every stage
#     (host memory, host->device transfer, device staging).
#
# MULTITHREADING: the file is split at record starts (a '>' that begins a
# line), each thread parses its chunk independently into a chunk-local
# bitstream, and the chunks are merged with an exact 32-bit funnel shift
# (`_merge_chunk!`). The merged output is bitwise identical for any chunk
# count, so the single-threaded and multithreaded results are assertable in
# the test. Same trim/skipN/randomize_N/header-sync semantics as
# `load_fasta_mmap_fixed` (headers are kept in sync with the kept reads).
#
# Limitation (shared with pack_reads): base offsets are Int32, so the total
# number of bases in the file must stay below 2^31.
# ==============================================================================

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

# ---------------------------------------------------------------------------
# Running bitstream accumulator: (words::Vector{UInt32}, acc::UInt64, nbits::Int)
# with the invariant 0 <= nbits <= 30 (even): acc holds the trailing nbits bits
# of the stream, everything before is already flushed to `words` -- the same
# state pack_reads keeps while iterating the reversed reads.
# ---------------------------------------------------------------------------

# append one full 32-bit word of stream bits
@inline function _push_word!(words, acc::UInt64, nbits::Int, w::UInt32)
    acc |= UInt64(w) << nbits
    if nbits == 0
        push!(words, w)
        acc = UInt64(0)
    else
        push!(words, acc % UInt32)
        acc >>= 32 # keeps the high nbits bits of w
    end
    return acc, nbits
end

# append the low pnbits of pacc (pnbits <= 62, chunk-local trailing accumulator)
@inline function _push_bits!(words, acc::UInt64, nbits::Int, pacc::UInt64, pnbits::Int)
    while pnbits >= 32
        acc, nbits = _push_word!(words, acc, nbits, pacc % UInt32)
        pacc >>= 32
        pnbits -= 32
    end
    if pnbits > 0
        acc |= pacc << nbits # nbits + pnbits <= 60, no overflow
        nbits += pnbits
    end
    while nbits >= 32 # restore the nbits <= 30 invariant
        push!(words, acc % UInt32)
        acc >>= 32
        nbits -= 32
    end
    return acc, nbits
end

# append a whole chunk-local stream (words + trailing accumulator) to the
# global one; the chunk's bits sit at global bit offset 2 * (bases so far),
# which is exactly what the accumulator state encodes
@inline function _merge_chunk!(gwords, gacc::UInt64, gnbits::Int, cwords, cacc::UInt64, cnbits::Int)
    for w in cwords
        gacc, gnbits = _push_word!(gwords, gacc, gnbits, w)
    end
    gacc, gnbits = _push_bits!(gwords, gacc, gnbits, cacc, cnbits)
    return gacc, gnbits
end

# ---------------------------------------------------------------------------
# Per-chunk parsing: raw[lo:hi-1] (hi exclusive) starts at a record start (or
# at the beginning of the file, where junk before the first '>' is skipped).
# Returns the chunk-local packed stream(s) plus per-read data:
#   (words, acc, nbits, rwords, racc, rnbits, starts, stops, heads, nbases)
# where `words` is the forward batch (reversed base order, as v3 requires) and
# `rwords` the reverse-complement batch (only filled when rc == true).
# ---------------------------------------------------------------------------
function _parse_chunk_packed(raw, lo::Int, hi::Int; trim::Int = 0, skipN::Bool = false,
                             randomize_N::Bool = false, rc::Bool = false, progress::Bool = false)
    words = UInt32[]; rwords = UInt32[]
    starts = Int32[]; stops = Int32[]; heads = String[]
    acc = UInt64(0); nbits = 0
    racc = UInt64(0); rnbits = 0
    nbases = 0
    buf = UInt8[] # reusable code buffer of the current read

    lut = _PACK_LUT
    GT = UInt8('>'); NL = UInt8('\n'); CR = UInt8('\r')
    N1 = UInt8('N'); N2 = UInt8('n')

    p = nothing
    plast = 0
    if progress && hi - lo > (1 << 22)
        p = Progress(hi - lo + 1; dt = 1.0)
        plast = lo
    end

    i = lo
    while i < hi
        if p !== nothing && i - plast > (1 << 22)
            update!(p, i - lo + 1)
            plast = i
        end
        @inbounds b = raw[i]
        if b == GT
            # --- header line (kept with the leading '>'), tolerate CRLF ---
            hstart = i
            while i < hi
                @inbounds b = raw[i]
                b == NL && break
                i += 1
            end
            hend = i - 1
            (hend >= hstart && raw[hend] == CR) && (hend -= 1)
            head = String(raw[hstart:hend])
            i += 1 # past the newline (or past the end for a headerless newline)

            # --- sequence codes until the next header or the chunk end ---
            empty!(buf)
            skip_seq = false
            while i < hi
                @inbounds c = raw[i]
                c == GT && break
                i += 1
                @inbounds v = lut[c + 1]
                v == 0xff && continue # newlines, CR, spaces, other junk
                if c == N1 || c == N2
                    if skipN
                        skip_seq = true
                    elseif randomize_N
                        v = rand(UInt8(0):UInt8(3))
                    end
                end
                push!(buf, v)
            end

            # --- finalize the read (trim/skipN filters, header kept in sync
            #     with the kept reads, as in load_fasta_mmap_fixed) ---
            L = length(buf)
            if !skip_seq && (trim == 0 || L >= trim)
                keep = trim == 0 ? L : trim
                push!(starts, Int32(nbases + 1))
                nbases += keep
                push!(stops, Int32(nbases))
                push!(heads, head)

                # forward batch: REVERSED base order (the v3 endianness fix)
                @inbounds for j in keep:-1:1
                    acc |= UInt64(buf[j] & 3) << nbits
                    nbits += 2
                    if nbits == 32
                        push!(words, acc % UInt32)
                        acc = UInt64(0)
                        nbits = 0
                    end
                end
                # reverse-complement batch: FORWARD order, complemented codes
                # (packing comp(reverse(s)) reversed == comp(s) read forward)
                if rc
                    @inbounds for j in 1:keep
                        racc |= UInt64((buf[j] ⊻ 0x03) & 3) << rnbits
                        rnbits += 2
                        if rnbits == 32
                            push!(rwords, racc % UInt32)
                            racc = UInt64(0)
                            rnbits = 0
                        end
                    end
                end
            end
        else
            i += 1 # junk before the first header
        end
    end

    return words, acc, nbits, rwords, racc, rnbits, starts, stops, heads, nbases
end

# byte offsets of chunk boundaries: record starts (a '>' at the beginning of a
# line) nearest to an even split of the file; returns [1, ..., n + 1]
function _chunk_bounds(raw, nchunks::Int)
    n = length(raw)
    GT = UInt8('>'); NL = UInt8('\n'); CR = UInt8('\r')
    bounds = [1]
    if nchunks > 1
        step = cld(n, nchunks)
        b = step
        while b < n
            # first '>' at/after b that begins a line (tolerate CR before it)
            i = something(findnext(==(GT), raw, b), n + 1)
            while i <= n && i > 1 && raw[i-1] != NL && raw[i-1] != CR
                i = something(findnext(==(GT), raw, i + 1), n + 1)
            end
            i <= n || break # no further record: the last chunk covers the rest
            push!(bounds, i)
            b += step
        end
    end
    push!(bounds, n + 1)
    return bounds
end

"""
    load_fasta_packed(file; trim=0, skipN=false, randomize_N=false, revcomp=false,
                      chunks=0, progress=true)
        -> (words, starts, stops, heads)                    # revcomp = false
        -> (words, starts, stops, rwords, heads)            # revcomp = true

Single-pass fused FASTA reader for the v3 encoder: parses `file` and packs the
reads directly into the flat 2-bit bitstream `words::Vector{UInt32}` (reversed
base order, one zero guard word at the end -- the exact `pack_reads` format).
`starts`/`stops::Vector{Int32}` are the per-read base ranges (1-based,
inclusive) and `heads` the raw header lines (kept in sync with the kept reads,
as in `load_fasta_mmap_fixed`).

With `revcomp = true` the same pass also produces `rwords`, the bitstream of
the reverse-complemented reads (same segmentation: reuse `starts`/`stops`).
`trim > 0` keeps only reads of at least `trim` codes, truncated to the first
`trim`; `skipN` drops reads containing N; `randomize_N` replaces N with a
random code (all matching `load_fasta_mmap_fixed`).

`chunks > 1` forces that many (record-aligned) parallel chunks; `chunks = 0`
picks `Threads.nthreads()` when the file is large enough to be worth it. The
output is bitwise identical for any `chunks`.
"""
function load_fasta_packed(file::String; trim::Int = 0, skipN::Bool = false,
                           randomize_N::Bool = false, revcomp::Bool = false,
                           chunks::Int = 0, progress::Bool = true)
    raw = open(file, "r") do io
        Mmap.mmap(io)
    end
    n = length(raw)

    if chunks <= 0
        chunks = nthreads()
        chunks > 1 && (chunks = clamp(div(n, 1 << 20), 1, chunks)) # >= ~1MB per chunk
    end

    bounds = _chunk_bounds(raw, chunks)
    nchunks = length(bounds) - 1

    if nchunks == 1
        words, acc, nbits, rwords, racc, rnbits, starts, stops, heads, _ =
            _parse_chunk_packed(raw, 1, n + 1; trim, skipN, randomize_N, rc = revcomp, progress)
    else
        results = Vector{Any}(undef, nchunks)
        @sync for ci in 1:nchunks
            lo, hi = bounds[ci], bounds[ci+1]
            Threads.@spawn results[ci] = _parse_chunk_packed(raw, lo, hi;
                                                             trim, skipN, randomize_N, rc = revcomp)
        end
        # merge in order: exact bitwise concatenation of the chunk-local streams
        words = UInt32[]; acc = UInt64(0); nbits = 0
        rwords = UInt32[]; racc = UInt64(0); rnbits = 0
        starts = Int32[]; stops = Int32[]; heads = String[]
        nbases = 0
        p = progress ? Progress(nchunks; dt = 1.0) : nothing
        for ci in 1:nchunks
            cwords, cacc, cnbits, crwords, cracc, crnbits, cstarts, cstops, cheads, _ = results[ci]
            append!(starts, cstarts .+ Int32(nbases))
            append!(stops, cstops .+ Int32(nbases))
            append!(heads, cheads)
            acc, nbits = _merge_chunk!(words, acc, nbits, cwords, cacc, cnbits)
            if revcomp
                racc, rnbits = _merge_chunk!(rwords, racc, rnbits, crwords, cracc, crnbits)
            end
            nbases += isempty(cstops) ? 0 : Int(cstops[end])
            p === nothing || next!(p)
        end
    end

    # trailing partial word + one zero guard word (as in pack_reads)
    nbits > 0 && push!(words, acc % UInt32)
    push!(words, UInt32(0))
    if revcomp
        rnbits > 0 && push!(rwords, racc % UInt32)
        push!(rwords, UInt32(0))
    end

    return revcomp ? (words, starts, stops, rwords, heads) : (words, starts, stops, heads)
end

"""
    encode_fasta_cuda(re, file; normalize=0, load_kw...) -> (cropes, cropes_rc, heads)

The newindex2.jl workflow in two lines: read the FASTA (fused, packed, with the
reverse-complement batch) and encode both batches with the v3 kernels. The two
encodings correspond to the old path's `cropes[:, :, 1:n]` / `cropes[:, :, n+1:2n]`.
"""
function encode_fasta_cuda(re::RopeEncoder, file::String; normalize = 0, load_kw...)
    words, starts, stops, rwords, heads = load_fasta_packed(file; revcomp = true, load_kw...)
    cropes, _ = encode_batch_cuda_all_best_v3(re, cu(words); starts = cu(starts), stops = cu(stops), normalize = normalize)
    cropes_rc, _ = encode_batch_cuda_all_best_v3(re, cu(rwords); starts = cu(starts), stops = cu(stops), normalize = normalize)
    return cropes, cropes_rc, heads
end

# ==============================================================================

# v3-vs-v2 comparison: reuse check_pair3 from newencoder2bit.jl (shared-memory
# atomics make the F32 summation order nondeterministic; both kernels use the
# same phase math, hence its loose tolerances rtol=1e-3, atol=2e-4)

function test()
    @show CUDA.name(device())
    @show nthreads()

    dir = mktempdir(prefix = "newfasta_")
    L = 20_000
    n_reads = 2^13

    # ---- 1. generate a FASTA file with the project's Utils -------------------
    @info "Generating $n_reads mutated reads of ~$L bp and saving the FASTA"
    ref = generate_reference(2^22; seed = 1234)
    reads, pos = generate_reads(ref, L, n_reads; err = 0.02, seed = 42) # indels -> jagged lengths
    fasta = joinpath(dir, "reads.fasta")
    heads = [">read_$i pos=$(pos[i])" for i in eachindex(reads)]
    @time save_fasta(reads, fasta, heads = heads)
    @show filesize(fasta)

    # ---- 2. reference path: load_fasta_mmap_fixed + pack_reads ----------------
    @info "Loading with the reference path (load_fasta_mmap_fixed + pack_reads)"
    @time seqs_ref, heads_ref = load_fasta_mmap_fixed(fasta)
    @time words_ref, starts_ref, stops_ref = pack_reads(seqs_ref)
    seqs_rc = [UInt8(3) .- reverse(a) for a in seqs_ref]
    rwords_ref, rstarts_ref, rstops_ref = pack_reads(seqs_rc)
    @assert (rstarts_ref, rstops_ref) == (starts_ref, stops_ref) "rc segmentation must match"
    @show (sum(length, seqs_ref), sizeof(words_ref)) # 4x smaller packed stream

    # ---- 3. the fused loader: bitwise-identical streams ------------------------
    @info "Loading with load_fasta_packed (fused parse + 2-bit pack, fwd + revcomp)"
    @time words, starts, stops, rwords, heads_ld = load_fasta_packed(fasta; revcomp = true, progress = false)
    @assert words == words_ref "forward packed stream mismatch"
    @assert rwords == rwords_ref "revcomp packed stream mismatch"
    @assert starts == starts_ref && stops == stops_ref "starts/stops mismatch"
    @assert heads_ld == heads_ref "headers mismatch"

    # chunked (multithreaded) parsing: bitwise identical for any chunk count
    words1, starts1, stops1, rwords1, heads1 = load_fasta_packed(fasta; revcomp = true, chunks = 1, progress = false)
    words4, starts4, stops4, rwords4, heads4 = load_fasta_packed(fasta; revcomp = true, chunks = 4, progress = false)
    @assert words1 == words4 == words && starts1 == starts4 == starts
    @assert rwords1 == rwords4 == rwords && heads1 == heads4 == heads_ld
    @info "Single-chunk, 4-chunk and default-chunk outputs are bitwise identical"

    # ---- 4. synthetic edge cases (junk, lowercase, N, CRLF, empty/short reads) -
    syn = joinpath(dir, "synthetic.fasta")
    write(syn,
        "junk before the first record\n",   # skipped
        ">r1 desc\nACGTacgtNnGT\n",         # lowercase; N/n treated as G
        ">r2\r\nAC\r\nGTAA\r\n",            # CRLF line endings
        ">r3\n",                            # empty sequence -> zero-length read
        ">r4\nACGT\n",                      # shorter than trim -> dropped
        ">r5 last\nTTTTTTTT")               # no trailing newline
    for trim in (0, 5)
        seqs_s, heads_s = load_fasta_mmap_fixed(syn; trim = trim)
        words_s, starts_s, stops_s = pack_reads(seqs_s)
        w2, s2, e2, h2 = load_fasta_packed(syn; trim = trim, chunks = 1, progress = false)
        @assert (w2, s2, e2, h2) == (words_s, starts_s, stops_s, heads_s) "synthetic mismatch (trim=$trim)"
        w3, s3, e3, h3 = load_fasta_packed(syn; trim = trim, chunks = 3, progress = false)
        @assert (w3, s3, e3, h3) == (words_s, starts_s, stops_s, heads_s) "synthetic chunked mismatch (trim=$trim)"
        @info "Synthetic edge cases OK (trim=$trim): $(length(seqs_s)) reads, $(sum(length, seqs_s)) bases"
    end

    # ---- 5. encoding: v3 on the fused stream vs v2 on the byte path ------------
    dnas_bytes = jagged_array(seqs_ref; dev = :gpu).vect
    rdnas_bytes = jagged_array(seqs_rc; dev = :gpu).vect
    dnas_packed = cu(words)
    rdnas_packed = cu(rwords)
    starts_g = cu(starts)
    stops_g = cu(stops)

    re = RopeEncoder(k = L, s = 8, m = 4, c = 4)
    @info "Checking v3-on-fused-stream against v2-on-bytes (s=8, m=4, c=4)"
    for normalize in 0:3
        check_pair3(re, dnas_packed, dnas_bytes, starts_g, stops_g, normalize)
    end

    @info "Checking the m=1 configuration (s=5, c=4)"
    re1 = RopeEncoder(k = L, s = 5, m = 1, c = 4)
    for normalize in (0, 1)
        check_pair3(re1, dnas_packed, dnas_bytes, starts_g, stops_g, normalize)
    end

    @info "Checking the s=16 configuration (m=2, c=4)"
    re16 = RopeEncoder(k = L, s = 16, m = 2, c = 4)
    for normalize in (0, 1)
        check_pair3(re16, dnas_packed, dnas_bytes, starts_g, stops_g, normalize)
    end

    @info "Checking the reverse-complement batch"
    for normalize in (0, 1)
        check_pair3(re, rdnas_packed, rdnas_bytes, starts_g, stops_g, normalize)
    end

    # ---- 6. timings -------------------------------------------------------------
    @info "Warming the page cache"
    open(fasta, "r") do io
        while !eof(io)
            read(io, 1 << 22)
        end
    end

    normalize = 0
    @info "Timing the old path: load_fasta_mmap_fixed -> rc append -> pack_reads -> cu"
    local seqs_t, words_t, starts_t, stops_t, jt, w_g, s_g, e_g # survive the loop scope
    for _ in 1:3
        @time begin
            seqs_t, _ = load_fasta_mmap_fixed(fasta)
            seqs2_t = deepcopy(seqs_t)
            append!(seqs2_t, [UInt8(3) .- reverse(a) for a in seqs_t])
            words_t, starts_t, stops_t = pack_reads(seqs2_t)
            jt = jagged_array(seqs2_t; dev = :gpu)
            w_g = cu(words_t); s_g = cu(starts_t); e_g = cu(stops_t)
        end
    end
    CUDA.@time cropes_old, _ = encode_batch_cuda_all_best_v2(re, jt.vect; starts = s_g, stops = e_g, normalize = normalize)

    @info "Timing the new path: load_fasta_packed (fwd + revcomp in one pass) -> cu"
    local wt, st, et, rwt, w_g2, s_g2, e_g2, rw_g2 # survive the loop scope
    for _ in 1:3
        @time begin
            wt, st, et, rwt, _ = load_fasta_packed(fasta; revcomp = true, progress = false)
            w_g2 = cu(wt); s_g2 = cu(st); e_g2 = cu(et); rw_g2 = cu(rwt)
        end
    end
    CUDA.@time cropes_new, _ = encode_batch_cuda_all_best_v3(re, w_g2; starts = s_g2, stops = e_g2, normalize = normalize)
    # the old path encodes fwd+rc as one 2n batch; compare the forward half
    @info "End-to-end max diff old vs new (forward batch)" max_diff = maximum(abs.(cropes_old[:, :, 1:length(st)] .- cropes_new))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
