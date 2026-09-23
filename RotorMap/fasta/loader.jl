# =============================================================================
# loader.jl -- the v2 batched packed FASTA loader (from legacy newfasta_v2.jl,
# verbatim): record-aligned batches, SWAR pair-LUT scanning (the E1/E2 engine),
# a chunk-local buffer pool, and cross-chunk merging of reads that straddle
# chunk boundaries.  Produces bitwise-identical packed content to the v1
# `load_fasta_packed` (legacy newfasta.jl, not carried over -- superseded).
#
#   _PAIR_LUT / _pair_lut_check   2-byte -> packed-code SWAR LUT (derived from
#                                 fasta/pack.jl's _PACK_LUT at include time)
#   _V2Pool / _v2take / _v2give   chunk-local scratch-buffer pool
#   _scan_slice! / _scan_batch_parallel! / _v2_parse_chunk! / _v2_merge_chunk!
#   _parse_batch_v2               the batched parse pipeline
#   FastaBatchesV2                the streaming iterator (revcomp, swar, keep_heads)
#   load_fasta_packed_v2          whole-file convenience wrapper
#
# REQUIRES: fasta/pack.jl included FIRST (needs _PACK_LUT).
# EXTERNAL DEPS: none in the loader core (the legacy experiment mains,
#   disk/micro benches and synthetic-file builders were dropped / moved to
#   common/testref.jl + common/util.jl).
# =============================================================================
using Base.Threads
using Printf
using Mmap

# ==============================================================================
# Pair LUT: 2 bytes -> UInt8.  bits 0-1: code of the first byte, bits 2-3:
# code of the second byte, bit 4: both bytes are valid (ACGTN/acgtn, the
# _PACK_LUT domain), bit 5: at least one of the bytes is N/n.
# Invalid pairs map to 0x00 (bit 4 clear).  64 KB, L2-resident.
# ==============================================================================
const _PAIR_LUT = let
    t = Vector{UInt8}(undef, 65536)
    lut = _PACK_LUT
    N1 = UInt8('N'); N2 = UInt8('n')
    for x in 0:65535
        b0 = UInt8(x & 0xff); b1 = UInt8(x >> 8)
        v0 = lut[Int(b0) + 1]; v1 = lut[Int(b1) + 1]
        if v0 != 0xff && v1 != 0xff
            hasN = (b0 == N1 || b0 == N2 || b1 == N1 || b1 == N2)
            t[x + 1] = v0 | (v1 << 2) | 0x10 | (hasN ? 0x20 : 0x00)
        else
            t[x + 1] = 0x00
        end
    end
    t
end

function _pair_lut_check()
    lut = _PACK_LUT
    N1 = UInt8('N'); N2 = UInt8('n')
    for x in 0:65535
        b0 = UInt8(x & 0xff); b1 = UInt8(x >> 8)
        v0 = lut[Int(b0) + 1]; v1 = lut[Int(b1) + 1]
        e = _PAIR_LUT[x + 1]
        if v0 != 0xff && v1 != 0xff
            hasN = (b0 == N1 || b0 == N2 || b1 == N1 || b1 == N2)
            @assert e == v0 | (v1 << 2) | 0x10 | (hasN ? 0x20 : 0x00) "pair LUT mismatch at $x"
        else
            @assert e == 0x00 "pair LUT junk mismatch at $x"
        end
    end
    return nothing
end

# ==============================================================================
# Buffer pool: chunk-local scratch buffers, taken/given back per chunk so a
# re-parse (benchmark reps, streaming batches) allocates ~nothing steady-state.
# ==============================================================================
struct _V2Pool
    lk::Threads.SpinLock
    words::Vector{Vector{UInt32}}
    rwords::Vector{Vector{UInt32}}
    i32::Vector{Vector{Int32}}
    heads::Vector{Vector{String}}
    poss::Vector{Vector{Int64}}
    u16::Vector{Vector{UInt16}}
end
const _POOL = _V2Pool(Threads.SpinLock(), Vector{Vector{UInt32}}(), Vector{Vector{UInt32}}(),
                      Vector{Vector{Int32}}(), Vector{Vector{String}}(),
                      Vector{Vector{Int64}}(), Vector{Vector{UInt16}}())

function _v2take(list::Vector{Vector{T}}, n::Int) where T
    b = lock(() -> isempty(list) ? Vector{T}(undef, 0) : pop!(list), _POOL.lk)
    length(b) < n && resize!(b, n)
    return b
end
function _v2give(list::Vector{Vector{T}}, b::Vector{T}) where T
    lock(() -> (push!(list, b); nothing), _POOL.lk)
    return nothing
end

const _EMPTY_U32 = Vector{UInt32}(undef, 0)
const _EMPTY_STR = String[]

function _grow16!(b::Vector{UInt16})
    return resize!(b, max(2 * length(b), 4096))
end

# ==============================================================================
# Stage-time accumulators (see _v2time_print): 1 = batch record scan, 2 = output
# prefault, 3 = parse (sum over chunk tasks), 4 = merge (sum over chunk tasks)
# ==============================================================================
const _V2TIME = Float64[0.0, 0.0, 0.0, 0.0]
const _V2TLK = Threads.SpinLock()
_v2time_reset!() = lock(() -> fill!(_V2TIME, 0.0), _V2TLK)
function _v2time_add!(i::Int, ns::Float64)
    lock(() -> (_V2TIME[i] += ns; nothing), _V2TLK)
end
function _v2time_print()
    lab = ("batch record scan (parallel)", "output prefault", "parse (sum of tasks)", "merge (sum of tasks)")
    tot = sum(_V2TIME)
    for i in 1:4
        @printf("  stage %-28s %9.1f ms  %5.1f%%\n", lab[i], _V2TIME[i] / 1e6, 100 * _V2TIME[i] / max(tot, eps()))
    end
end

# ==============================================================================
# line-start '>' positions of slice [lo, hi) (pooled vector, reused)
function _scan_slice!(v::Vector{Int64}, raw, lo::Int, hi::Int)
    empty!(v)
    length(raw) == 0 && return v
    base = pointer(raw)
    p = base + (lo - 1)
    pend = base + (hi - 1)
    NL = UInt8('\n'); CR = UInt8('\r')
    while p < pend
        q = ccall(:memchr, Ptr{UInt8}, (Ptr{UInt8}, Cint, Csize_t), p, Int('>'), Csize_t(pend - p))
        q == C_NULL && break
        i = Int(q - base) + 1
        if i == 1 || raw[i - 1] == NL || raw[i - 1] == CR
            push!(v, i)
        end
        p = q + 1
    end
    return v
end

# Parallel batch scan: slices of the remaining file [pos, n] are memchr'd in
# parallel; the ordered concatenation of line-start '>' positions yields the
# batch's records and its exclusive end (the (nrec+1)-th record start).
function _scan_batch_parallel!(poss_all::Vector{Int64}, raw, pos::Int, nrec::Int, nscan::Int)
    empty!(poss_all)
    n = length(raw)
    n == 0 && return 1
    nscan = clamp(nscan, 1, 256)
    tot = n - pos + 1
    step = cld(tot, nscan)
    parts = Vector{Vector{Int64}}(undef, nscan)
    @sync for ci in 1:nscan
        Threads.@spawn begin
            lo = pos + (ci - 1) * step
            hi2 = min(n, pos + ci * step - 1)
            v = _v2take(_POOL.poss, 4096)
            _scan_slice!(v, raw, lo, hi2 + 1) # no-op for empty ranges; MUST
            # run even when lo > hi2 to clear the pooled vector's stale content
            parts[ci] = v
        end
    end
    for ci in 1:nscan
        v = parts[ci]
        need = nrec + 1 - length(poss_all)
        need <= 0 && break
        if length(v) <= need
            append!(poss_all, v)
        else
            append!(poss_all, @view v[1:need])
            break
        end
    end
    hi = n + 1
    length(poss_all) == nrec + 1 && (hi = pop!(poss_all)) # the record AFTER the batch
    for ci in 1:nscan
        _v2give(_POOL.poss, parts[ci])
    end
    return hi
end

# ==============================================================================
# v2 chunk parse: parses window [lo, hi) (record-aligned), whose record starts
# are in `poss`. Fills pooled buffers, returns a _ChunkParse.
#
# Collection is ONE pass per byte (vs baseline's push!(buf) collect + reversed
# re-emission): codes are appended 8 at a time into 16-bit "groups" in
# `scratch`, with a sub-group tail (pending, < 8 codes). At the end of the
# read the groups are emitted reversed (the v3 base-order fix; the eight
# 2-bit codes of each group are field-reversed with three shift-mask steps),
# and the reverse-complement stream is emitted from the same groups (xor
# 0xffff per full group) in forward order.
#
# The SWAR fast path (swar=true, E2; disabled by randomize_N because N needs
# per-byte rand consumption) processes 8 bytes at a time through _PAIR_LUT;
# any block containing junk (newlines, other letters) falls back to the
# byte-exact scalar path. A pending tail is folded into the next stored group
# (prepend + keep the overflow), so the fast path stays active everywhere.
# ==============================================================================
function _v2_parse_chunk!(raw, lo::Int, hi::Int, poss::AbstractVector{Int64};
                          trim::Int, skipN::Bool, randomize_N::Bool,
                          swar::Bool, rc::Bool, keep_heads::Bool)
    nrec = length(poss)
    win = hi - lo
    capw = cld(win, 16) + 2  # stream words upper bound: bases <= win bytes

    words = _v2take(_POOL.words, capw)
    rwords = rc ? _v2take(_POOL.rwords, capw) : _EMPTY_U32
    starts = _v2take(_POOL.i32, max(nrec, 1))
    stops = _v2take(_POOL.i32, max(nrec, 1))
    heads = keep_heads ? _v2take(_POOL.heads, max(nrec, 1)) : _EMPTY_STR
    scratch = _v2take(_POOL.u16, 4096)
    scap = length(scratch)

    lut = _PACK_LUT
    lut16 = _PAIR_LUT
    NL = UInt8('\n'); CR = UInt8('\r')
    N1 = UInt8('N'); N2 = UInt8('n')
    pbase = pointer(raw)
    use_swar = swar && !randomize_N

    wlen = 0; acc = UInt64(0); nbits = 0
    rlen = 0; racc = UInt64(0); rnbits = 0
    nk = 0; nbases = 0

    @inbounds for ri in 1:nrec
        p0 = Int(poss[ri])
        re = ri < nrec ? Int(poss[ri + 1]) : hi  # exclusive region end

        # --- header line (kept with the leading '>'), tolerate CRLF ---------
        hstart = p0
        i = p0
        while i < re
            raw[i] == NL && break
            i += 1
        end
        hend = i - 1
        (hend >= hstart && raw[hend] == CR) && (hend -= 1)
        i += 1 # past the newline

        # --- collect codes into groups (scratch) + sub-group tail -----------
        ng = 0; pending = UInt64(0); pnbits = 0; hasN = false
        if use_swar
            while i < re
                if i + 8 <= re
                    x = unsafe_load(Ptr{UInt64}(pbase + (i - 1))) # 8 bytes at raw[i]
                    q0 = lut16[Int(x & 0x000000000000ffff) + 1]
                    q1 = lut16[Int((x >> 16) & 0x000000000000ffff) + 1]
                    q2 = lut16[Int((x >> 32) & 0x000000000000ffff) + 1]
                    q3 = lut16[Int(x >> 48) + 1]
                    if (q0 & q1 & q2 & q3) & 0x10 != 0  # all 8 bytes are bases
                        if (q0 | q1 | q2 | q3) & 0x20 != 0 # block contains N/n
                            skipN && (hasN = true; break)
                        end
                        g = UInt32(q0 & 0x0f) | UInt32(q1 & 0x0f) << 4 |
                            UInt32(q2 & 0x0f) << 8 | UInt32(q3 & 0x0f) << 12
                        (ng < scap) || (scratch = _grow16!(scratch); scap = length(scratch))
                        if pnbits == 0
                            scratch[ng + 1] = g % UInt16
                        else
                            # prepend the pending codes, keep the overflow tail
                            k = 16 - pnbits
                            scratch[ng + 1] = (pending | ((g & ((UInt32(1) << k) - 1)) << pnbits)) % UInt16
                            pending = g >> k
                        end
                        ng += 1
                        i += 8
                        continue
                    end
                end
                c = raw[i]; i += 1
                (c == NL || c == CR) && continue
                v = lut[Int(c) + 1]
                v == 0xff && continue
                if c == N1 || c == N2
                    skipN && (hasN = true; break)
                end
                pending |= UInt64(v) << pnbits
                pnbits += 2
                if pnbits == 16
                    (ng < scap) || (scratch = _grow16!(scratch); scap = length(scratch))
                    scratch[ng + 1] = pending % UInt16
                    ng += 1; pending = UInt64(0); pnbits = 0
                end
            end
        else
            while i < re
                c = raw[i]; i += 1
                v = lut[Int(c) + 1]
                v == 0xff && continue
                if c == N1 || c == N2
                    if skipN
                        hasN = true; break
                    elseif randomize_N
                        v = rand(UInt8(0):UInt8(3))
                    end
                end
                pending |= UInt64(v) << pnbits
                pnbits += 2
                if pnbits == 16
                    (ng < scap) || (scratch = _grow16!(scratch); scap = length(scratch))
                    scratch[ng + 1] = pending % UInt16
                    ng += 1; pending = UInt64(0); pnbits = 0
                end
            end
        end

        # --- finalize the read: trailing tail becomes the last group --------
        npart = pnbits >> 1
        L = 8 * ng + npart
        if pnbits > 0
            (ng < scap) || (scratch = _grow16!(scratch); scap = length(scratch))
            scratch[ng + 1] = pending % UInt16
        end

        kept = !(hasN && skipN) && (trim == 0 || L >= trim)
        if kept
            keep = trim == 0 ? L : trim
            nk += 1
            starts[nk] = Int32(nbases + 1)
            nbases += keep
            stops[nk] = Int32(nbases)
            keep_heads && (heads[nk] = String(raw[hstart:hend]))

            if keep > 0
                nfull = keep >> 3
                r = keep & 7
                # forward batch: REVERSED base order (the v3 endianness fix)
                if r > 0
                    w16 = scratch[nfull + 1] # bases 8*nfull+1 .. 8*nfull+r in the low bits
                    k = 2 * r - 2
                    while k >= 0
                        acc |= UInt64((w16 >> k) & UInt16(3)) << nbits
                        nbits += 2
                        if nbits >= 32
                            wlen += 1; words[wlen] = acc % UInt32
                            acc >>= 32; nbits -= 32
                        end
                        k -= 2
                    end
                end
                for gi in nfull:-1:1
                    # reverse the 8 two-bit codes WITHIN the group (the group
                    # stores them in sequence order; the stream wants reversed)
                    g16 = scratch[gi]
                    g16 = (g16 >> 2) & 0x3333 | (g16 & 0x3333) << 2
                    g16 = (g16 >> 4) & 0x0f0f | (g16 & 0x0f0f) << 4
                    g16 = (g16 >> 8) | (g16 & 0x00ff) << 8
                    acc |= UInt64(g16) << nbits
                    nbits += 16
                    if nbits >= 32
                        wlen += 1; words[wlen] = acc % UInt32
                        acc >>= 32; nbits -= 32
                    end
                end
                # reverse-complement batch: FORWARD order, complemented codes
                if rc
                    for gi in 1:nfull
                        racc |= UInt64(scratch[gi] ⊻ UInt16(0xffff)) << rnbits
                        rnbits += 16
                        if rnbits >= 32
                            rlen += 1; rwords[rlen] = racc % UInt32
                            racc >>= 32; rnbits -= 32
                        end
                    end
                    if r > 0
                        w16 = scratch[nfull + 1]
                        mask = (UInt16(1) << (2 * r)) - UInt16(1) # padding must stay 0
                        racc |= UInt64((w16 ⊻ UInt16(0xffff)) & mask) << rnbits
                        rnbits += 2 * r
                        if rnbits >= 32
                            rlen += 1; rwords[rlen] = racc % UInt32
                            racc >>= 32; rnbits -= 32
                        end
                    end
                end
            end
        end
    end

    return _ChunkParse(words, wlen, acc, nbits, rwords, rlen, racc, rnbits,
                       starts, stops, heads, nk, nbases, scratch)
end

# Concrete chunk-parse result (so an Array of results stays type-stable)
struct _ChunkParse
    words::Vector{UInt32}; wlen::Int; acc::UInt64; nbits::Int
    rwords::Vector{UInt32}; rlen::Int; racc::UInt64; rnbits::Int
    starts::Vector{Int32}; stops::Vector{Int32}; heads::Vector{String}; nkept::Int
    nbases::Int
    scratch::Vector{UInt16}
end

# Per-chunk merge state, computed by a cheap serial prefix pass over the
# chunk results: bit offset of the chunk in the batch stream, output word
# offset, pending tail bits, and output offsets for offsets/heads.
struct _Prefix
    bb::Int64             # kept bases before this chunk (bit offset = 2*bb)
    wo::Int               # complete output words before this chunk
    ko::Int; ro::Int      # kept reads / rc words before this chunk
    gacc::UInt64; gnb::Int
    racc::UInt64; rnb::Int
end

# Parallel merge: a serial prefix pass fixes each chunk's bit/word output
# offsets from the chunk results, then every chunk funnels its OWN words into
# its own output range -- each output word depends only on the chunk's words
# plus the predecessor's pending tail bits (known from the prefix), so the
# merges are fully parallel with no cross-chunk dependency chain.
#
# Chunk c writes:
#   out[W+1]   = pred_pending | w[1] << P        (pred bits + its head)
#   out[W+j]   = w[j-1] >>> (32-P) | w[j] << P   (constant-shift funnel)
#   tail word  = carry | acc << P -- only when P + t >= 32 (a complete word);
#                otherwise the tail lives in the NEXT chunk's first word, and
#                the batch-final leftover is flushed by the serial finalize.
function _v2_merge_chunk!(gwords, grwords, gstarts, gstops, gheads,
                          rp::_ChunkParse, pf::_Prefix, rc::Bool)
    P = pf.gnb; gpos = pf.wo
    w = rp.words; n = rp.wlen
    if n > 0
        @inbounds begin
            w1 = UInt64(w[1])
            gwords[gpos + 1] = (pf.gacc | w1 << P) % UInt32
            gpos += 1
            carry = w1 >> (32 - P)
            for j in 2:n
                wj = UInt64(w[j])
                gwords[gpos + 1] = (carry | wj << P) % UInt32
                gpos += 1
                carry = wj >> (32 - P)
            end
            if rp.nbits > 0
                v = carry | rp.acc << P
                (P + rp.nbits >= 32) && (gwords[gpos + 1] = v % UInt32)
            end
        end
    elseif rp.nbits > 0 && (P + rp.nbits >= 32)
        @inbounds gwords[gpos + 1] = (pf.gacc | rp.acc << P) % UInt32
    end

    if rc
        P = pf.rnb; rpos = pf.ro
        rw = rp.rwords; n = rp.rlen
        if n > 0
            @inbounds begin
                w1 = UInt64(rw[1])
                grwords[rpos + 1] = (pf.racc | w1 << P) % UInt32
                rpos += 1
                carry = w1 >> (32 - P)
                for j in 2:n
                    wj = UInt64(rw[j])
                    grwords[rpos + 1] = (carry | wj << P) % UInt32
                    rpos += 1
                    carry = wj >> (32 - P)
                end
                if rp.rnbits > 0
                    v = carry | rp.racc << P
                    (P + rp.rnbits >= 32) && (grwords[rpos + 1] = v % UInt32)
                end
            end
        elseif rp.rnbits > 0 && (P + rp.rnbits >= 32)
            @inbounds grwords[rpos + 1] = (pf.racc | rp.racc << P) % UInt32
        end
    end

    off = Int32(pf.bb) # batch-local: < 2^31 bases per batch
    gk = pf.ko
    if length(gheads) > 0
        @inbounds for j in 1:rp.nkept
            gstarts[gk + j] = rp.starts[j] + off
            gstops[gk + j] = rp.stops[j] + off
            gheads[gk + j] = rp.heads[j]
        end
    else
        @inbounds for j in 1:rp.nkept
            gstarts[gk + j] = rp.starts[j] + off
            gstops[gk + j] = rp.stops[j] + off
        end
    end
    return nothing
end

# Parse one batch starting at byte `pos`:
#   round A: parallel record scan (one memchr pass over slices of the file)
#   round B: parallel chunk parse into pooled buffers (no merging)
#   serial prefix pass over the (tiny) chunk results: output offsets
#   round C: parallel merge (carry-free funnels into disjoint ranges)
function _parse_batch_v2(it, pos::Int)
    raw = it.raw
    poss_all = _v2take(_POOL.poss, it.batch_reads + 1)
    nscan = it.chunks > 0 ? it.chunks : nthreads()
    t0 = time_ns()
    hi = _scan_batch_parallel!(poss_all, raw, pos, it.batch_reads, nscan)
    _v2time_add!(1, Float64(time_ns() - t0))
    nrec_total = length(poss_all)

    if nrec_total == 0 # junk-only tail (or empty file): an empty batch
        _v2give(_POOL.poss, poss_all)
        gwords = UInt32[0]
        grwords = it.revcomp ? UInt32[0] : UInt32[]
        return (words = gwords, starts = Int32[], stops = Int32[],
                rwords = it.revcomp ? grwords : nothing, heads = String[]), hi
    end

    nch = it.chunks
    if nch <= 0
        nch = nthreads()
        nch > 1 && (nch = clamp(div(hi - pos, 1 << 20), 1, nch)) # >= ~1MB per chunk
    end
    nch = min(nch, nrec_total)

    # record-aligned chunk bounds from the position vector (no extra scan)
    cuts = [round(Int, nrec_total * ci / nch) for ci in 0:nch]
    cuts[1] = 0; cuts[end] = nrec_total

    capw = cld(hi - pos, 16) + 2
    gwords = Vector{UInt32}(undef, capw)
    grwords = it.revcomp ? Vector{UInt32}(undef, capw) : Vector{UInt32}(undef, 0)
    ncap = max(nrec_total, 1)
    gstarts = Vector{Int32}(undef, ncap)
    gstops = Vector{Int32}(undef, ncap)
    gheads = it.keep_heads ? Vector{String}(undef, ncap) : String[]

    t0 = time_ns()
    @sync for ci in 1:nch # parallel page-touch: round C must not fault
        Threads.@spawn begin
            lo = 1 + (ci - 1) * div(capw, nch)
            hi2 = ci == nch ? capw : 1 + ci * div(capw, nch)
            for j in lo:4096:min(hi2, capw)
                @inbounds gwords[j] = 0
            end
            if it.revcomp
                for j in lo:4096:min(hi2, capw)
                    @inbounds grwords[j] = 0
                end
            end
        end
    end
    _v2time_add!(2, Float64(time_ns() - t0))

    # round B: parse chunks in parallel
    results = Vector{_ChunkParse}(undef, nch)
    t0 = time_ns()
    @sync for ci in 1:nch
        Threads.@spawn begin
            clo = Int(poss_all[cuts[ci] + 1])
            chi = ci < nch ? Int(poss_all[cuts[ci + 1] + 1]) : hi
            poss = @view poss_all[cuts[ci] + 1:cuts[ci + 1]]
            tp = time_ns()
            r = _v2_parse_chunk!(raw, clo, chi, poss;
                                 trim = it.trim, skipN = it.skipN,
                                 randomize_N = it.randomize_N, swar = it.swar,
                                 rc = it.revcomp, keep_heads = it.keep_heads)
            parse_ns = Float64(time_ns() - tp)
            lock(() -> (_V2TIME[3] += parse_ns; nothing), _V2TLK)
            results[ci] = r
        end
    end
    _v2give(_POOL.poss, poss_all)

    # serial prefix: output offsets per chunk
    prefix = Vector{_Prefix}(undef, nch)
    bb = Int64(0); wo = 0; ko = 0; ro = 0
    gacc = UInt64(0); gnb = 0; racc = UInt64(0); rnb = 0
    for ci in 1:nch
        rp = results[ci]
        prefix[ci] = _Prefix(bb, wo, ko, ro, gacc, gnb, racc, rnb)
        bb += rp.nbases; ko += rp.nkept
        gacc = rp.wlen > 0 ? UInt64(rp.words[rp.wlen]) >> (32 - gnb) : gacc
        wo += rp.wlen
        if rp.nbits > 0
            gacc |= rp.acc << gnb
            gnb += rp.nbits
            if gnb >= 32
                gacc >>= 32; gnb -= 32
                wo += 1
            end
        end
        if it.revcomp
            racc = rp.rlen > 0 ? UInt64(rp.rwords[rp.rlen]) >> (32 - rnb) : racc
            ro += rp.rlen
            if rp.rnbits > 0
                racc |= rp.racc << rnb
                rnb += rp.rnbits
                if rnb >= 32
                    racc >>= 32; rnb -= 32
                    ro += 1
                end
            end
        end
    end

    # round C: parallel merge into disjoint output ranges
    @sync for ci in 1:nch
        Threads.@spawn begin
            tm = time_ns()
            r = results[ci]
            _v2_merge_chunk!(gwords, grwords, gstarts, gstops, gheads,
                             r, prefix[ci], it.revcomp)
            merge_ns = Float64(time_ns() - tm)
            lock(() -> (_V2TIME[4] += merge_ns; nothing), _V2TLK)
            _v2give(_POOL.words, r.words)
            it.revcomp && _v2give(_POOL.rwords, r.rwords)
            _v2give(_POOL.i32, r.starts)
            _v2give(_POOL.i32, r.stops)
            it.keep_heads && _v2give(_POOL.heads, r.heads)
            _v2give(_POOL.u16, r.scratch)
        end
    end

    # finalize: leftover pending word + one zero guard word (as in pack_reads)
    gwl = wo
    if gnb > 0
        gwl += 1
        gwords[gwl] = gacc % UInt32
    end
    gwl += 1
    gwords[gwl] = UInt32(0)
    resize!(gwords, gwl)
    if it.revcomp
        grl = ro
        if rnb > 0
            grl += 1
            grwords[grl] = racc % UInt32
        end
        grl += 1
        grwords[grl] = UInt32(0)
        resize!(grwords, grl)
    end
    resize!(gstarts, ko)
    resize!(gstops, ko)
    heads_out = it.keep_heads ? resize!(gheads, ko) : String[]

    nt = (words = gwords, starts = gstarts, stops = gstops,
          rwords = it.revcomp ? grwords : nothing, heads = heads_out)
    return nt, hi
end

# ==============================================================================
# The v2 lazy batch iterator + whole-file convenience wrapper.
# ==============================================================================
struct FastaBatchesV2
    raw::Vector{UInt8}
    batch_reads::Int
    chunks::Int
    trim::Int
    skipN::Bool
    randomize_N::Bool
    revcomp::Bool
    swar::Bool
    keep_heads::Bool
end

function FastaBatchesV2(file::String; batch_reads::Int = 2^15, chunks::Int = 0,
                        trim::Int = 0, skipN::Bool = false, randomize_N::Bool = false,
                        revcomp::Bool = true, swar::Bool = true, keep_heads::Bool = true)
    @assert batch_reads > 0
    raw = filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end
    return FastaBatchesV2(raw, batch_reads, chunks, trim, skipN, randomize_N, revcomp, swar, keep_heads)
end

Base.eltype(::Type{FastaBatchesV2}) = Any # variable-length named tuples

function Base.iterate(it::FastaBatchesV2, pos::Int = 1)
    pos > length(it.raw) && return nothing
    return _parse_batch_v2(it, pos) # (batch, next position)
end

"""
    load_fasta_packed_v2(file; revcomp=true, swar=true, chunks=0, trim=0,
                         skipN=false, randomize_N=false, keep_heads=true)
        -> (words, starts, stops, heads)          # revcomp = false
        -> (words, starts, stops, rwords, heads)  # revcomp = true

The whole file as one batch -- same output shapes and (for files below the
2^31-base Int32 offset limit) bitwise-identical content to `load_fasta_packed`.
"""
function load_fasta_packed_v2(file::String; revcomp::Bool = true, kw...)
    it = FastaBatchesV2(file; revcomp, batch_reads = 10^9, kw...)
    nt, _ = iterate(it)
    return revcomp ? (nt.words, nt.starts, nt.stops, nt.rwords, nt.heads) :
                     (nt.words, nt.starts, nt.stops, nt.heads)
end
