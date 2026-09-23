# ==============================================================================
# newfasta_v2.jl -- bottleneck hunt + v2 experiments for the FASTA -> 2-bit
# packed bitstream loader, at 2^17 reads x ~20,000 bp (~2.6 GB FASTA).
#
# QUESTION (prompts/newfasta_v2.md): the newfasta.jl / newflow.jl loader does
# not saturate disk read speeds when using 16 CPU cores. Where does the time
# actually go, and what removes the bottleneck?
#
# Known context (prompts/newflow.md section 4): on kau, warm page-cache reads
# run at ~19 GB/s and cold NVMe at ~2.0-2.3 GB/s, while the loader consumes
# only 0.42-0.72 GB/s at 1-16 threads -- so neither the disk nor DRAM was ever
# the ceiling; the suspects are the scalar per-byte parse loop and the GC
# churn of push!-grown output arrays. This file measures all of that at the
# larger scale and experiments with fixes:
#
#   E0  baseline   : load_fasta_packed / FastaBatches (newfasta.jl/newflow.jl)
#   E1  alloc-free : same scalar per-byte parse, but no push!-growth anywhere:
#                    exact preallocated outputs (upper bound + resize!), pooled
#                    and reused chunk buffers, memchr record scan, and a
#                    pipelined parallel merge into exact buffers
#   E2  swar       : E1 + a 64 KB pair-LUT fast path (8 bytes per iteration,
#                    branchless ACGTN decode + validity bit, byte-exact
#                    fallback for any block containing junk)
#
# The v2 loader (FastaBatchesV2 / load_fasta_packed_v2) produces bitwise
# identical output to the baseline (asserted in run_test) and keeps the same
# semantics (trim/skipN/randomize_N/revcomp, headers in sync, per-batch Int32
# offsets). Like newflow, files with more than 2^31 bases must be read in
# batches (batch_reads) because offsets are Int32 -- the 2^17-read file (2.6e9
# bases) is exactly such a case, which is why the big-file benches run batched
# (batch_reads = 2^15 -> 4 batches of ~655M bases) on BOTH implementations.
#
# Run modes (ARGV[1]):
#   gen     generate the test files if missing (cached under NEWFASTA_V2_DIR)
#   test    correctness: v2 == baseline, bitwise, small file + big file +
#           synthetic edge cases + pair-LUT self-check
#   disk    read-bandwidth ceiling: warm read!, cold read! (posix_fadvise
#           DONTNEED), cold mmap page-in
#   micro   isolated microbenchmarks: array growth (push!/sizehint!/prealloc)
#           and per-byte parse-loop variants (baseline 2-pass vs fused vs SWAR)
#   bench   loader timings E0/E1/E2 (+ stage breakdown + heads cost)
#   profile flat flame summary of E0 vs E2
#   all     everything above (default)
#
# Thread scaling must be measured by running the script under different
# `julia -t N` (nthreads cannot be changed at runtime).
# ==============================================================================

include(joinpath(@__DIR__, "newflow.jl")) # baseline FastaBatches + load_fasta_packed
                                           # (+ transitive: newfasta.jl -> newencoder2bit.jl)

using Random
using Printf
using Mmap
using Profile
using RotorMap
using RotorMap.Utils
using Base.Threads

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

# ==============================================================================
# Data generation (cached under NEWFASTA_V2_DIR, default /tmp)
# ==============================================================================
const _SMALL_READS = 2^13  #  8192 reads -> 164 MB, whole-file comparisons legal
const _BIG_READS = 2^17    # 131072 reads -> ~2.6 GB, EXCEEDS the Int32 offset
                           # limit whole-file: benches must batch
const _READ_LEN = 20_000

_data_dir() = get(ENV, "NEWFASTA_V2_DIR", "/tmp")

function _ensure_file(n_reads::Int)
    path = joinpath(_data_dir(), "newfasta_v2_$(n_reads).fasta")
    isfile(path) && filesize(path) > n_reads * 19_000 && return path
    @info "Generating $(n_reads) mutated reads of ~$(_READ_LEN) bp -> $path"
    mkpath(_data_dir())
    ref = generate_reference(2^22; seed = 1234)
    reads, pos = generate_reads(ref, _READ_LEN, n_reads; err = 0.02, seed = 42)
    heads = [">read_$i pos=$(pos[i])" for i in eachindex(reads)]
    t = @elapsed save_fasta(reads, path, heads = heads)
    @info "Wrote $path ($(filesize(path)) bytes) in $(round(t, digits = 1)) s"
    return path
end
ensure_data() = (_ensure_file(_SMALL_READS), _ensure_file(_BIG_READS))

_reps() = parse(Int, get(ENV, "NEWFASTA_V2_REPS", "3"))

# ==============================================================================
# Measurement helpers
# ==============================================================================
const _SINK = Ref(0)

function _timed_min(f, label::String; bytes::Int, reps::Int)
    best_t = Inf; best_gc = 0.0; best_alloc = 0
    for r in 1:reps
        GC.gc(); GC.gc()
        g0 = Base.gc_num()
        t0 = time_ns()
        out = f()
        dt = (time_ns() - t0) / 1e9
        g1 = Base.gc_num()
        r == 1 && (_SINK[] += out) # keep/observe the result
        if dt < best_t
            best_t = dt
            best_gc = (g1.total_time - g0.total_time) / 1e9
            best_alloc = g1.total_allocd - g0.total_allocd
        end
    end
    @printf("  %-48s %7.3f s  %7.2f GB/s  alloc %8.1f MiB  gc %5.1f%%\n",
            label, best_t, bytes / best_t / 1e9, best_alloc / 2^20, 100 * best_gc / best_t)
    return best_t
end

function _warm_cache(file)
    open(file, "r") do io
        buf = Vector{UInt8}(undef, 1 << 22)
        sz = filesize(file)
        while !eof(io)
            nb = min(length(buf), sz - position(io))
            read!(io, view(buf, 1:nb))
        end
    end
    return nothing
end

function _majflt()
    v = Vector{Clong}(undef, 18)
    ccall(:getrusage, Cint, (Cint, Ptr{Clong}), 0, v)
    return Int(v[10]) # ru_majflt (2+2 timevals, maxrss, ixrss, idrss, isrss, minflt)
end

# ==============================================================================
# disk: the read-bandwidth ceiling
# ==============================================================================
function _fadvise_drop!(file)
    io = open(file, "r")
    fd = Base.fd(io)
    ccall(:posix_fadvise, Cint, (Cint, Clonglong, Clonglong, Cint), fd, 0, 0, 4) # DONTNEED
    close(io)
    return nothing
end

function run_disk(; reps = 3)
    small, big = ensure_data()
    n = filesize(big)
    @info "Disk ceiling on $(basename(big)) ($(n) bytes)"

    t = Inf
    for _ in 1:reps
        GC.gc()
        t0 = time_ns()
        _warm_cache(big)
        t = min(t, (time_ns() - t0) / 1e9)
    end
    @printf("  %-48s %7.3f s  %7.2f GB/s\n", "warm read! (page cache)", t, n / t / 1e9)

    t = Inf
    for _ in 1:reps
        GC.gc()
        _fadvise_drop!(big)
        t0 = time_ns()
        _warm_cache(big)
        t = min(t, (time_ns() - t0) / 1e9)
    end
    @printf("  %-48s %7.3f s  %7.2f GB/s\n", "cold read! (fadvise DONTNEED first)", t, n / t / 1e9)

    # cold mmap page-in only (one byte per page)
    GC.gc()
    _fadvise_drop!(big)
    f0 = _majflt()
    t0 = time_ns()
    raw = open(big, "r") do io
        Mmap.mmap(io)
    end
    s = 0
    @inbounds for j in 1:4096:length(raw)
        s += raw[j]
    end
    t = (time_ns() - t0) / 1e9
    flt = _majflt() - f0
    _SINK[] += s
    @printf("  %-48s %7.3f s  %7.2f GB/s  (major faults: %d)\n",
            "cold mmap page-in (1 byte / 4 KB page)", t, n / t / 1e9, flt)

    _warm_cache(big) # leave the cache warm for the other benches
    return nothing
end

# ==============================================================================
# microbenchmarks: allocation churn + per-byte loop cost in isolation
# ==============================================================================
function _micro_growth(; total_words = 163_840_000, reps = 3) # ~654 MB, one big-file stream
    ntasks = nthreads()
    per = cld(total_words, ntasks)
    @info "Micro: filling ~$(round(Int, total_words * 4 / 2^20)) MiB of UInt32 across $ntasks tasks"
    for mode in (:push, :sizehint, :prealloc)
        _timed_min("growth mode: $(mode)"; bytes = total_words * 4, reps) do
            Threads.@sync for _ in 1:ntasks
                Threads.@spawn begin
                    if mode === :push
                        w = UInt32[]
                        for j in 1:per
                            push!(w, j % UInt32)
                        end
                        _SINK[] += length(w)
                    elseif mode === :sizehint
                        w = UInt32[]
                        sizehint!(w, per)
                        for j in 1:per
                            push!(w, j % UInt32)
                        end
                        _SINK[] += length(w)
                    else
                        w = Vector{UInt32}(undef, per)
                        for j in 1:per
                            @inbounds w[j] = j % UInt32
                        end
                        _SINK[] += length(w)
                    end
                end
            end
            total_words
        end
    end
    return nothing
end

# per-byte parse loop variants over a real 32 MB slice of the big file
function _micro_loops(big; mb = 32, reps = 3)
    raw = open(big, "r") do io
        Mmap.mmap(io)
    end
    lo = 1
    hi = min(1 + (mb << 20), length(raw))
    bytes = hi - lo
    lut = _PACK_LUT
    lut16 = _PAIR_LUT
    NL = UInt8('\n')
    pbase = pointer(raw)

    @info "Micro: per-byte loop variants over $mb MB of real FASTA (single task)"

    _timed_min("loop: baseline 2-pass (push! buf + reversed re-emit)"; bytes, reps) do
        words = UInt32[]; buf = UInt8[]
        acc = UInt64(0); nbits = 0; n = 0
        i = lo
        while i < hi
            b = raw[i]
            if b == UInt8('>')
                while i < hi
                    b = raw[i]
                    b == NL && break
                    i += 1
                end
                i += 1
                empty!(buf)
                while i < hi
                    c = raw[i]
                    c == UInt8('>') && break
                    i += 1
                    v = lut[Int(c) + 1]
                    v == 0xff && continue
                    push!(buf, v)
                end
                for j in length(buf):-1:1
                    acc |= UInt64(buf[j] & 3) << nbits
                    nbits += 2
                    if nbits == 32
                        push!(words, acc % UInt32)
                        acc = UInt64(0); nbits = 0
                    end
                end
                n += length(buf)
            else
                i += 1
            end
        end
        length(words) + n
    end

    _timed_min("loop: fused one-pass, scalar collect (E1 inner)"; bytes, reps) do
        scratch = Vector{UInt16}(undef, 1 << 22)
        words = Vector{UInt32}(undef, (bytes >> 4) + 2)
        wi = 0
        acc = UInt64(0); nbits = 0
        ng = 0; pending = UInt64(0); pnbits = 0
        for i in lo:hi-1
            c = raw[i]
            v = lut[Int(c) + 1]
            v == 0xff && continue
            pending |= UInt64(v) << pnbits
            pnbits += 2
            if pnbits == 16
                scratch[ng + 1] = pending % UInt16
                ng += 1; pending = UInt64(0); pnbits = 0
            end
        end
        for gi in ng:-1:1
            g16 = scratch[gi]
            g16 = (g16 >> 2) & 0x3333 | (g16 & 0x3333) << 2
            g16 = (g16 >> 4) & 0x0f0f | (g16 & 0x0f0f) << 4
            g16 = (g16 >> 8) | (g16 & 0x00ff) << 8
            acc |= UInt64(g16) << nbits
            nbits += 16
            if nbits >= 32
                wi += 1; words[wi] = acc % UInt32
                acc >>= 32; nbits -= 32
            end
        end
        wi + ng
    end

    _timed_min("loop: fused one-pass, pair-LUT SWAR (E2 inner)"; bytes, reps) do
        scratch = Vector{UInt16}(undef, 1 << 22)
        words = Vector{UInt32}(undef, (bytes >> 4) + 2)
        wi = 0
        acc = UInt64(0); nbits = 0
        ng = 0; pending = UInt64(0); pnbits = 0
        i = lo
        while i < hi
            if i + 8 <= hi
                x = unsafe_load(Ptr{UInt64}(pbase + (i - 1))) # 8 bytes at raw[i]
                q0 = lut16[Int(x & 0x000000000000ffff) + 1]
                q1 = lut16[Int((x >> 16) & 0x000000000000ffff) + 1]
                q2 = lut16[Int((x >> 32) & 0x000000000000ffff) + 1]
                q3 = lut16[Int(x >> 48) + 1]
                if (q0 & q1 & q2 & q3) & 0x10 != 0
                    g = UInt32(q0 & 0x0f) | UInt32(q1 & 0x0f) << 4 | UInt32(q2 & 0x0f) << 8 | UInt32(q3 & 0x0f) << 12
                    if pnbits == 0
                        scratch[ng + 1] = g % UInt16
                    else
                        k = 16 - pnbits
                        scratch[ng + 1] = (pending | ((g & ((UInt32(1) << k) - 1)) << pnbits)) % UInt16
                        pending = g >> k
                    end
                    ng += 1; i += 8
                    continue
                end
            end
            c = raw[i]; i += 1
            v = lut[Int(c) + 1]
            v == 0xff && continue
            pending |= UInt64(v) << pnbits
            pnbits += 2
            if pnbits == 16
                scratch[ng + 1] = pending % UInt16
                ng += 1; pending = UInt64(0); pnbits = 0
            end
        end
        for gi in ng:-1:1
            g16 = scratch[gi]
            g16 = (g16 >> 2) & 0x3333 | (g16 & 0x3333) << 2
            g16 = (g16 >> 4) & 0x0f0f | (g16 & 0x0f0f) << 4
            g16 = (g16 >> 8) | (g16 & 0x00ff) << 8
            acc |= UInt64(g16) << nbits
            nbits += 16
            if nbits >= 32
                wi += 1; words[wi] = acc % UInt32
                acc >>= 32; nbits -= 32
            end
        end
        wi + ng
    end
    return nothing
end

function run_micro()
    small, big = ensure_data()
    _warm_cache(big)
    _micro_growth()
    _micro_loops(big)
    return nothing
end

# ==============================================================================
# correctness
# ==============================================================================
function _lockstep(file, batch_reads::Int)
    # count baseline batches first (zip stops at the shorter iterator, so
    # equality of counts must be checked explicitly)
    nb_base = 0
    for _ in FastaBatches(file; batch_reads, revcomp = true)
        nb_base += 1
    end
    for swar in (false, true)
        nb = 0
        for (ntb, ntv) in zip(FastaBatches(file; batch_reads, revcomp = true),
                              FastaBatchesV2(file; batch_reads, revcomp = true, swar))
            @assert ntb.words == ntv.words "words mismatch (swar=$swar, batch $(nb + 1))"
            @assert ntb.rwords == ntv.rwords "rwords mismatch (swar=$swar, batch $(nb + 1))"
            @assert ntb.starts == ntv.starts && ntb.stops == ntv.stops "offsets mismatch (swar=$swar, batch $(nb + 1))"
            @assert ntb.heads == ntv.heads "heads mismatch (swar=$swar, batch $(nb + 1))"
            nb += 1
        end
        @assert nb == nb_base "batch count mismatch (baseline $nb_base, v2 $nb)"
    end
    return nb_base
end

function run_test()
    _pair_lut_check()
    @info "Pair LUT self-check OK (65536 entries vs _PACK_LUT)"

    small, big = ensure_data()

    # ---- small file: whole-file parity vs the reference paths ----------------
    @info "Reference path on the small file (load_fasta_mmap_fixed + pack_reads)"
    seqs_ref, heads_ref = load_fasta_mmap_fixed(small)
    words_ref, starts_ref, stops_ref = pack_reads(seqs_ref)
    seqs_rc = [UInt8(3) .- reverse(a) for a in seqs_ref]
    rwords_ref, rstarts_ref, rstops_ref = pack_reads(seqs_rc)
    @assert (rstarts_ref, rstops_ref) == (starts_ref, stops_ref)

    @info "Baseline load_fasta_packed (chunks=1) vs reference"
    base = load_fasta_packed(small; revcomp = true, chunks = 1, progress = false)
    @assert base == (words_ref, starts_ref, stops_ref, rwords_ref, heads_ref)

    @info "v2 vs baseline: whole file, chunks {1, auto, 4} x swar {false, true}"
    for swar in (false, true), chunks in (1, 0, 4)
        v = load_fasta_packed_v2(small; revcomp = true, swar, chunks)
        @assert v == base "v2 mismatch (swar=$swar, chunks=$chunks)"
    end
    @info "v2 single batch == baseline whole file: OK"

    # randomize_N / skipN (chunks=1: deterministic rand consumption order)
    for kw in ((randomize_N = true,), (skipN = true,))
        b = load_fasta_packed(small; revcomp = true, chunks = 1, progress = false, kw...)
        v = load_fasta_packed_v2(small; revcomp = true, chunks = 1, kw...)
        @assert v == b "v2 N-handling mismatch ($(first(kw)))"
    end
    @info "v2 randomize_N / skipN (chunks=1) == baseline: OK"

    # ---- small file: batched v2 vs batched baseline, lockstep ----------------
    nb = _lockstep(small, 3000) # 8192 reads -> 3 batches (3000+3000+2192)
    @info "Batched lockstep on the small file: OK ($nb batches)"

    # ---- big file: batched lockstep (2^17 reads -> 4 batches of 2^15) --------
    nb = _lockstep(big, 2^15)
    @info "Batched lockstep on the big file: OK ($nb batches, $(filesize(big)) bytes)"

    # ---- synthetic edge cases -------------------------------------------------
    dir = mktempdir(prefix = "newfasta_v2_")
    syn = joinpath(dir, "synthetic.fasta")
    write(syn,
        "junk before the first record\n",
        ">r1 desc\nACGTacgtNnGT\n",
        ">r2\r\nAC\r\nGTAA\r\n",
        ">r3\n",
        ">r4\nACGT\n",
        ">r5 last\nTTTTTTTT")
    for trim in (0, 5)
        seqs_s, heads_s = load_fasta_mmap_fixed(syn; trim = trim)
        wr, sr, er = pack_reads(seqs_s)
        wrc, _, _ = pack_reads([UInt8(3) .- reverse(a) for a in seqs_s])
        for swar in (false, true)
            w2, s2, e2, rw2, h2 = load_fasta_packed_v2(syn; revcomp = true, swar, trim)
            @assert (w2, s2, e2, rw2, h2) == (wr, sr, er, wrc, heads_s) "synthetic whole mismatch (trim=$trim, swar=$swar)"
        end
        nb = 0
        for swar in (false, true)
            first = 1; nb = 0
            for nt in FastaBatchesV2(syn; batch_reads = 1, revcomp = true, swar, trim)
                nn = length(nt.starts)
                sl = first:first + nn - 1
                lw, ls, le = pack_reads(seqs_s[sl])
                @assert nt.words == lw && nt.starts == ls && nt.stops == le "single-read batch mismatch (trim=$trim, swar=$swar)"
                lwrc, _, _ = pack_reads([UInt8(3) .- reverse(a) for a in seqs_s[sl]])
                @assert nt.rwords == lwrc "single-read rc batch mismatch (trim=$trim, swar=$swar)"
                @assert nt.heads == heads_s[sl]
                first += nn; nb += 1
            end
            @assert first - 1 == length(seqs_s)
        end
        @info "Synthetic edge cases OK (trim=$trim): whole-file + $nb single-read batches x swar {false,true}"
    end

    # empty file -> zero batches
    empty_fasta = joinpath(dir, "empty.fasta")
    touch(empty_fasta)
    n_b = sum(1 for _ in FastaBatchesV2(empty_fasta; batch_reads = 1); init = 0)
    @assert n_b == 0
    @info "Empty file OK: 0 batches"

    @info "ALL CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# benchmarks
# ==============================================================================
function run_bench(; reps = _reps())
    small, big = ensure_data()
    _warm_cache(small); _warm_cache(big)
    @info "Loader benchmarks" julia_threads = nthreads() reps
    @printf("  small file: %d bytes | big file: %d bytes\n", filesize(small), filesize(big))

    @info "-- small file, whole-file load (fwd + revcomp) --"
    _timed_min("E0  load_fasta_packed (baseline)"; bytes = filesize(small), reps) do
        w, s, e, rw, h = load_fasta_packed(small; revcomp = true, progress = false)
        length(w) + length(rw)
    end
    for swar in (false, true)
        _timed_min("E$(swar ? 2 : 1)  load_fasta_packed_v2(swar=$swar)"; bytes = filesize(small), reps) do
            w, s, e, rw, h = load_fasta_packed_v2(small; revcomp = true, swar)
            length(w) + length(rw)
        end
    end

    @info "-- big file, batched parse-only (batch_reads = 2^15, fwd + revcomp) --"
    _timed_min("E0  FastaBatches (baseline)"; bytes = filesize(big), reps) do
        s = 0
        for nt in FastaBatches(big; batch_reads = 2^15, revcomp = true)
            s += length(nt.words) + length(nt.rwords)
        end
        s
    end
    for swar in (false, true)
        _v2time_reset!()
        t = _timed_min("E$(swar ? 2 : 1)  FastaBatchesV2(swar=$swar)"; bytes = filesize(big), reps) do
            s = 0
            for nt in FastaBatchesV2(big; batch_reads = 2^15, revcomp = true, swar)
                s += length(nt.words) + length(nt.rwords)
            end
            s
        end
        swar && _v2time_print() # stage shares, accumulated over all reps
    end

    @info "-- big file variants --"
    _timed_min("E2  FastaBatchesV2(keep_heads=false)"; bytes = filesize(big), reps) do
        s = 0
        for nt in FastaBatchesV2(big; batch_reads = 2^15, revcomp = true, keep_heads = false)
            s += length(nt.words) + length(nt.rwords)
        end
        s
    end
    for br in (2^13, 2^14)
        _timed_min("E2  FastaBatchesV2(batch_reads=2^$(round(Int, log2(br))))"; bytes = filesize(big), reps) do
            s = 0
            for nt in FastaBatchesV2(big; batch_reads = br, revcomp = true)
                s += length(nt.words) + length(nt.rwords)
            end
            s
        end
    end
    println("  (E1 = swar=false, E2 = swar=true; GB/s = FASTA bytes consumed per second)")
    return nothing
end

# ==============================================================================
# flat profile of baseline vs v2 (evidence for where the time goes)
# ==============================================================================
function _profile_one(f, label::String)
    @info label
    f() # warm up (compile) outside the profile
    Profile.clear()
    Profile.@profile f()
    Profile.print(format = :flat, mincount = 25)
    return nothing
end

function run_profile(; variant = get(ARGS, 2, "both"))
    small, big = ensure_data()
    _warm_cache(big)

    (variant in ("e0", "both")) && _profile_one(() -> begin
        s = 0
        for nt in FastaBatches(big; batch_reads = 2^15, revcomp = true)
            s += length(nt.words)
        end
        _SINK[] += s
    end, "Profiling E0 baseline (FastaBatches, big file) at -t $(nthreads())")

    (variant in ("e2", "both")) && _profile_one(() -> begin
        s = 0
        for nt in FastaBatchesV2(big; batch_reads = 2^15, revcomp = true)
            s += length(nt.words)
        end
        _SINK[] += s
    end, "Profiling E2 v2 (FastaBatchesV2, big file) at -t $(nthreads())")
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data()
    mode == "test" && run_test()
    mode == "disk" && run_disk()
    mode == "micro" && run_micro()
    mode == "bench" && run_bench()
    mode == "profile" && run_profile()
    mode == "all" && (ensure_data(); run_test(); run_disk(); run_micro(); run_bench(); run_profile())
    mode in ("gen", "test", "disk", "micro", "bench", "profile", "all") ||
        error("unknown mode $mode (use gen|test|disk|micro|bench|profile|all)")
end
