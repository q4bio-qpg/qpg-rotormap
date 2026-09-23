# ==============================================================================
# fastareads_v3.jl -- specialized parallel FASTA reader: the file is evenly
# split into `parts` byte ranges (one per thread), every thread parses ONLY its
# own range sequentially, and the output is a single shared Channel of
# independently 2-bit-encoded fragments (header attached to each fragment).
#
# Specification (prompts/fastareads_v3.md + user clarifications):
#   1. the file is evenly split into n = nthreads() parts;
#   2. each thread is assigned its part and works only with it;
#   3. each thread scans its part sequentially (SWAR pair-LUT collect, like
#      newfasta_v2's E2 path);
#   4. if a part starts with dna letters (prior to the '>' symbol) they are
#      saved to ONE buffer and made available to the previous thread through a
#      dedicated channel (clarifications: every thread except the first reads
#      to the first "\n>" occasion -- newline followed by '>' -- and pushes
#      the buffer as a whole, never letter by letter; if a header line is cut
#      by the boundary, the whole record lands in the buffer and is processed
#      by the previous thread);
#   5. the thread output is a sequence of 2-bit encoded dna fragments built
#      from its fasta records, headers saved along with the encoding;
#   6. fragments from all threads are collected into one shared Channel (they
#      interleave arbitrarily); a fragment shorter than k is skipped, a longer
#      one is trimmed to exactly k (so every emitted fragment has length k).
#
# Encoding conventions (per clarifications):
#   - PLAIN FORWARD packing (base j of the fragment -> word (j-1)>>5 + 1,
#     bit offset 2*((j-1) & 31)); downstream compatibility is not a goal;
#   - every IUPAC letter except A/C/G/T (and any other junk byte) is
#     substituted with G when met WITHIN a sequence; '\n'/'\r' are line
#     structure and are skipped; bytes before the first '>' of a part/thread
#     that belong to no record are discarded (thread 1 junk prefix);
#   - a mid-line '>' is NOT a record start (it is junk -> G); only a '>'
#     preceded by '\n'/'\r' opens a record (newfasta_v2 convention).
#
# Boundary protocol (clarification 5).  chan[t] (t = 2..parts) is the channel
# dedicated to thread t; thread t pushes its leading-run buffer into chan[t]
# and thread t-1 consumes it:
#   - thread t found a line-start '>' in its part: push buffer (possibly
#     empty), CLOSE chan[t];
#   - thread t found no '>' at all (pathological): push its WHOLE part as one
#     buffer and keep chan[t] OPEN, then forward everything arriving on
#     chan[t+1] into chan[t] (so a chain of '>'-less parts flows back to the
#     waiting thread automatically) and close only when chan[t+1] closes;
#   - the last thread closes its channel after its buffer is pushed (EOF
#     terminates the record);
#   - channels are NEVER closed because of trimming: once a fragment has k
#     bases the consumer simply stops draining and abandons the rest.
#
# Memory/flow properties: fragments are exact-size allocations (no push!
# growth anywhere on the hot path -- the newfasta_v2 lesson), the shared
# channel is bounded (backpressure), so a streaming consumer holds only
# ~out_cap fragments in flight.  There are no global Int32 base offsets, so
# files above 2^31 bases stream WITHOUT batching (unlike newflow/v2).
#
# Run modes (ARGV[1]):
#   gen   generate the test files if missing (cached under NEWFASTA_V3_DIR)
#   test  correctness: reference serial parser on synthetic edge cases +
#         deterministic boundary cuts + generated files, all part counts,
#         plus a cross-check against FastaBatchesV2 (newfasta_v2.jl)
#   bench parse throughput at the current thread count (run under `julia -t N`)
#   all   everything above (default)
#
# RESULTS (kau, Ryzen 9 9950X, min over 3 reps, warm page cache, k = 20,000;
# big file = 2.6 GB streamed in ONE pass -- no batching, no Int32 offsets;
# v2 context = FastaBatchesV2 fwd-only, batched at 2^15 reads):
#
#   -t | v3 big GB/s | v2 big GB/s | v3 small GB/s
#    1 |     3.88     |    1.84     |    3.79
#    2 |     7.00     |    3.66     |    6.30
#    4 |     9.43     |    5.82     |    5.28
#    8 |     9.08     |    8.04     |    7.71
#   16 |    10.07     |    9.53     |   11.29
#
# v3 saturates ~9.5-10 GB/s from 4 threads up: the remaining cost is the
# per-fragment payload allocation (~640 MiB per big-file pass, gc 2-10%) and
# the shared-channel funnel -- the price of the streaming fragment API (a
# consumer that collects into its own storage pays none of it).
# ==============================================================================

include(joinpath(@__DIR__, "newfasta_v2.jl")) # FastaBatchesV2 (cross-check) +
                                              # _timed_min/_warm_cache helpers

using Printf
using Mmap
using RotorMap
using RotorMap.Utils
using Base.Threads

# ------------------------------------------------------------------------------
# Byte LUTs.  _LUT3: within a sequence EVERY byte encodes, ACGT/acgt -> own
# code, everything else (IUPAC, junk, digits, ...) -> G = 0x02; '\n'/'\r' are
# filtered before the LUT is consulted.  _PAIR3: 2 bytes -> UInt8, bits 0-3 =
# the two codes, bit 4 = "neither byte is a line terminator" (a block
# containing '\n'/'\r' falls back to the byte-exact scalar path).  64 KB, L2.
# ------------------------------------------------------------------------------
const _LUT3 = let
    lut = fill(UInt8(0x02), 256) # junk / IUPAC -> G
    lut[Int('A') + 1] = 0x00; lut[Int('C') + 1] = 0x01
    lut[Int('G') + 1] = 0x02; lut[Int('T') + 1] = 0x03
    lut[Int('a') + 1] = 0x00; lut[Int('c') + 1] = 0x01
    lut[Int('g') + 1] = 0x02; lut[Int('t') + 1] = 0x03
    lut
end

const _PAIR3 = let
    t = Vector{UInt8}(undef, 65536)
    NL = UInt8('\n'); CR = UInt8('\r')
    for x in 0:65535
        b0 = UInt8(x & 0xff); b1 = UInt8(x >> 8)
        if b0 == NL || b0 == CR || b1 == NL || b1 == CR
            t[x + 1] = 0x00
        else
            t[x + 1] = _LUT3[Int(b0) + 1] | (_LUT3[Int(b1) + 1] << 2) | 0x10
        end
    end
    t
end

# One parsed fragment: header line (verbatim, with the leading '>'), kept base
# count (== k for every emitted fragment) and the forward-packed 2-bit words.
struct FastaFragment
    header::String
    len::Int                # kept bases (== k)
    words::Vector{UInt32}   # cld(len, 16) words, base j -> word (j-1)>>4 + 1,
end                          # bit offset 2*((j-1) & 15), zero-padded tail

_grow3!(b::Vector{UInt16}) = resize!(b, max(2 * length(b), 4096))

const _V3DBG = Ref(get(ENV, "V3DBG", "0") == "1")

# ------------------------------------------------------------------------------
# Position of the first line-start '>' in [lo, hi] (hi inclusive), or 0.
# A '>' is a record start iff it is at a line start (preceded by '\n'/'\r' or
# at position 1).
# ------------------------------------------------------------------------------
function _next_rec3(raw, from::Int, hi::Int)
    n = length(raw)
    p = from
    p > hi && return hi + 1
    base = pointer(raw)
    NL = UInt8('\n'); CR = UInt8('\r')
    while p <= hi
        q = ccall(:memchr, Ptr{UInt8}, (Ptr{UInt8}, Cint, Csize_t),
                  base + (p - 1), Int('>'), Csize_t(hi - p + 1))
        q == C_NULL && break
        t = Int(q - base) + 1
        (t == 1 || raw[t - 1] == NL || raw[t - 1] == CR) && return t
        p = t + 1
    end
    return hi + 1
end

# ------------------------------------------------------------------------------
# Thread t's part may begin in the middle of the previous thread's last line.
# Everything from the part start up to (excluding) the first line-start '>'
# belongs to the record opened by an earlier thread -- that is the leading run
# which is pushed as ONE buffer into chan[t] for thread t-1.  Returns
# (buffer::Vector{UInt8}, gt) where gt is the '>' position of thread t's first
# OWN record, or 0 if the part contains no line-start '>' at all (then the
# whole part is the buffer).
# ------------------------------------------------------------------------------
function _leading_run3(raw, lo::Int, hi::Int)
    lo > hi && return (Vector{UInt8}(undef, 0), 0)
    NL = UInt8('\n'); CR = UInt8('\r')
    if raw[lo] == UInt8('>') && (lo == 1 || raw[lo - 1] == NL || raw[lo - 1] == CR)
        return (Vector{UInt8}(undef, 0), lo) # the part begins exactly at a record
    end                                      # start: nothing for the previous thread
    base = pointer(raw)
    p = lo + 1
    while p <= hi
        q = ccall(:memchr, Ptr{UInt8}, (Ptr{UInt8}, Cint, Csize_t),
                  base + (p - 1), Int('>'), Csize_t(hi - p + 1))
        q == C_NULL && break
        t = Int(q - base) + 1
        if raw[t - 1] == NL || raw[t - 1] == CR
            return (raw[lo:t - 1], t) # one buffer, includes the trailing '\n'
        end
        p = t + 1
    end
    # no record start anywhere in the part: the WHOLE part continues an earlier
    # record (clarification 5) -- the entire part goes into the buffer, and the
    # chain continues into the next part through the forwarding path
    return (raw[lo:hi], 0)
end

# ------------------------------------------------------------------------------
# SWAR collect: append codes of bytes [p, p+len) to the group scratch,
# skipping '\n'/'\r' and mapping every other byte through _LUT3 (junk -> G).
# Stops once `bases` reaches k (the fragment will be trimmed to k anyway).
# Groups are 16-bit, 8 codes each, in FORWARD order; `pending` carries the
# < 8 codes tail.  (8 bytes per iteration through _PAIR3, scalar fallback for
# blocks containing line terminators -- the newfasta_v2 E2 inner loop, minus
# the reversal and the N flags.)
# ------------------------------------------------------------------------------
@inline function _collect3!(sc::Vector{UInt16}, ng::Int, pending::UInt64, pnbits::Int,
                            bases::Int, p::Ptr{UInt8}, len::Int, k::Int)
    scap = length(sc)
    lut = _LUT3; lut16 = _PAIR3
    NL = UInt8('\n'); CR = UInt8('\r')
    i = 0
    @inbounds while i < len
        bases >= k && break
        if i + 8 <= len
            x = unsafe_load(Ptr{UInt64}(p + i)) # 8 bytes at once
            q0 = lut16[Int(x & 0x000000000000ffff) + 1]
            q1 = lut16[Int((x >> 16) & 0x000000000000ffff) + 1]
            q2 = lut16[Int((x >> 32) & 0x000000000000ffff) + 1]
            q3 = lut16[Int(x >> 48) + 1]
            if (q0 & q1 & q2 & q3) & 0x10 != 0  # no line terminators in block
                g = UInt32(q0 & 0x0f) | UInt32(q1 & 0x0f) << 4 |
                    UInt32(q2 & 0x0f) << 8 | UInt32(q3 & 0x0f) << 12
                (ng < scap) || (sc = _grow3!(sc); scap = length(sc))
                if pnbits == 0
                    sc[ng + 1] = g % UInt16
                else
                    kk = 16 - pnbits # prepend pending, keep the overflow
                    sc[ng + 1] = (pending | ((g & ((UInt32(1) << kk) - 1)) << pnbits)) % UInt16
                    pending = UInt64(g) >> kk # NB: keep pending UInt64 (g is UInt32)
                end
                ng += 1; bases += 8; i += 8
                continue
            end
        end
        c = unsafe_load(p + i); i += 1
        (c == NL || c == CR) && continue
        pending |= UInt64(lut[Int(c) + 1]) << pnbits
        pnbits += 2
        if pnbits == 16
            (ng < scap) || (sc = _grow3!(sc); scap = length(sc))
            sc[ng + 1] = pending % UInt16
            ng += 1; pending = UInt64(0); pnbits = 0
        end
        bases += 1
    end
    return (sc, ng, pending, pnbits, bases)
end

# Emit one fragment of exactly k bases from the scratch (caller guarantees
# that at least k codes were collected).  Forward packing: two consecutive
# 16-bit groups make one UInt32 word.
@inline function _emit3!(ch::Channel{FastaFragment}, header::String,
                         sc::Vector{UInt16}, ng::Int, pending::UInt64, pnbits::Int, k::Int)
    npart = pnbits >> 1
    if npart > 0 # fold the sub-group tail into the final group
        (ng < length(sc)) || resize!(sc, ng + 1)
        sc[ng + 1] = pending % UInt16
        ng += 1
    end
    words = Vector{UInt32}(undef, cld(k, 16)) # 16 bases (32 bits) per word
    npair = k >> 4 # complete words, each = two consecutive 8-code groups
    @inbounds for j in 1:npair
        words[j] = UInt32(sc[2 * j - 1]) | UInt32(sc[2 * j]) << 16
    end
    r = k & 15
    if r > 0
        gi = 2 * npair + 1
        rf = r >> 3 # complete 8-code groups in the partial word (0 or 1)
        rt = r & 7  # remaining codes (low bits of the next group)
        w = UInt32(0)
        @inbounds for m in 0:rf-1
            w |= UInt32(sc[gi + m]) << (16 * m)
        end
        if rt > 0
            # NB: Julia's << binds TIGHTER than *, so "1 << 2 * rt" would be
            # (1 << 2) * rt -- the explicit parens below are load-bearing
            w |= (UInt32(sc[gi + rf]) & ((UInt32(1) << (2 * rt)) - 1)) << (16 * rf)
        end
        words[npair + 1] = w
    end
    put!(ch, FastaFragment(header, k, words))
    return nothing
end

# ------------------------------------------------------------------------------
# chan[t] (t = 2..parts) carries thread t's leading buffer to thread t-1; the
# channels are UNBOUNDED (Inf) so a put! never blocks: a thread may stop
# draining early (fragment already has k bases) and abandon its channel.
# ------------------------------------------------------------------------------
function _part_task3(raw::Vector{UInt8}, lo::Int, hi::Int, k::Int, t::Int, parts::Int,
                     chans::Vector{Channel{Vector{UInt8}}}, ch::Channel{FastaFragment})
    NL = UInt8('\n'); CR = UInt8('\r')
    chan_out = t >= 2 ? chans[t - 1] : nothing # carries my leading buffer to t-1
    chan_in = t < parts ? chans[t] : nothing   # carries t+1's buffer to me

    own_lo = lo
    if chan_out !== nothing
        buf, gt = _leading_run3(raw, lo, hi)
        _V3DBG[] && (println("T$t: leading gt=$gt blen=$(length(buf))"); flush(stdout))
        put!(chan_out, buf) # one buffer, never letter by letter
        if gt != 0
            close(chan_out) # a record start exists in my part: done with chan[t]
            own_lo = gt
        else
            # no '>' anywhere in my part: the whole part continues an earlier
            # record; forward the upstream chain back to thread t-1
            if chan_in !== nothing
                for b in chan_in
                    put!(chan_out, b)
                end
            end
            close(chan_out)
            return
        end
    end

    sc = Vector{UInt16}(undef, 4096) # task-local scratch (grows to k/16 + 1)
    ng = 0; pending = UInt64(0); pnbits = 0; bases = 0

    # open-record bookkeeping (os == 0 <=> no record open at the part end)
    os = 0; ohend = 0; mid_header = false

    s = own_lo
    if chan_out === nothing
        # thread 1: the part may start with junk before the first record --
        # find the first line-start '>' instead of assuming a record at lo
        s = _next_rec3(raw, lo, hi)
    end
    while s <= hi
        e = _next_rec3(raw, s + 1, hi) # exclusive region end (hi+1 => open)
        _V3DBG[] && (println("T$t: rec s=$s e=$e"); flush(stdout))
        # --- header line (verbatim, '>' included, tolerate CRLF) ------------
        h = s
        while h < e && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        if h == e
            # the part ends INSIDE the header line: the record's header tail
            # and its whole sequence will arrive through chan[t] -> thread t-1
            os = s; ohend = hend; mid_header = true
            break
        end
        # --- sequence bytes [h+1, e) ----------------------------------------
        (sc, ng, pending, pnbits, bases) =
            _collect3!(sc, ng, pending, pnbits, 0, pointer(raw) + h, e - h - 1, k)
        if e <= hi # record closed by the '>' at e: emit or skip right away
            bases >= k && _emit3!(ch, String(raw[s:hend]), sc, ng, pending, pnbits, k)
            ng = 0; pending = UInt64(0); pnbits = 0; bases = 0
            s = e
        else       # open at the part boundary: completed from chan[t] below
            os = s; ohend = hend; mid_header = false
            break
        end
    end

    # ---- complete the open record from the upstream channel -----------------
    if os != 0
        _V3DBG[] && (println("T$t: drain os=$os bases=$bases"); flush(stdout))
        hbuf = Vector{UInt8}(undef, 0) # header tail (mid_header case only)
        if chan_in !== nothing && bases < k
            mode_hdr = mid_header
            for b in chan_in # drains until chan[t] is closed by the '>' finder
                lb = length(b)
                if mode_hdr
                    j = 1
                    while j <= lb
                        c = b[j]; j += 1
                        c == NL && (mode_hdr = false; break)
                        push!(hbuf, c)
                    end
                    if !mode_hdr && j <= lb
                        (sc, ng, pending, pnbits, bases) =
                            _collect3!(sc, ng, pending, pnbits, bases,
                                       pointer(b) + (j - 1), lb - j + 1, k)
                    end
                else
                    (sc, ng, pending, pnbits, bases) =
                        _collect3!(sc, ng, pending, pnbits, bases, pointer(b), lb, k)
                end
                bases >= k && break # trimmed to k: abandon the rest of chan[t]
            end
        end
        if bases >= k
            hdr = String(raw[os:ohend])
            if !isempty(hbuf)
                hbuf[end] == CR && pop!(hbuf)
                hdr *= String(hbuf) # header line was cut: tail from the buffer
            end
            _V3DBG[] && (println("T$t: emit2 hdr=$hdr bases=$bases"); flush(stdout))
            _emit3!(ch, hdr, sc, ng, pending, pnbits, k)
        end
    end
    _V3DBG[] && (println("T$t: return"); flush(stdout))
    return nothing
end

# ------------------------------------------------------------------------------
# fasta_reads(file; k, parts, out_cap, err_out) -> Channel{FastaFragment}
#
# Splits the file evenly into `parts` byte ranges, parses them in parallel and
# returns a bounded channel that yields one FastaFragment per kept record.
# Consume fully (iterate to the end) or `close(ch)` it early.  If a worker
# errors, the stream closes early and the exception is stored in `err_out[]`.
# ------------------------------------------------------------------------------
function fasta_reads(file::String; k::Int = 20_000, parts::Int = Threads.nthreads(),
                     out_cap::Int = 4 * parts, err_out::Ref{Any} = Ref{Any}(nothing))
    parts >= 1 || throw(ArgumentError("parts must be >= 1"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    raw = filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end
    ch = Channel{FastaFragment}(out_cap)
    Threads.@spawn begin
        try
            chans = [Channel{Vector{UInt8}}(Inf) for _ in 2:parts] # chans[t-1] = chan[t]
            # NB: Channel() default (sz_max = 0) is a RENDEZVOUS channel on
            # Julia 1.12 -- put! blocks until a take! matches, which deadlocks
            # whenever the predecessor stops draining early (trim-to-k).  Inf
            # is the truly unbounded channel the buffer protocol needs.
            n = length(raw)
            q, r = divrem(n, parts)
            @sync for t in 1:parts
                lo = (t - 1) * q + min(t - 1, r) + 1
                hi = lo + q - 1 + (t - 1 < r ? 1 : 0)
                Threads.@spawn _part_wrapper3(raw, lo, hi, k, t, parts, chans, ch)
            end
        catch err
            err_out[] = err
        finally
            close(ch)
        end
    end
    return ch
end

# run one part; if it dies with an exception, print it immediately (threaded
# errors are otherwise easy to miss), close the outgoing channel so a blocked
# predecessor cannot hang, then let @sync rethrow into the driver.
# EXCEPTION: an InvalidStateException is always teardown, never a data error --
# in this protocol it can only arise from put!/take! on a closed channel of the
# fragment/buffer network, i.e. the consumer closed the stream early (the
# ropeflow_v3 use case).  Such a shutdown must stay QUIET (no print, no rethrow
# into @sync => err_out stays clean) while still closing the outgoing channel so
# the blocked predecessor unwinds.
function _part_wrapper3(raw::Vector{UInt8}, lo::Int, hi::Int, k::Int, t::Int, parts::Int,
                        chans::Vector{Channel{Vector{UInt8}}}, ch::Channel{FastaFragment})
    try
        _part_task3(raw, lo, hi, k, t, parts, chans, ch)
    catch err
        if err isa Base.InvalidStateException
            t >= 2 && (try close(chans[t - 1]) catch end)
            return nothing
        end
        println(stderr, "fastareads_v3: thread $t failed: ",
                sprint(showerror, err, catch_backtrace()))
        flush(stderr)
        t >= 2 && (try close(chans[t - 1]) catch end) # unblock thread t-1
        throw(err)
    end
end

# ==============================================================================
# Data generation (cached under NEWFASTA_V3_DIR, default /tmp; same shape as
# the newfasta_v2 files: mutated ~20,000 bp reads of a 4 Mbp reference)
# ==============================================================================
const _V3_SMALL_READS = 2^13  #  8192 reads -> 164 MB
const _V3_BIG_READS = 2^17    # 131072 reads -> ~2.6 GB (streams in ONE pass)
const _V3_READ_LEN = 20_000

_v3_data_dir() = get(ENV, "NEWFASTA_V3_DIR", "/tmp")

function _ensure_file3(n_reads::Int)
    path = joinpath(_v3_data_dir(), "fastareads_v3_$(n_reads).fasta")
    isfile(path) && filesize(path) > n_reads * 19_000 && return path
    @info "Generating $(n_reads) mutated reads of ~$(_V3_READ_LEN) bp -> $path"
    mkpath(_v3_data_dir())
    ref = generate_reference(2^22; seed = 1234)
    reads, pos = generate_reads(ref, _V3_READ_LEN, n_reads; err = 0.02, seed = 42)
    heads = [">read_$i pos=$(pos[i])" for i in eachindex(reads)]
    t = @elapsed save_fasta(reads, path, heads = heads)
    @info "Wrote $path ($(filesize(path)) bytes) in $(round(t, digits = 1)) s"
    return path
end

ensure_data3() = (_ensure_file3(_V3_SMALL_READS), _ensure_file3(_V3_BIG_READS))

_v3_reps() = parse(Int, get(ENV, "NEWFASTA_V3_REPS", "3"))

# ==============================================================================
# Correctness.  The reference is a deliberately naive serial parser of the
# whole file (same semantics, forward packing written the obvious way) -- an
# independent implementation, so shared bugs cannot hide.
# ==============================================================================
function _ref_forward_pack(codes::Vector{UInt8}, L::Int)
    words = fill(UInt32(0), cld(L, 16)) # 16 bases (32 bits) per word
    for pos in 1:L
        words[(pos - 1) >> 4 + 1] |= UInt32(codes[pos]) << (2 * ((pos - 1) & 15))
    end
    return words
end

function _reference_fragments(raw::Vector{UInt8}, k::Int)
    frags = Dict{String,Vector{UInt32}}() # header -> words (len == k when kept)
    order = String[]
    NL = UInt8('\n'); CR = UInt8('\r'); GT = UInt8('>')
    n = length(raw)
    i = 1
    while i <= n # junk before the first record is discarded
        raw[i] == GT && (i == 1 || raw[i - 1] == NL || raw[i - 1] == CR) && break
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
        header = String(raw[s:hend])
        e = n + 1
        j = min(h + 1, n + 1)
        while j <= n
            if raw[j] == GT && (raw[j - 1] == NL || raw[j - 1] == CR)
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
        if length(codes) >= k
            frags[header] = _ref_forward_pack(codes, k)
            push!(order, header)
        end
    end
    return frags, order
end

# Run fasta_reads at several part counts and compare against the reference
# (fragment sets are compared by header: the shared channel interleaves parts).
function _check_file(path::String; k::Int, parts_list)
    raw = read(path)
    ref, _ = _reference_fragments(raw, k)
    for parts in parts_list
        err = Ref{Any}(nothing)
        got = Dict{String,Vector{UInt32}}()
        nfrag = 0
        for f in fasta_reads(path; k, parts, err_out = err)
            nfrag += 1
            @assert f.len == k && length(f.words) == cld(k, 16)
            @assert !haskey(got, f.header) "duplicate header $(f.header) (parts=$parts)"
            got[f.header] = f.words
        end
        @assert err[] === nothing "worker error at parts=$parts: $(err[])"
        @assert nfrag == length(ref) "fragment count mismatch (parts=$parts): got $nfrag, want $(length(ref))"
        for (h, w) in ref
            @assert haskey(got, h) "missing fragment $h (parts=$parts)"
            @assert got[h] == w "words mismatch for $h (parts=$parts)"
        end
    end
    return length(ref)
end

function _synthetic_tests(dir::String)
    files = [
        ("plain", ">r1 desc\nACGTACGTNN\n>r2\nacgtnryswkmbdhv\n>r3\nAC\n>r4\n" * "ACGT"^10 *
                  "\n>r5\n" * "TTAG"^3 * "GC\n"),
        ("junkfirst", "junk line before\nanother  junk\n>r1\n" * "ACGT"^5 * "\n>r2\nACGTACGTAC\n"),
        ("crlf", ">r1\r\n" * "ACGT"^4 * "\r\n>r2\r\n" * "TTTT"^3 * "\r\n"),
        ("notrailnl", ">r1\nACGTACGTAC\n>r2\n" * "ACGT"^4),
        ("emptyrec", ">r1\n>r2\nACGTACGTAC\n>r3\n\n>r4\nACGT\n"),
        ("junkseq", ">r1\nAC*G T12>weird^^acg\n>r2\n" * "ACGT"^3 * "\n"),
        ("iupac", ">r1\nRYSWKMBDHVNNNacgtn\n>r2\n" * "A"^15 * "\n"),
    ]
    for (name, content) in files
        path = joinpath(dir, "syn_$name.fasta")
        write(path, content)
        for k in (5, 10, 17)
            _check_file(path; k, parts_list = (1, 2, 3, 4, 7, 16))
        end
    end
    @info "Synthetic edge cases OK (7 files x k in {5,10,17} x parts in {1,2,3,4,7,16})"

    # --- deterministic cuts ---------------------------------------------------
    # cut exactly through a header line (byte 40 of 79 is inside r2's header)
    content = ">r1\n" * "ACGT"^6 * ">r2_header_cut_here_xx\n" * "ACGT"^7
    @assert length(content) == 79 && content[40] == 'c'
    path = joinpath(dir, "syn_cutheader.fasta")
    write(path, content)
    _check_file(path; k = 8, parts_list = (1, 2, 3, 4, 7))
    @info "Deterministic mid-header cut OK (parts=2/4 both cut at byte 40|41)"

    # cut exactly through a sequence line (byte 67 of 134, inside r2's sequence)
    content = ">r1\n" * "ACGT"^11 * "A" * ">r2\n" * "ACGT"^14 * "\n>r3\n" * "ACGT"^5
    @assert length(content) == 134 && 54 <= 67 <= 108 # cut lands inside r2's sequence
    path = joinpath(dir, "syn_cutseq.fasta")
    write(path, content)
    _check_file(path; k = 40, parts_list = (1, 2, 3, 4, 7))
    @info "Deterministic mid-sequence cut OK (parts=2/4 cut at byte 67|68)"

    # one record spanning every part: threads 2..end have no '>' at all ->
    # whole-part buffers chained back through the forwarders.  k=100 forces
    # thread 1 to actually CONSUME the chained buffers (its own part holds
    # fewer than 100 bases, so the drain cannot be skipped)
    path = joinpath(dir, "syn_onebig.fasta")
    write(path, ">only_one_record\n" * "ACGT"^100)
    for k in (17, 50, 100)
        _check_file(path; k, parts_list = (1, 2, 3, 4, 16))
    end
    @info "Pathological no-'>' chain OK (one record over up to 16 parts, k up to 100)"

    # empty file
    path = joinpath(dir, "empty.fasta")
    touch(path)
    for parts in (1, 4)
        err = Ref{Any}(nothing)
        n = 0
        for _ in fasta_reads(path; parts = parts, err_out = err)
            n += 1
        end
        @assert n == 0 && err[] === nothing
    end
    @info "Empty file OK (0 fragments)"
    return nothing
end

# Cross-checks on the generated 164 MB file:
#  1. v3 fragments == independent reference parser (parts = nthreads());
#  2. v3 fragments == FastaBatchesV2 (newfasta_v2.jl) reads decoded from its
#     reversed-order bitstream (reads with length >= k; v3 trims/skips).
function _crosscheck_v2(small::String; k::Int = 20_000)
    raw = read(small)
    ref, _ = _reference_fragments(raw, k)

    err = Ref{Any}(nothing)
    got = Dict{String,Vector{UInt32}}()
    for f in fasta_reads(small; k, parts = nthreads(), err_out = err)
        got[f.header] = f.words
    end
    @assert err[] === nothing
    @assert length(got) == length(ref)
    for (h, w) in ref
        @assert haskey(got, h) && got[h] == w "v3 vs reference mismatch for $h"
    end

    # v2 loader: stream bits are packed low-first, each read's bases REVERSED;
    # read r occupies bits [2*(starts-1), 2*stops).
    nkept = 0
    for nt in FastaBatchesV2(small; batch_reads = 2^15, revcomp = false, swar = true)
        for r in eachindex(nt.starts)
            len = Int(nt.stops[r] - nt.starts[r] + 1)
            len >= k || continue
            codes = Vector{UInt8}(undef, k)
            bit0 = 2 * (Int(nt.starts[r]) - 1)
            for pos in 1:k # base pos sits at stream bit bit0 + 2*(len - pos)
                b = bit0 + 2 * (len - pos)
                codes[pos] = UInt8((nt.words[(b >> 5) + 1] >> (b & 31)) & 3)
            end
            hdr = nt.heads[r]
            @assert got[hdr] == _ref_forward_pack(codes, k) "v2 cross-check mismatch for $hdr"
            nkept += 1
        end
    end
    @assert nkept == length(ref)
    @info "Generated-file check OK: $(length(ref)) fragments (k=$k) == reference == FastaBatchesV2 decode"
    return nothing
end

function run_test3()
    small, big = ensure_data3()
    dir = mktempdir(prefix = "fastareads_v3_")
    _synthetic_tests(dir)
    _check_file(small; k = 20_000, parts_list = (1, 2, 4, nthreads()))
    @info "Small generated file OK ($(filesize(small)) bytes, k=20000)"
    _crosscheck_v2(small)
    @info "ALL FASTAREADS_V3 CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
# Benchmarks (run under `julia -t N`; parts = nthreads())
# ==============================================================================
function run_bench3(; reps = _v3_reps())
    small, big = ensure_data3()
    _warm_cache(small); _warm_cache(big)
    @info "fasta_reads benchmarks" julia_threads = nthreads() reps
    @printf("  small file: %d bytes | big file: %d bytes\n", filesize(small), filesize(big))

    for (lbl, path) in (("small", small), ("big", big))
        _timed_min("v3  fasta_reads($lbl, k=20000)"; bytes = filesize(path), reps) do
            nf = 0; nw = 0
            for f in fasta_reads(path; k = 20_000, parts = nthreads())
                nf += 1
                nw += length(f.words)
            end
            nf + nw
        end
        @printf("  (%s: fragments counted over reps)\n", lbl)
    end

    @info "-- context: v2 batched loader (FastaBatchesV2, swar, fwd only) --"
    _timed_min("v2  FastaBatchesV2(big, batch=2^15, revcomp=false)"; bytes = filesize(big), reps) do
        s = 0
        for nt in FastaBatchesV2(big; batch_reads = 2^15, revcomp = false, swar = true)
            s += length(nt.words)
        end
        s
    end
    println("  (GB/s = FASTA bytes consumed per second; v3 emits forward-packed,")
    println("   k-trimmed fragments with headers through a shared Channel)")
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    mode == "gen" && ensure_data3()
    mode == "test" && run_test3()
    mode == "bench" && run_bench3()
    mode == "all" && (ensure_data3(); run_test3(); run_bench3())
    mode in ("gen", "test", "bench", "all") ||
        error("unknown mode $mode (use gen|test|bench|all)")
end
