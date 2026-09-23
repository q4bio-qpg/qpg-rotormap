# reader.jl -- the v3 multi-part parallel FASTA reader (from legacy
# fastareads_v3.jl, verbatim): the file is evenly split into nthreads()
# byte ranges, each thread parses ONLY its own range sequentially (SWAR
# pair-LUT collect), and fragments are pushed to a single shared channel as
# independently 2-bit-encoded FastaFragments with their headers attached.
#
#   _LUT3 / _PAIR3     byte/pair code LUTs (independent copies of the pack
#                      convention; line terminators gate the scalar path)
#   FastaFragment      (codes, head, rec) -- one read's 2-bit codes + header
#   _next_rec3         record-start scan (memchr-style)
#   _leading_run3      leading partial-record handling for a byte range
#   _part_task3 / _part_wrapper3 / fasta_reads   the reader itself
#
# REQUIRES: nothing (self-contained).  The legacy verification/bench mains
#   (_crosscheck_v2 vs FastaBatchesV2, run_test3/run_bench3, synthetic-file
#   builders) were dropped with the old experiment script -- cross-checks
#   live on in the experiments' test modes.
# =============================================================================
using Base.Threads

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

