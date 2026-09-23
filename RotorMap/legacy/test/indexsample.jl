# ==============================================================================
# indexsample.jl -- sample n reads of size k from a REFERENCE fasta (the human
# genome), add err mutations to each and save them as a plain fasta whose
# headers carry the sample's PROVENANCE: the reference record it came from
# (verbatim header) and the location of the sample inside that record.
#
# SPEC (prompts/indexsample.md):
#   * input: a reference fasta in the indexflowreal sense (whole-chromosome
#     records; the file RotorMap/test/indexflowreal.jl processes), default the
#     human GRCh38.p14 genome;
#   * sample n reads of length k from the valid k-windows (fully inside ONE
#     record, the indexflowreal window rule; records shorter than k contribute
#     nothing).  TWO draw modes:
#       :even   DEFAULT, DETERMINISTIC (seed-free selection): record j
#               provides n_j reads proportional to its window count
#               w_j = len_j - k + 1 (largest-remainder rounding, sum n_j = n
#               exactly); inside a record the n_j reads sit at the MIDPOINTS
#               of n_j equal strata of the window range,
#               start_t = 1 + div((2t-1)*W, 2*n_j) -- evenly spaced, distinct
#               when n_j <= W, spanning the record from near its start to
#               near its end; a record with n_j > W (only possible at
#               k-scale lengths) cycles its windows with replacement.  Reads
#               are emitted record by record in file order.
#       :random the original draw, kept as an option (INDEXSAMPLE_DRAW= /
#               draw=; its default name carries a _drandom tag): n iid
#               uniform draws over ALL valid windows -- record i with
#               probability (len_i - k + 1) / sum_j (len_j - k + 1), start
#               uniform in 1:(len_i - k + 1) -- WITH replacement, exactly
#               like Utils.generate_reads on a single-record reference
#               (which this generalizes);
#   * N-FREE POOL BY DEFAULT (nfree = true / INDEXSAMPLE_NFREE, default 1):
#     a window whose REFERENCE bytes contain an N -- any non-ACGT byte:
#     N/n, the other IUPAC codes, junk (everything _LUT_IS would silently
#     substitute to a fake G) -- is EXCLUDED from the sampling pool before
#     either draw: :even quotas are proportional to the N-free window counts
#     and the strata midpoints index the N-free start list, :random draws
#     uniformly over the N-free windows (one streaming scan per record: a
#     bad byte poisons exactly the starts c-k+1..c).  nfree = false restores
#     the raw valid-window pool and tags the default output name `_withn`
#     (verify parses the tag back like _drandom).  On an N-free reference
#     both settings coincide bit for bit;
#   * each read is mutated with `Utils.mutate` (err = FRACTION of the read
#     length: ceil(k*err) mutations total, ~1/3 insertions, ~1/3 deletions,
#     the rest substitutions; the equal ins/del counts keep every read exactly
#     k letters long);
#   * the reads are saved as ACGT letters via `Utils.save_fasta` with headers
#         >read_<i> start=<1-based> len=<k> src=<record header sans '>'
#     e.g. >read_7 start=5402311 len=20000 src=NC_000001.11 Homo sapiens
#     chromosome 1, GRCh38.p14 Primary Assembly
#     `start` counts every sequence byte except \n/\r (junk/IUPAC/N -> G keeps
#     its position -- the fastareads_v3/indexflowreal convention), so it is
#     directly comparable with the `starts` meta of the indexflowreal .bin;
#   * the output is a PURE FUNCTION of (fasta, n, k, err, seed, draw, nfree):
#     the :even selection is closed-form and the per-read child Xoshiro rngs
#     are drawn serially up front (the generate_reads pattern), so threading
#     never leaks into the result and `verify` can check a saved file by
#     bitwise regeneration.  The DEFAULT output name is UNTAGGED (:even,
#     N-free); a :random gen appends `_drandom` and an nfree = false gen
#     appends `_withn` (fixed order `_drandom_withn`); `verify` parses both
#     tags back (absent tags mean the defaults).
#
# This file is deliberately self-contained and CPU-only (the only RotorMap
# dependency is Utils.mutate/save_fasta): it mirrors the fastareads_v3 record
# walk (_next_rec3-style memchr line-start '>' bounds + the _LUT3 code map)
# instead of including the GPU flow, and run_test cross-checks the extraction
# against an independent naive (String/lines-based) parser -- shared bugs
# cannot hide.
#
# Run modes (ARGV[1]; ARGS[2] optionally overrides the fasta / output file):
#   gen     sample + save (cached: skipped when the output file exists;
#           INDEXSAMPLE_FORCE=1 or `gen!` regenerates)
#   gen!    force regeneration
#   verify  validate a saved reads fasta against the reference: every header
#           parses and names an existing record + valid start, every read is
#           exactly k letters, sampled rows match an independent re-extraction
#           (err = 0: bitwise; err > 0: differs + 16-mer containment + the
#           StringDistances LEVENSHTEIN edit rate of the pair, which must sit
#           just under err -- see the spot-check block), plus a bitwise
#           regeneration comparison when the parameters are known
#   test    correctness suite: mutate no-op/shape, synthetic edge-case
#           reference (junk prefix, mixed case, IUPAC/N, mid-line '>', CRLF,
#           exactly-k, too-short, empty record, N-gap + all-N records, no
#           trailing newline) vs the naive parser, the N-free pool (all-N
#           record untouched, N-gap windows excluded, nfree=false reaching
#           N-carrying windows), determinism + seed/nfree sensitivity,
#           gen/verify roundtrips (all name tags), and the human fasta
#           end-to-end (if present)
#   all     test + gen + verify (default)
#
# Configuration (env):
#   INDEXSAMPLE_FASTA  reference fasta (default the human GRCh38.p14 genome)
#   INDEXSAMPLE_OUT    output fasta (default
#                      <fasta>.indexsample_n<n>_k<k>_e<err>_s<seed>.fasta
#                      [*_drandom for the random draw] -- the name is the
#                      parameter record `verify` reads back)
#   INDEXSAMPLE_N      number of reads   (default 1000)
#   INDEXSAMPLE_K      read length       (default 20000)
#   INDEXSAMPLE_ERR    mutation fraction (default 0.05)
#   INDEXSAMPLE_SEED   rng seed          (default 42)
#   INDEXSAMPLE_DRAW   read draw: even | random (default even)
#   INDEXSAMPLE_NFREE  1 = sample only N-free windows (default); 0 = the raw
#                      valid-window pool (default-name tag `_withn`)
#   INDEXSAMPLE_ROWS   verify spot-check rows (default 24)
#   INDEXSAMPLE_LEV    max spot rows for the O(k^2) Levenshtein DP (default 256)
#   INDEXSAMPLE_FORCE  1 = regenerate even if the output exists
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16; GRCh38.p14 = 3.1 GB,
# 68 records, 3,095,453,524 seq chars; n = 1000, k = 20,000, err = 0.05,
# seed = 42 -> <fasta>.indexsample_n1000_k20000_e0.05_s42.fasta, 19.2 MiB):
#
#   reference table walk (all records + char counts)     0.29 s
#   gen: sample + mutate 1000 reads (26 records hit)     2.64 s   save: 0.1 s
#   verify: structural (all 1000 headers) + 24 spot rows
#     vs independent re-extraction + bitwise regeneration of all 1000 reads
#
#   EVEN DRAW (the new default; :random kept via INDEXSAMPLE_DRAW=random,
#   tagging its files `_drandom`): seed-free, closed-form selection -- record
#   j gets n_j ∝ len_j-k+1 (largest remainder) at the equal-strata midpoints
#   of its window range.  n = 131,072 gen: ~6.6 s, 66 of 68 records sampled
#   (vs 61-62 under :random; records whose quota rounds to zero still get
#   none).  The e0.05/e0.1/e0.15 s42 batches were REWRITTEN in place (verify:
#   structural + 24 spot rows + bitwise regeneration with draw=even, ALL
#   PASS) and score the same as the old random batches on the kstep-2000
#   index: 95.04/95.01/59.64% top-20 vs 94.98/94.93/59.47% (e2ehuman.jl).
#
#   N-FREE DEFAULT (nfree = true): the sampling pool counts only N-free
#   k-windows (no byte the code map would fake into a G).  For k = 20,000
#   the whole-genome scan (68 records, 3.1 Gb, -t 16) takes 0.33 s and cuts
#   the pool from 3,094,097,022 raw windows to 2,931,795,607 N-free ones
#   (94.75%; 5.25% of all windows touch an N-run/IUPAC/junk letter).
#   The n = 1000 s42 default batch was REGENERATED in place (gen! 3.85 s,
#   25 records sampled; verify: structural with the per-read N-freeness
#   check + 24 spot rows + bitwise regeneration, ALL PASS).  Pre-nfree
#   batches (the untagged n131072/2^20 files) predate the `_withn` tag:
#   their windows may contain N-runs decoded as G; e2ehuman.jl keeps
#   consuming them as-is.
#
# run_test (all on kau): A) Utils.mutate no-op (err=0) + shape (ins=del keeps
# the length); B) synthetic edge-case reference (junk prefix, mixed case,
# IUPAC/N, mid-line '>', exactly-k, k+1, too-short, empty record, N-gap +
# all-N records, CRLF, no trailing newline): err=0 sampling bitwise equal to
# an independent naive lines-based parser (64 reads, locations included),
# only N-FREE windows ever sampled (the all-N record untouched, the chrE
# N-gap excluded, exact pool-strata midpoints vs the naive pool lists),
# while nfree=false reaches N-carrying windows; C) determinism + seed/nfree
# sensitivity; D) err=0.1: lengths kept, every read differs from its window,
# 16-mer containment; E) gen/verify roundtrips (err=0 and err>0, all four
# name-tag combinations) + gen caching; F) the human fasta end-to-end
# (gen -> verify, err=0 exactness at chromosome scale, N-free pool at full
# k).  ALL PASS.
# ==============================================================================

using Random
using Mmap
using ProgressMeter
using StringDistances # Levenshtein() for the verify spot rows' edit-rate witness
using RotorMap
using RotorMap.Utils # mutate, save_fasta
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides; ARGS[2] can override the fasta / output file)
# ------------------------------------------------------------------------------
const IS_FASTA = get(ENV, "INDEXSAMPLE_FASTA",
                     "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
const IS_N = parse(Int, get(ENV, "INDEXSAMPLE_N", "1000"))          # number of reads
const IS_K = parse(Int, get(ENV, "INDEXSAMPLE_K", "20000"))         # read length
const IS_ERR = parse(Float64, get(ENV, "INDEXSAMPLE_ERR", "0.05"))  # mutation fraction
const IS_SEED = parse(Int, get(ENV, "INDEXSAMPLE_SEED", "42"))
const IS_DRAW = Symbol(get(ENV, "INDEXSAMPLE_DRAW", "even")) # read draw: even|random
const IS_NFREE = parse(Bool, get(ENV, "INDEXSAMPLE_NFREE", "true")) # only N-free windows
const IS_ROWS = parse(Int, get(ENV, "INDEXSAMPLE_ROWS", "24"))      # verify spot rows
const IS_LEV = parse(Int, get(ENV, "INDEXSAMPLE_LEV", "256"))       # levenshtein pairs cap
const IS_FORCE = parse(Bool, get(ENV, "INDEXSAMPLE_FORCE", "false"))

# the output name doubles as the parameter record: verify parses n/k/err/
# seed/draw/nfree back out of it (env INDEXSAMPLE_OUT overrides for
# non-default names).  The defaults (even draw, N-free pool) keep the
# UNTAGGED name; a :random gen appends `_drandom`, an nfree = false gen
# appends `_withn` (fixed order: `_drandom_withn`), and absent tags parse
# back as the defaults.
_is_out_default(fasta, n, k, err, seed, draw, nfree) =
    string(splitext(fasta)[1], ".indexsample_n$(n)_k$(k)_e$(err)_s$(seed)",
           draw === :even ? "" : "_drandom", nfree ? "" : "_withn", ".fasta")
function is_out(fasta::String = IS_FASTA; n::Int = IS_N, k::Int = IS_K,
                err::Real = IS_ERR, seed::Int = IS_SEED, draw::Symbol = IS_DRAW,
                nfree::Bool = IS_NFREE)
    get(ENV, "INDEXSAMPLE_OUT", _is_out_default(fasta, n, k, err, seed, draw, nfree))
end

_mmap_fasta(file::String) =
    filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end

# ==============================================================================
# FASTA core: the fastareads_v3/indexflowreal conventions, CPU-only mirror
# ==============================================================================

# every byte is a base via this LUT except \n/\r (skipped, not positions);
# junk / IUPAC / N -> G keeping its position (fastareads_v3's _LUT3)
const _LUT_IS = let
    lut = fill(UInt8(0x02), 256)
    lut[Int('A') + 1] = 0x00; lut[Int('C') + 1] = 0x01
    lut[Int('G') + 1] = 0x02; lut[Int('T') + 1] = 0x03
    lut[Int('a') + 1] = 0x00; lut[Int('c') + 1] = 0x01
    lut[Int('g') + 1] = 0x02; lut[Int('t') + 1] = 0x03
    lut
end

# the N-free pool's bad bytes: everything that is NOT a plain ACGT/acgt base
# (N/n, the other IUPAC codes, junk like a mid-line '>') -- all of it is
# silently substituted to G by _LUT_IS, so a window touching any such letter
# is corrupted at the source.  \n/\r are skipped by the callers before the
# lookup, never classified.
const _LUT_ISBAD = let
    lut = fill(true, 256)
    for c in "ACGTacgt"
        lut[Int(c) + 1] = false
    end
    lut
end

# Position of the first line-start '>' in [from, hi], or hi+1 (fastareads_v3's
# _next_rec3: a '>' opens a record iff it is at position 1 or follows \n/\r).
function _next_rec(raw::AbstractVector{UInt8}, from::Int, hi::Int)
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

"""
    _ref_table(raw) -> (heads, lo, hi, nchar)

Walk the mmap'd fasta bytes record by record (junk before the first record is
discarded) and return, per record: the verbatim header line (with '>'), the
inclusive byte span of its sequence and its base count (every byte except
\\n/\\r -- the indexflowreal sequence-character space).
"""
function _ref_table(raw::Vector{UInt8})
    heads = String[]
    lo = Int[]
    hi = Int[]
    nchar = Int64[]
    NL = UInt8('\n'); CR = UInt8('\r')
    n = length(raw)
    s = _next_rec(raw, 1, n)
    while s <= n
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = _next_rec(raw, s + 1, n)
        cnt = 0
        @inbounds for p in (h+1):(e-1)
            b = raw[p]
            (b == NL || b == CR) || (cnt += 1)
        end
        push!(heads, String(raw[s:hend]))
        push!(lo, h + 1)
        push!(hi, e - 1)
        push!(nchar, cnt)
        s = e
    end
    return (heads, lo, hi, nchar)
end

# ------------------------------------------------------------------------------
# Char-position -> byte-offset location.  A sampled start is a SEQUENCE
# position, but the bytes are wrapped in lines: a full per-record index of
# every line start costs chr1 ~3.6e6 entries, so it is built lazily, ONCE per
# sampled record (records are huge; reads rarely touch all of them), and each
# read then locates its start by a binary search + a few bytes of scanning.
# ------------------------------------------------------------------------------
struct _RecIndex
    loffs::Vector{Int}     # byte offset of each sequence line's first byte
    lchars::Vector{Int64}  # sequence chars before each line (same order)
end

function _rec_index(raw::Vector{UInt8}, lo::Int, hi::Int)
    NL = UInt8('\n'); CR = UInt8('\r')
    loffs = Int[]
    lchars = Int64[]
    sizehint!(loffs, 1024); sizehint!(lchars, 1024)
    c = Int64(0)
    i = lo
    while i <= hi
        push!(loffs, i)
        push!(lchars, c)
        j = i
        @inbounds while j <= hi && raw[j] != NL
            j += 1
        end
        @inbounds for p in i:(j-1) # count the line's chars, skipping \n/\r
            b = raw[p]
            (b == NL || b == CR) || (c += 1)
        end
        i = j + 1
    end
    return _RecIndex(loffs, lchars)
end

# byte offset of the 1-based sequence-char position `start` (validity is the
# caller's business: 1 <= start <= nchar - k + 1)
function _char_pos_byte(idx::_RecIndex, raw::Vector{UInt8}, start::Int)
    t = searchsortedlast(idx.lchars, start - 1) # the line holding `start`
    t >= 1 || error("start $start precedes the record's sequence")
    need = start - idx.lchars[t]                # 1-based char within that line
    p = idx.loffs[t]
    NL = UInt8('\n'); CR = UInt8('\r')
    got = 0
    @inbounds while true
        b = raw[p]
        (b == NL || b == CR) || (got += 1)
        got == need && return p
        p += 1
    end
end

# collect the k codes of the window starting at `start` (into a preallocated
# vector) -- the same scan indexflowreal's verification uses (_window_codes_indep)
function _extract_read!(codes::Vector{UInt8}, raw::Vector{UInt8}, idx::_RecIndex,
                        start::Int, k::Int)
    p = _char_pos_byte(idx, raw, start)
    lut = _LUT_IS
    NL = UInt8('\n'); CR = UInt8('\r')
    got = 0
    @inbounds while got < k
        b = raw[p]
        p += 1
        (b == NL || b == CR) && continue
        got += 1
        codes[got] = lut[Int(b) + 1]
    end
    return codes
end

# ------------------------------------------------------------------------------
# The N-free pool: one ascending scan per record.  A bad byte at char c
# poisons exactly the window starts c-k+1..c, so a "last bad char" watermark
# decides each window as it closes: at char c the window [c-k+1, c] is N-free
# iff lastbad <= c - k.  With ranks === nothing the COUNT of N-free windows
# is returned; with ranks (ascending 1-based positions in the N-free start
# list, repetitions allowed for the with-replacement cycling) the
# corresponding char starts are returned instead.
# ------------------------------------------------------------------------------
function _scan_nfree(raw::Vector{UInt8}, lo::Int, hi::Int, k::Int,
                     ranks::Union{Nothing,Vector{Int}} = nothing)
    NL = UInt8('\n'); CR = UInt8('\r')
    lut = _LUT_ISBAD
    cnt = 0
    lastbad = 0                # char position of the most recent bad byte
    c = 0                      # chars consumed in [lo, hi] (sequence space)
    picked = ranks === nothing ? Int[] : Vector{Int}(undef, length(ranks))
    nxt = 1
    @inbounds for p in lo:hi
        b = raw[p]
        (b == NL || b == CR) && continue
        c += 1
        if lut[Int(b) + 1]
            lastbad = c
        elseif c >= k && lastbad <= c - k
            cnt += 1
            if ranks !== nothing
                # consume EVERY rank equal to this window ordinal (the
                # with-replacement cycling repeats ranks, so the sorted
                # rank list holds duplicates)
                start = c - k + 1
                while nxt <= length(ranks) && ranks[nxt] == cnt
                    picked[nxt] = start
                    nxt += 1
                end
            end
        end
    end
    if ranks !== nothing
        nxt > length(ranks) ||
            error("N-free window rank $(ranks[nxt]) beyond the record's $cnt N-free windows")
    end
    return ranks === nothing ? cnt : picked
end

# true iff the k-window starting at `start` (byte offset via the record's
# line index) holds no bad byte -- verify's per-read N-freeness check
function _window_is_nfree(raw::Vector{UInt8}, idx::_RecIndex, start::Int, k::Int)
    p = _char_pos_byte(idx, raw, start)
    NL = UInt8('\n'); CR = UInt8('\r')
    lut = _LUT_ISBAD
    got = 0
    @inbounds while got < k
        b = raw[p]
        p += 1
        (b == NL || b == CR) && continue
        got += 1
        lut[Int(b) + 1] && return false
    end
    return true
end

# convert in-record pool RANKS (starts[]) into actual 1-based char starts
# (the N-free pool is not the contiguous 1:W range): one _scan_nfree pick
# pass per distinct record, threaded over records -- each thread writes only
# its own record's entries, so the result is schedule-independent
function _ranks_to_starts!(raw::Vector{UInt8}, rlo::Vector{Int}, rhi::Vector{Int},
                           k::Int, recs::Vector{Int}, starts::Vector{Int})
    byrec = Dict{Int,Vector{Int}}() # record -> read indices
    for i in eachindex(recs)
        push!(get!(byrec, recs[i], Int[]), i)
    end
    js = collect(keys(byrec))
    @threads for j in js
        pick = sort!([(starts[i], i) for i in byrec[j]]) # (rank, read) ascending
        got = _scan_nfree(raw, rlo[j], rhi[j], k, [p[1] for p in pick])
        length(got) == length(pick) ||
            error("internal: record $j -- picked $(length(got)) of $(length(pick)) N-free starts")
        for t in eachindex(pick)
            starts[pick[t][2]] = got[t]
        end
    end
    return nothing
end

# ==============================================================================
# Provenance headers:  >read_<i> start=<1-based> len=<k> src=<record header>
# (`src` is the record's verbatim header minus the leading '>'; everything
# after " src=" is free text and never parsed -- the keys before it are fixed
# single tokens, so the format stays machine-readable despite the spaces)
# ==============================================================================

_read_head(i::Int, rec_head::AbstractString, start::Int, k::Int) =
    ">read_$(i) start=$(start) len=$(k) src=$(rec_head[2:end])"

function _parse_read_head(h::AbstractString)
    startswith(h, ">read_") || return nothing
    sp = findfirst(" src=", h)
    sp === nothing && return nothing
    fields = split(h[1:first(sp)-1])
    length(fields) >= 2 || return nothing
    kv = Dict{String,String}()
    for f in fields[2:end]
        eq = findfirst('=', f)
        eq === nothing && return nothing
        kv[String(f[1:eq-1])] = String(f[eq+1:end])
    end
    haskey(kv, "start") && haskey(kv, "len") || return nothing
    start = tryparse(Int, kv["start"])
    len = tryparse(Int, kv["len"])
    (start === nothing || len === nothing) && return nothing
    return (id = String(fields[1]), start = start, len = len,
            src = String(h[last(sp)+1:end]))
end

# ==============================================================================
# Sampling: two draw modes over the WINDOW POOL, then Utils.mutate.  The pool
# is a record's N-free k-windows (nfree = true, the default) or all its valid
# k-windows (nfree = false); the selection works on in-record pool RANKS,
# which _ranks_to_starts! converts to char starts when the pool is N-free.
#   :even  (default) record j contributes n_j ∝ its pool size w_j (largest-
#          remainder rounding, sum n_j = n) at the midpoints of n_j equal
#          strata of the pool -- deterministic and seed-free;
#   :random n iid uniform draws over the pooled windows, with replacement.
# ==============================================================================

# largest-remainder (Hamilton) allocation of n reads over the records in
# proportion to their window counts: n_j = floor(n*w_j/total), the leftover
# goes to the largest fractional remainders (ties -> lower record index via
# the explicit (frac, j) sort key).  sum(alloc) == n exactly, and every
# record lands within 1 read of its exact quota n*w_j/total.
function _even_alloc(w::Vector{Int64}, n::Int)
    total = sum(w)
    quota = n .* w ./ total
    alloc = floor.(Int, quota)
    fracs = quota .- alloc
    order = sort(eachindex(w), by = j -> (-fracs[j], j))
    for t in 1:(n - sum(alloc))
        alloc[order[t]] += 1
    end
    @assert sum(alloc) == n
    return alloc
end

"""
    sample_reads(fasta; n, k, err, seed, draw, nfree = true)
        -> (reads::Vector{Vector{UInt8}}, heads::Vector{String},
            recs::Vector{Int}, starts::Vector{Int})

Sample `n` reads of length `k` from the reference fasta over the window
pool -- by default (`nfree = true`) the record's N-FREE k-windows only: no
window whose reference bytes hold an N (any non-ACGT byte -- N/n, the other
IUPAC codes, junk -- that the code map would substitute to a fake G); a
record without a single clean window contributes nothing.  `nfree = false`
restores the raw pool of all valid k-windows (fully inside one record).
The pool's reads are decoded through _LUT_IS and mutated with
`Utils.mutate(read, err)` (err = mutation FRACTION of the read length; the
read stays exactly k long).  Two draw modes over the pool:

  `draw = :even` (default): record j provides n_j reads proportional to its
  pool size w_j (largest-remainder rounding, sum n_j = n); its reads sit at
  the midpoints of n_j equal strata of the pool's ascending start list,
  evenly spaced from the pool's first to last window (n_j > w_j cycles the
  pool with replacement).  Seed-free selection.

  `draw = :random`: the original sampler -- n iid uniform draws over ALL
  pooled windows, with replacement (like generate_reads).

`reads[i]` holds the MUTATED codes; `heads[i]` the provenance header naming
the reference record (`recs[i]`, 1-based) and the 1-based window start
(`starts[i]`, the record's sequence-character space).  Deterministic in
(fasta, n, k, err, seed, draw, nfree) regardless of the thread schedule;
:even's selection does not depend on the seed at all (the seed still drives
the mutation rngs).  On a reference without N/junk bytes both nfree
settings produce identical output.
"""
function sample_reads(fasta::String; n::Int = IS_N, k::Int = IS_K,
                      err::Real = IS_ERR, seed::Int = IS_SEED,
                      draw::Symbol = IS_DRAW, nfree::Bool = IS_NFREE)
    n >= 1 || throw(ArgumentError("n must be >= 1"))
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    err >= 0 || throw(ArgumentError("err must be >= 0"))
    draw in (:even, :random) ||
        throw(ArgumentError("draw must be :even or :random (got $draw)"))
    err = Float64(err)
    isfile(fasta) || error("fasta not found: $fasta")
    raw = _mmap_fasta(fasta)
    t_table = @elapsed (rheads, rlo, rhi, rlen) = _ref_table(raw)
    total_bases = sum(rlen)
    @info "reference: $(length(rheads)) records, $total_bases seq chars ($(round(t_table; digits = 2)) s)"

    # window counts per record = the sampling weights.  nfree (the default)
    # counts only N-FREE k-windows -- one streaming scan per record, threaded
    # over records (a window touching an N/junk byte would decode to fake
    # Gs).  nfree = false keeps the raw valid-window counts.  Either way a
    # record without a single pool window contributes nothing.
    R = length(rlen)
    if nfree
        w = Vector{Int64}(undef, R)
        @threads for j in 1:R
            w[j] = _scan_nfree(raw, rlo[j], rhi[j], k)
        end
    else
        w = [max(l - k + 1, Int64(0)) for l in rlen]
    end
    total = sum(w)
    total > 0 || error(nfree ?
        "no N-free k = $k window in $(length(rheads)) records " *
        "($(total_bases) seq chars; every window touches N/junk?)" :
        "no record has >= k = $k bases ($(total_bases) seq chars in $(length(rheads)) records)")
    @info "pool: $total $(nfree ? "N-free " : "")k-windows in " *
          "$(count(>(0), w)) of $R records"

    # every random choice up front, serially: the output never depends on the
    # thread schedule (generate_reads's pattern); :even's SELECTION is
    # closed-form and consumes no rng draws at all
    rng = Xoshiro(seed)
    cum = accumulate(+, w) # cum[j] >= r > cum[j-1] <=> window r lives in record j
    recs = Vector{Int}(undef, n)
    starts = Vector{Int}(undef, n)
    if draw === :even
        alloc = _even_alloc(w, n)
        g = 0
        for j in eachindex(w)
            nj = alloc[j]
            nj == 0 && continue
            W = w[j]
            @assert W > 0 "allocated reads to a record with an empty pool"
            for t in 1:nj
                g += 1
                recs[g] = j
                # pool rank: midpoint of the t-th of nj equal strata of the W
                # pooled windows; nj > W (k-scale records) cycles the pool
                starts[g] = nj <= W ? 1 + Int(div((2 * t - 1) * W, 2 * nj)) :
                                      ((t - 1) % W) + 1
            end
        end
        @assert g == n
    else
        for i in 1:n
            r = rand(rng, Int64(1):total)
            j = searchsortedfirst(cum, r)
            recs[i] = j
            # pool rank within record j (== the window start for the raw
            # 1:W pool; _ranks_to_starts! maps it into the N-free list)
            starts[i] = Int(r - (j > 1 ? cum[j-1] : Int64(0)))
        end
    end
    # the N-free pool is not contiguous: convert ranks to real char starts
    nfree && _ranks_to_starts!(raw, rlo, rhi, k, recs, starts)
    rngs = [Xoshiro(rand(rng, Int64)) for _ in 1:n] # one child rng per read

    # lazy per-record line indexes (a record can hold millions of lines; build
    # only for the sampled ones, once, outside the threaded loop)
    ridx = Dict{Int,_RecIndex}()
    for j in unique(recs)
        ridx[j] = _rec_index(raw, rlo[j], rhi[j])
    end

    reads = Vector{Vector{UInt8}}(undef, n)
    @showprogress @threads for i in 1:n
        codes = _extract_read!(Vector{UInt8}(undef, k), raw, ridx[recs[i]],
                               starts[i], k)
        reads[i] = Utils.mutate(codes, err; rng = rngs[i])
    end
    heads = [_read_head(i, rheads[recs[i]], starts[i], k) for i in 1:n]
    return (reads = reads, heads = heads, recs = recs, starts = starts)
end

# ------------------------------------------------------------------------------
# gen: sample + save (cached by output name unless forced)
# ------------------------------------------------------------------------------
# NB: `out` comes after n/k/err/seed so its default can build the parameter-
# record name from the CALLER's values (keyword defaults evaluate left -> right)
function run_gen(; fasta::String = IS_FASTA, n::Int = IS_N, k::Int = IS_K,
                 err::Real = IS_ERR, seed::Int = IS_SEED, draw::Symbol = IS_DRAW,
                 nfree::Bool = IS_NFREE,
                 out::String = is_out(fasta; n, k, err, seed, draw, nfree),
                 force::Bool = IS_FORCE)
    if !force && isfile(out)
        @info "indexsample gen: $out exists, skipping (mode gen! or INDEXSAMPLE_FORCE=1 regenerates)"
        return out
    end
    @info "indexsample gen" fasta n k err seed draw nfree out
    t = @elapsed (reads, heads, recs, _) = sample_reads(fasta; n, k, err, seed, draw, nfree)
    @info "sampled $(length(reads)) reads x $k from $(length(unique(recs))) record(s) in $(round(t; digits = 2)) s"
    t = @elapsed Utils.save_fasta(reads, out; heads)
    @info "saved $out ($(round(filesize(out) / 2^20; digits = 1)) MiB) in $(round(t; digits = 1)) s"
    return out
end

# ==============================================================================
# verify: validate a saved reads fasta against the reference
# ==============================================================================

# the output name doubles as the parameter record (absent tags mean the
# defaults :even + nfree; only non-default gens write the tags, in the fixed
# order `_drandom` then `_withn`)
function _params_from_name(path::String)
    m = match(r"_n(\d+)_k(\d+)_e([0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?)_s(\d+)" *
              r"(?:_d(random))?(_withn)?\.fasta$", path)
    m === nothing && return nothing
    (n = parse(Int, m[1]), k = parse(Int, m[2]), err = parse(Float64, m[3]),
     seed = parse(Int, m[4]),
     draw = m[5] === nothing ? :even : :random, nfree = m[6] === nothing)
end

# strict ACGT decode of a saved read's bytes (the output format contract:
# letters only; the record span's line breaks are skipped, anything else fails)
function _read_codes_strict(raw::Vector{UInt8}, lo::Int, hi::Int, n::Int)
    codes = Vector{UInt8}(undef, n)
    got = 0
    @inbounds for p in lo:hi
        c = raw[p]
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        code = c == UInt8('A') ? 0x00 : c == UInt8('C') ? 0x01 :
               c == UInt8('G') ? 0x02 : c == UInt8('T') ? 0x03 : 0xff
        code != 0xff || error("non-ACGT byte '$(Char(c))' in a saved read at $p")
        got += 1
        got <= n || error("saved read at $lo: more than $n letters")
        codes[got] = code
    end
    got == n || error("saved read at $lo: only $got of $n letters")
    return codes
end

_letters(codes::Vector{UInt8}) = String([Utils.dna_letters[c+1] for c in codes])

# number of shared 16-mers between two code vectors (UInt32 rolling hash);
# a robust "the read really is a mutated copy of THIS window" witness whenever
# the expected longest unmutated run (k / n_err) dwarfs 16
function _shared16(a::Vector{UInt8}, b::Vector{UInt8})
    s = Set{UInt32}()
    h = UInt32(0)
    @inbounds for j in 1:16
        h = (h << 2) | UInt32(a[j])
    end
    push!(s, h)
    @inbounds for j in 17:length(a)
        h = (h << 2) | UInt32(a[j])
        push!(s, h)
    end
    h = UInt32(0)
    @inbounds for j in 1:16
        h = (h << 2) | UInt32(b[j])
    end
    n = h in s ? 1 : 0
    @inbounds for j in 17:length(b)
        h = (h << 2) | UInt32(b[j])
        n += (h in s) ? 1 : 0
    end
    return n
end

"""
    run_verify(; out, fasta, k, err, seed, n, draw, nfree, rows)

Validate the saved reads fasta `out` against the reference `fasta`:

  * EVERY read: the header parses, `len` == k, the sequence is exactly k
    strict-ACGT letters, `src` names a real reference record and `start` is a
    valid window start in that record (1 .. len-k+1); with nfree = true the
    window must additionally hold no N/junk byte (the sampling pool's
    contract);
  * `rows` sampled reads: err == 0 -> bitwise equal to an independent
    re-extraction of the claimed window; err > 0 -> different from it, and
    (when the expected unmutated run k/n_err >= 32) sharing at least one
    16-mer with it;
  * when n/k/err/seed/draw/nfree are all known (arguments > the filename's
    parameter record > nothing): the whole file is compared against a bitwise
    regeneration (the sampler is deterministic by construction).

Parameters default to the output filename's
`_n.._k.._e.._s..[_drandom][_withn]` record; pass them explicitly for
non-default output names.
"""
function run_verify(; out::String = is_out(), fasta::String = IS_FASTA,
                    k::Union{Int,Nothing} = nothing, err::Union{Real,Nothing} = nothing,
                    seed::Union{Int,Nothing} = nothing, n::Union{Int,Nothing} = nothing,
                    draw::Union{Symbol,Nothing} = nothing,
                    nfree::Union{Bool,Nothing} = nothing, rows::Int = IS_ROWS)
    @info "indexsample verify" fasta out
    isfile(fasta) || error("reference fasta not found: $fasta")
    isfile(out) || error("reads fasta not found: $out")

    pn = _params_from_name(out)
    if pn !== nothing
        k = k === nothing ? pn.k : k
        err = err === nothing ? pn.err : err
        seed = seed === nothing ? pn.seed : seed
        n = n === nothing ? pn.n : n
        draw = draw === nothing ? pn.draw : draw
        nfree = nfree === nothing ? pn.nfree : nfree
    end

    # reference: records + a src -> record lookup (src = verbatim head sans '>')
    rraw = _mmap_fasta(fasta)
    (rheads, rlo, rhi, rlen) = _ref_table(rraw)
    bysrc = Dict(String(h[2:end]) => j for (j, h) in enumerate(rheads))

    # the reads file: same walker (each read = one record: header + one line)
    rraw2 = _mmap_fasta(out)
    (heads, lo, hi, nchar) = _ref_table(rraw2)
    N = length(heads)
    N > 0 || error("no reads in $out")
    k === nothing && (k = Int(nchar[1])) # derivable: all reads share one length
    k >= 1 || error("bad k = $k")

    # ---- structural checks on every read ------------------------------------
    ridx = Dict{Int,_RecIndex}()
    nwin_checked = 0
    for i in 1:N
        h = _parse_read_head(heads[i])
        h !== nothing || error("unreadable header at read $i: $(heads[i])")
        h.len == k || error("read $i: len=$(h.len) != k=$k")
        nchar[i] == k || error("read $i: $(nchar[i]) seq chars != k=$k")
        haskey(bysrc, h.src) ||
            error("read $i: src record not found in the reference: $(h.src)")
        j = bysrc[h.src]
        1 <= h.start <= rlen[j] - k + 1 ||
            error("read $i: start $(h.start) outside 1:$(rlen[j] - k + 1) of $(rheads[j])")
        if nfree === true # the N-free pool's contract, re-checked per read
            idx = get!(() -> _rec_index(rraw, rlo[j], rhi[j]), ridx, j)
            _window_is_nfree(rraw, idx, h.start, k) ||
                error("read $i: window at $(h.start) of $(rheads[j]) carries N/junk (nfree=true)")
        end
        nwin_checked += 1
    end
    @info "structural OK: $nwin_checked reads x $k, all provenance headers valid" *
          (nfree === true ? ", all windows N-free" : "")

    # ---- spot checks vs the reference window --------------------------------
    rs = MersenneTwister(1234)
    if err !== nothing
        n_err = ceil(Int, k * err)
        lev = Tuple{Int,String,String}[] # (read index, ref letters, read letters)
        for i in rand(rs, 1:N, min(rows, N))
            h = _parse_read_head(heads[i])
            j = bysrc[h.src]
            idx = get!(() -> _rec_index(rraw, rlo[j], rhi[j]), ridx, j)
            ref = _extract_read!(Vector{UInt8}(undef, k), rraw, idx, h.start, k)
            got = _read_codes_strict(rraw2, lo[i], hi[i], nchar[i])
            if err == 0
                got == ref ||
                    error("read $i ($(heads[i])): err=0 but != the reference window")
            else
                got != ref ||
                    error("read $i ($(heads[i])): err>0 but identical to the reference window")
                if k >= 32 * n_err
                    _shared16(ref, got) >= 1 ||
                        error("read $i ($(heads[i])): shares no 16-mer with its window")
                end
                push!(lev, (i, _letters(ref), _letters(got)))
            end
        end
        @info "spot rows OK ($(min(rows, N)) reads, err=$err vs independent re-extraction)"
        if !isempty(lev)
            # Levenshtein witness that err is the real mutation rate: the
            # generative script (subs + ins + dels) IS an edit script of
            # n_err = ceil(k*err) single-character edits, so d <= n_err ALWAYS,
            # while accidental coincidences (an inserted/deleted base matching
            # its context, a duplicate substitution position) only SAVE edits
            # -- the observed edit rate must sit JUST UNDER err: nowhere near
            # 0 (not mutated) and never above err (over-mutated).  The DP is
            # O(k^2) per pair (~4e10 cells for k = 20,000), so the pairs run
            # @threaded and are capped at IS_LEV with an explicit message.
            nlev = min(length(lev), IS_LEV)
            nlev < length(lev) &&
                @info "levenshtein on the first $nlev of $(length(lev)) spot rows (INDEXSAMPLE_LEV cap, O(k^2) DP)"
            rates = Vector{Float64}(undef, nlev)
            @threads for t in 1:nlev
                (i, a, b) = lev[t]
                d = Levenshtein()(a, b)
                rate = d / k
                0.5 * err <= rate <= err + 1.0 / k ||
                    error("read $i: levenshtein rate $(round(rate, digits = 4)) " *
                          "outside [$(0.5 * err), $err] (d=$d, k=$k, n_err=$n_err)")
                rates[t] = rate
            end
            @info "levenshtein OK: edit rate $(round(sum(rates) / nlev; digits = 4)) " *
                  "(min $(round(minimum(rates); digits = 4)), max $(round(maximum(rates); digits = 4))) " *
                  "vs err=$err over $nlev pair(s)"
        end
    else
        @warn "err unknown (non-default name): skipping the spot row checks"
    end

    # ---- bitwise regeneration when the full parameter set is known ----------
    if err !== nothing && seed !== nothing && n !== nothing &&
       draw !== nothing && nfree !== nothing
        n == N || error("regeneration: $n reads expected, file holds $N")
        (reads, gheads, _, _) = sample_reads(fasta; n, k, err, seed, draw, nfree)
        for i in 1:N
            gheads[i] == heads[i] ||
                error("regeneration: header $i differs:\n  $(heads[i])\n  $(gheads[i])")
            _read_codes_strict(rraw2, lo[i], hi[i], k) == reads[i] ||
                error("regeneration: read $i differs from the saved letters")
        end
        @info "regeneration OK: all $N reads bitwise identical (seed=$seed, draw=$draw" *
              (nfree ? ")" : ", withn)")
    else
        @warn "n/err/seed/draw/nfree unknown (non-default name): skipping the regeneration check"
    end

    @info "INDEXSAMPLE VERIFY PASSED ($N reads vs $(length(rheads)) reference records)"
    return nothing
end

# ==============================================================================
# Test data: a small deterministic edge-case reference (the indexflowreal
# builder's little sibling, sized around a test k of 50) -- including an
# N-gap record and an all-N record for the N-free pool tests
# ==============================================================================

function _ensure_test_fasta(path::String; k::Int = 50)
    isfile(path) && filesize(path) > 2 * k && return path
    rs = MersenneTwister(77)
    bases = "ACGT"
    iupac = collect("NRYSWKMBDHV")
    randseq(len) = String([bases[rand(rs, 1:4)] for _ in 1:len])
    function sprinkle(s) # lowercase + N + IUPAC junk at fixed strides
        v = collect(s)
        for i in 1:7:length(v)
            v[i] = lowercase(v[i])
        end
        for i in 5:13:length(v)
            v[i] = rand(rs, iupac)
        end
        return String(v)
    end
    wrapped(s, width) = join((s[i:min(i + width, end)] for i in 1:width:length(s)), "\n")
    seqA = sprinkle(randseq(3k + 27))
    seqA = seqA[1:2k] * ">" * seqA[2k+1:end] # a mid-line '>' (junk -> G)
    open(path, "w") do io
        write(io, "junk bytes before the first record\nACGTACGT\n")
        write(io, ">chrA mixed case iupac\n", wrapped(seqA, 60), "\n")
        write(io, ">chrB exactly_k\n", randseq(k), "\n")
        write(io, ">chrC k_plus_1\n", randseq(k + 1), "\n")
        write(io, ">mt too_short\n", randseq(k - 1), "\n")
        write(io, ">empty\n")
        write(io, ">chrD crlf_record\n",
              replace(wrapped(randseq(2k + 5), 30), "\n" => "\r\n"), "\n")
        write(io, ">chrE n_gap\n", wrapped("A"^100 * "N"^30 * "G"^60, 60), "\n")
        write(io, ">chrF all_n\n", "N"^(2k - 30)) # no trailing newline (last)
    end
    return path
end

# ==============================================================================
# The independent reference: a deliberately naive String/lines-based parser
# (different code path from the production memchr/LUT walk -- shared bugs
# cannot hide).  Returns (name, start, codes, raw letters) for every valid
# k-window; the raw letters (BEFORE the junk/IUPAC/N -> G mapping) are what
# the N-free pool tests filter on.
# ==============================================================================

_naive_code(c::Char) = c == 'A' || c == 'a' ? 0x00 :
                       c == 'C' || c == 'c' ? 0x01 :
                       c == 'G' || c == 'g' ? 0x02 :
                       c == 'T' || c == 't' ? 0x03 : 0x02 # junk / IUPAC / N -> G

function _naive_windows(path::String, k::Int)
    recs = Tuple{String,Vector{UInt8},Vector{Char}}[]
    name = nothing
    codes = UInt8[]
    chars = Char[]
    for line in eachline(path)
        line = chomp(line) # strips the trailing \r\n / \n / \r
        if startswith(line, ">")
            name === nothing || push!(recs, (name, codes, chars))
            name = String(line[2:end])
            codes = UInt8[]
            chars = Char[]
        elseif name !== nothing # junk before the first record is discarded
            for c in line # a mid-line '>' is just a junk base here
                push!(codes, _naive_code(c))
                push!(chars, c)
            end
        end
    end
    name === nothing || push!(recs, (name, codes, chars))
    wins = Vector{Tuple{String,Int,Vector{UInt8},String}}()
    for (nm, cs, chs) in recs, s in 1:(length(cs) - k + 1)
        push!(wins, (nm, s, cs[s:s+k-1], String(chs[s:s+k-1])))
    end
    return wins
end

# ==============================================================================
# Correctness suite
# ==============================================================================

function run_test()
    dir = mktempdir(prefix = "indexsample_")
    k = 50
    n = 64

    # ==========================================================================
    # A. mutate: err = 0 is a no-op; err > 0 keeps the length and changes bases
    # ==========================================================================
    x = UInt8.(rand(MersenneTwister(1), 0:3, 1000))
    @assert Utils.mutate(x, 0.0; rng = MersenneTwister(2)) == x "mutate(x, 0) is not a no-op"
    m = Utils.mutate(x, 0.1; rng = MersenneTwister(3))
    @assert length(m) == length(x) "mutate changed the length (ins != del)"
    @assert m != x "mutate(x, 0.1) changed nothing"
    @info "A. Utils.mutate no-op + shape OK"

    # ==========================================================================
    # B. synthetic edge-case reference: err = 0 sampling is EXACT vs the naive
    #    parser (headers, locations and letters), only valid windows are ever
    #    sampled, and the default N-FREE pool never touches a window holding
    #    N/IUPAC/junk while nfree = false can and does
    # ==========================================================================
    small = _ensure_test_fasta(joinpath(dir, "ref.fasta"); k)
    nwins = _naive_windows(small, k)
    wins = Dict((w[1], w[2]) => w[3] for w in nwins)
    # the N-free pool = naive windows whose RAW letters are all clean ACGT
    # (exactly what nfree = true may sample), plus per-record start lists
    isclean(w) = all(c -> c == 'A' || c == 'C' || c == 'G' || c == 'T', w[4])
    nfreewins = Set((w[1], w[2]) for w in nwins if isclean(w))
    pools(nf::Bool) = begin
        d = Dict{String,Vector{Int}}()
        for w in nwins
            nf && !isclean(w) && continue
            push!(get!(d, w[1], Int[]), w[2])
        end
        d
    end
    nfree_by_rec = pools(true)
    all_by_rec = pools(false)
    @info "B. synthetic reference: $(length(wins)) valid k=$k windows, " *
          "$(length(nfreewins)) N-free"
    s1 = sample_reads(small; n, k, err = 0.0, seed = 7)
    for i in 1:n
        h = _parse_read_head(s1.heads[i])
        @assert h !== nothing "unreadable header: $(s1.heads[i])"
        @assert h.id == ">read_$(i)" && h.len == k "bad header fields: $(s1.heads[i])"
        @assert haskey(wins, (h.src, h.start)) "sampled a non-window: $(s1.heads[i])"
        @assert (h.src, h.start) in nfreewins "nfree=true sampled an N-carrying window: $(s1.heads[i])"
        @assert s1.reads[i] == wins[(h.src, h.start)] "err=0 read $i != naive extraction ($(s1.heads[i]))"
    end
    @info "  err=0: $n reads bitwise equal to the naive windows (locations included, all N-free)"

    # even draw (the default): per-record counts within 1 read of the exact
    # quota n*W_j/total (sum = n) and EXACT equal-strata midpoints OVER THE
    # POOL's ascending start list (not the raw 1:len range); n >> W forces
    # the with-replacement cycling (still exact); nfree = false checks the
    # raw pool and MUST land on N-carrying windows here (chrA/chrF have no
    # clean ones at all, the chrE N-gap kills its middle)
    even_check = function (s, nreads; nf::Bool = true)
        (rheads, _, _, _) = _ref_table(_mmap_fasta(small))
        byrec = nf ? nfree_by_rec : all_by_rec
        tot = nf ? length(nfreewins) : length(wins)
        g = 0
        for j in unique(s.recs)
            nm = String(rheads[j][2:end])
            pool = get(byrec, nm, Int[])
            sel = Int[s.starts[t] for t in 1:nreads if s.recs[t] == j]
            cj = length(sel)
            @assert cj == 0 || !isempty(pool) "even draw: record $j ($nm) sampled with an empty pool"
            W = length(pool)
            ranks = [cj <= W ? 1 + div((2t - 1) * W, 2cj) : ((t - 1) % W) + 1
                     for t in 1:cj]
            @assert sel == [pool[r] for r in ranks] "even draw: record $j ($nm) starts off the pool strata midpoints"
            @assert abs(cj - nreads * W / tot) <= 1 + 1e-9 "even draw: record $j ($nm) count off quota"
            g += cj
        end
        @assert g == nreads "even draw: allocated $g of $nreads reads"
    end
    even_check(s1, n)
    s5 = sample_reads(small; n = 5000, k, err = 0.0, seed = 5) # nj > W on the small records
    even_check(s5, 5000)
    s1x = sample_reads(small; n, k, err = 0.0, seed = 7, nfree = false)
    even_check(s1x, n; nf = false)
    @assert any(((_parse_read_head(s1x.heads[i]).src,
                  _parse_read_head(s1x.heads[i]).start) ∉ nfreewins) for i in 1:n) "nfree=false must reach N-carrying windows here"
    @assert "chrF all_n" ∉ Set(_parse_read_head(h).src for h in s1.heads) "the all-N record must never be sampled (nfree=true)"
    @info "  even draw: counts ∝ pool sizes (±1), exact pool-strata midpoints, wrap exact; N-gap/all-N windows excluded (nfree=false reaches them)"

    # ==========================================================================
    # C. determinism: :even selection is SEED-FREE (at err = 0 the seed does
    #    not matter at all); :random is seed-bound; the seed drives mutation
    # ==========================================================================
    s2 = sample_reads(small; n, k, err = 0.0, seed = 7)
    @assert s1.reads == s2.reads && s1.heads == s2.heads "same seed, different output"
    s3 = sample_reads(small; n, k, err = 0.0, seed = 8)
    @assert s3.reads == s1.reads && s3.heads == s1.heads "even draw must be seed-free at err = 0"
    s3r = sample_reads(small; n, k, err = 0.0, seed = 8, draw = :random)
    @assert s3r.reads != s1.reads "random draw must depend on the seed"
    s3m1 = sample_reads(small; n, k, err = 0.1, seed = 9)
    s3m2 = sample_reads(small; n, k, err = 0.1, seed = 10)
    @assert s3m1.reads != s3m2.reads "the seed must drive the mutations"
    @assert s1x.heads != s1.heads "the nfree toggle must change the pool on this fixture"
    @info "C. determinism: even draw seed-free, random draw seed-bound, mutation seeded, nfree-sensitive OK"

    # ==========================================================================
    # D. err > 0: length preserved, every read differs from its window, and
    #    (when the expected unmutated run is long) shares a 16-mer with it
    # ==========================================================================
    e = 0.1
    n_err = ceil(Int, k * e)
    for dr in (:even, :random)
        s4 = sample_reads(small; n, k, err = e, seed = 9, draw = dr)
        for i in 1:n
            h = _parse_read_head(s4.heads[i])
            ref = wins[(h.src, h.start)]
            rd = s4.reads[i]
            @assert length(rd) == k "mutated read $i changed length"
            @assert rd != ref "err=$e read $i identical to its window"
            if k >= 32 * n_err
                @assert _shared16(ref, rd) >= 1 "err=$e read $i shares no 16-mer with its window"
            end
        end
    end
    @info "D. err=$e: $n reads differ from their windows and stay k long (both draws)"

    # ==========================================================================
    # E. gen/verify roundtrips on the synthetic file (both err = 0 and err > 0,
    #    default-name parameter parsing, gen caching)
    # ==========================================================================
    out0 = run_gen(fasta = small, out = joinpath(dir, "s0.fasta"), n = n, k = k,
                   err = 0.0, seed = 7, force = true)
    run_verify(out = out0, fasta = small, k = k, err = 0.0, seed = 7, n = n,
               draw = :even, nfree = true, rows = 8)
    out1 = run_gen(fasta = small, out = joinpath(dir, "s1.fasta"), n = n, k = k,
                   err = e, seed = 9, force = true)
    run_verify(out = out1, fasta = small, k = k, err = e, seed = 9, n = n,
               draw = :even, nfree = true, rows = 8)
    @assert run_gen(fasta = small, out = out1, n = n, k = k, err = e,
                    seed = 9) == out1 # cached: the second gen must skip
    # the RANDOM draw roundtrip via its tagged default name (every parameter,
    # draw + nfree included, parsed back from the tags)
    outr = run_gen(fasta = small, n = n, k = k, err = e, seed = 9,
                   draw = :random, force = true)
    @assert endswith(outr, "_drandom.fasta") "random draw must tag its default name"
    run_verify(out = outr, fasta = small, rows = 8)
    # the nfree = FALSE roundtrip via its `_withn` tag (N-carrying windows are
    # legal again; verify parses the tag back and drops the N-free check)
    outn = run_gen(fasta = small, n = n, k = k, err = e, seed = 9,
                   nfree = false, force = true)
    @assert endswith(outn, "_withn.fasta") "nfree=false must tag its default name"
    run_verify(out = outn, fasta = small, rows = 8)
    # tag parsing: untagged = even + N-free; _drandom = random draw; _withn =
    # nfree = false; fixed order when both apply
    pn = _params_from_name("x.indexsample_n8_k50_e0.1_s9.fasta")
    @assert pn !== nothing && pn.draw == :even && pn.nfree "untagged name must parse as even + N-free"
    pn = _params_from_name("x.indexsample_n8_k50_e0.1_s9_drandom.fasta")
    @assert pn !== nothing && pn.draw == :random && pn.nfree
    pn = _params_from_name("x.indexsample_n8_k50_e0.1_s9_withn.fasta")
    @assert pn !== nothing && pn.draw == :even && !pn.nfree
    pn = _params_from_name("x.indexsample_n8_k50_e0.1_s9_drandom_withn.fasta")
    @assert pn !== nothing && pn.draw == :random && !pn.nfree
    @info "E. gen/verify roundtrips (even + random, N-free + withn, err=0 and err=$e) + caching + name tags OK"

    # ==========================================================================
    # F. the human reference end-to-end (if present): real headers/spans at
    #    chromosome scale, gen -> verify roundtrip, err = 0 exactness at full k
    # ==========================================================================
    if isfile(IS_FASTA)
        nh = 8
        kh = 20_000
        @info "F. human reference end-to-end (n=$nh, k=$kh)"
        outh = run_gen(fasta = IS_FASTA, out = joinpath(dir, "human.fasta"),
                       n = nh, k = kh, err = 0.02, seed = 42, force = true)
        run_verify(out = outh, fasta = IS_FASTA, k = kh, err = 0.02, seed = 42,
                   n = nh, draw = :even, nfree = true, rows = 8)
        sh = sample_reads(IS_FASTA; n = nh, k = kh, err = 0.0, seed = 3)
        raw = _mmap_fasta(IS_FASTA)
        (rheads, rlo, rhi, rlen) = _ref_table(raw)
        ridx = Dict{Int,_RecIndex}()
        for i in 1:nh
            h = _parse_read_head(sh.heads[i])
            @assert h.len == kh && 1 <= h.start <= rlen[sh.recs[i]] - kh + 1 "bad human window: $(sh.heads[i])"
            idx = get!(() -> _rec_index(raw, rlo[sh.recs[i]], rhi[sh.recs[i]]),
                       ridx, sh.recs[i])
            @assert sh.reads[i] == _extract_read!(Vector{UInt8}(undef, kh), raw,
                                                  idx, h.start, kh) "human err=0 read $i mismatch"
            @assert occursin("start=$(sh.starts[i])", sh.heads[i])
        end
        @info "  human err=0 reads exact at chromosome scale OK"
    else
        @warn "human fasta not found, skipping F" IS_FASTA
    end

    @info "ALL INDEXSAMPLE CORRECTNESS TESTS PASSED"
    return nothing
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : ARGS[1]
    arg2 = length(ARGS) > 1 ? ARGS[2] : ""
    mode == "gen" && run_gen(fasta = isempty(arg2) ? IS_FASTA : arg2)
    mode == "gen!" && run_gen(fasta = isempty(arg2) ? IS_FASTA : arg2, force = true)
    mode == "verify" && (isempty(arg2) ? run_verify() : run_verify(out = arg2))
    mode == "test" && run_test()
    mode == "all" && begin
        run_test()
        _out = run_gen()
        run_verify(out = _out)
    end
    mode in ("gen", "gen!", "verify", "test", "all") ||
        error("unknown mode $mode (use gen|gen!|verify|test|all)")
end
