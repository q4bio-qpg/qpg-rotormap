# ==============================================================================
# probe_locus_windows.jl -- CPU float64 reconciliation of the mapping score
# around ONE read's true locus.
#
# WHAT
#   Pull the index's OWN window records near the read's provenance location
#   (same accession, |starts - start0| <= PLW_MARGIN), re-encode them from the
#   reference with the float64 CPU golden-reference encoder
#   (_ref_rope_frag_real, normalize = 0 -- the exact encoding math), and
#   inner-product against the read's (leading-k) encoding.  Each window row
#   also gets the dot against the index's STORED fp8 column -- the exact
#   database values the GPU GEMM multiplies.
#
# WHY
#   probe_top1_scores on maize_1X_15_shuf vs the s5m1c5 maize index found a
#   read (NC_050096.1:226793838:false) whose top-20 GEMM vals are all
#   ~0.993..1.040 -- while its true-location Levenshtein is 2492/23561
#   (10.6% edits).  Separating (a) fresh float64 rope inner products of the
#   LOCAL windows from (b) stored-fp8-column dots attributes the near-1.0
#   vals (duplicated-window stack? quantization? layout mismatch?).
#
# NOTES
#   * the GEMM statistic is the real dot of the split [Re; Im] vectors ==
#     Re<z_window, z_read>; the complex modulus is reported alongside.
#   * fwd columns compare against the window as-is; revcomp-strand columns
#     (index cols 2*n_frag) compare against the window's reverse complement.
#   * the exact-locus window (offset 0) is added even when the window grid
#     (stride kstep) has no column exactly at start0 (db8 = NaN there).
#   * reads longer than k are trimmed to their leading k bases (the flow's
#     behavior); rc reads are not supported here (assert).
#
# USAGE
#   PLW_REF=<primary12.fna> PLW_INDEX=<s5m1c5 index .bin> \
#   PLW_READS=<reads.fasta> \
#     julia --project=RotorMap RotorMap/experiments/probe_locus_windows.jl
#   PLW_HEAD (NC_050096.1:226793838:false), PLW_MARGIN (15000),
#   PLW_MAXROWS (80)
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("encode", "reference.jl") # _ref_rope_frag_real + RopeEncoder (pure Base)

using Serialization
using Printf
using Mmap
using LinearAlgebra
import DLFP8Types # the bin embeds Float8_E4M3FN arrays -- deserialize needs the module loaded

const PLW_REF = get(ENV, "PLW_REF", "")
const PLW_INDEX = get(ENV, "PLW_INDEX", "")
const PLW_READS = get(ENV, "PLW_READS", "")
const PLW_HEAD = get(ENV, "PLW_HEAD", "NC_050096.1:226793838:false")
const PLW_MARGIN = parse(Int, get(ENV, "PLW_MARGIN", "15000"))
const PLW_MAXROWS = parse(Int, get(ENV, "PLW_MAXROWS", "80"))

const NL = UInt8('\n')
const CR = UInt8('\r')
const GT = UInt8('>')

# fasta record scan (probe_levenshtein's): call f(header, lo, hi) per record
function _each_record(f, raw::AbstractVector{UInt8})
    n = length(raw)
    s = 1
    while s <= n && raw[s] != GT
        s += 1
    end
    while s <= n
        h0 = s + 1
        h1 = h0
        while h1 <= n && raw[h1] != NL
            h1 += 1
        end
        head = rstrip(String(raw[h0:h1-1]), '\r')
        e = h1 + 1
        while e <= n && raw[e] != GT
            e += 1
        end
        f(head, h1 + 1, e - 1)
        s = e
    end
    return nothing
end

"""ACGT bytes -> 0:3 codes; anything else -> 0, counted (returned)."""
function _codes!(dst::Vector{UInt8}, src::AbstractVector{UInt8})
    n = 0
    @inbounds for i in eachindex(src)
        b = src[i]
        c = b == UInt8('A') ? UInt8(0) : b == UInt8('C') ? UInt8(1) :
            b == UInt8('G') ? UInt8(2) : b == UInt8('T') ? UInt8(3) : UInt8(0)
        c == 0 && b != UInt8('A') && (n += 1)
        dst[i] = c
    end
    return n
end

"""Split [Re; Im] unit-energy ropes -> (Re<z0,z1>, |<z0,z1>|)."""
function rope_ip(e0::AbstractVector{<:Real}, e1::AbstractVector{<:Real})
    D = length(e0) >> 1
    ipre = dot(e0, e1)
    ipim = dot(view(e0, 1:D), view(e1, D+1:2D)) -
           dot(view(e0, D+1:2D), view(e1, 1:D))
    return ipre, abs(ComplexF64(ipre, ipim))
end

function main()
    # ---- the index: authentic window grid + stored fp8 columns ---------------
    @info "deserializing index" PLW_INDEX
    d = deserialize(PLW_INDEX)
    fmt = get(d, :format, nothing)
    @info "index" format = fmt k = d.k kstep = d.kstep s = d.s m = d.m c = d.c normalize = d.normalize n_frag = d.n_frag
    @assert fmt in ("indexreal.v1", "indexreal.fp8.v1",
                    "indexflowreal.v1", "indexflowreal.fp8.v1") "unexpected format"
    @assert (Int(d.s), Int(d.m), Int(d.c)) == (5, 1, 5) "expected the s5m1c5 index"
    @assert Int(d.normalize) == 0
    K = Int(d.k)
    kstep = Int(d.kstep)

    # ---- reference: accession -> contiguous uppercase sequence ---------------
    ref = open(PLW_REF) do io
        Mmap.mmap(io)
    end
    refseqs = Dict{String,Vector{UInt8}}()
    _each_record(ref) do head, lo, hi
        acc = String(split(head)[1])
        v = Vector{UInt8}(uppercase(String(ref[lo:hi])))
        filter!(b -> b != NL && b != CR, v)
        refseqs[acc] = v
    end
    @info "reference records" n = length(refseqs)

    # ---- the read (by provenance key), trimmed to its leading K bases --------
    reads = open(PLW_READS) do io
        Mmap.mmap(io)
    end
    rseq = UInt8[]
    found = Ref(false)
    _each_record(reads) do head, lo, hi
        found[] && return
        f = split(head, ':')
        length(f) == 3 && string(f[1], ":", f[2], ":", split(f[3], '#')[1]) ==
        PLW_HEAD || return # NOTE: _each_record strips '>', so f[1] is the bare accession
        append!(rseq, filter!(b -> b != NL && b != CR,
                              Vector{UInt8}(@view reads[lo:hi])))
        found[] = true
    end
    found[] || error("read with provenance `$PLW_HEAD` not found in the reads")
    length(rseq) >= K || error("read length $(length(rseq)) < k = $K")
    resize!(rseq, K)
    pf = split(PLW_HEAD, ':')
    acc_r = String(pf[1])
    start0 = parse(Int, pf[2])
    @assert split(pf[3], '#')[1] == "false" "rc reads not supported here"
    @info "read" head = PLW_HEAD start0 trimmed_to = K

    # ---- the read's float64 rope encoding ------------------------------------
    re = RopeEncoder(k = K, s = Int(d.s), m = Int(d.m), c = Int(d.c))
    rcode = Vector{UInt8}(undef, K)
    nn = _codes!(rcode, rseq)
    nn > 0 && @warn "non-ACGT bases in the read mapped to A" n = nn
    er, _ = _ref_rope_frag_real(rcode, re; normalize = 0)
    D2 = length(er)

    # ---- index columns near the locus (+ the exact-locus window) -------------
    rec = refseqs[acc_r] # the read's reference record (contiguous)
    rows = Tuple{Int,Int,Symbol,Float64,Float64,Float64}[] # start, offset, strand, re, abs, db8
    _codes!(rcode, rec[start0+1:start0+K]) # reuse the buffer: exact-locus window
    e0, _ = _ref_rope_frag_real(rcode, re; normalize = 0)
    ip0, ia0 = rope_ip(er, e0)
    push!(rows, (start0, 0, :exact, ip0, ia0, NaN))

    nsel = 0
    for c in eachindex(d.starts)
        h1 = split(d.heads[c])[1]
        accc = String(startswith(h1, '>') ? h1[2:end] : h1)
        accc == acc_r || continue
        st = Int(d.starts[c])
        0 <= st && st + K <= length(rec) || continue
        abs(st - start0) <= PLW_MARGIN || continue
        nsel += 1
        codes = Vector{UInt8}(undef, K)
        _codes!(codes, rec[st+1:st+K])
        if Int(d.strand[c]) != 0 # revcomp-strand column: compare its actual content
            reverse!(codes)
            for i in eachindex(codes)
                codes[i] = 3 - codes[i]
            end
        end
        ew, _ = _ref_rope_frag_real(codes, re; normalize = 0)
        ipr, iab = rope_ip(er, ew)
        db8 = dot(Float64.(view(d.embeds8, :, c)), er)
        push!(rows, (st, st - start0, Int(d.strand[c]) == 0 ? :fwd : :rc, ipr, iab, db8))
    end
    @info "local index columns" accession = acc_r margin = PLW_MARGIN n = nsel

    sort!(rows; by = t -> (abs(t[2]), t[3]))
    @printf("%11s %7s  %-6s %10s %10s %10s\n", "start", "offset", "strand",
            "fresh Re", "fresh |IP|", "db8 dot")
    shown = 0
    for t in rows
        shown >= PLW_MAXROWS && break
        @printf("%11d %7d  %-6s %10.6f %10.6f %10.6f\n", t[1], t[2], t[3],
                t[4], t[5], t[6])
        shown += 1
    end

    loc = collect(rows[2:end]) # exclude the exact-locus row
    !isempty(loc) || return
    _, j = findmax(t -> t[4], loc)
    @printf("\nSUMMARY stride %d | local cols %d | exact-locus fresh Re %.6f |IP| %.6f\n",
            kstep, nsel, ip0, ia0)
    @printf("best local fresh Re %.6f (|IP| %.6f) at offset %+d (%s)\n",
            loc[j][4], loc[j][5], loc[j][2], loc[j][3])
    cands = filter(t -> !isnan(t[6]), loc)
    _, j8 = findmax(t -> t[6], cands)
    @printf("best local db8 dot %.6f at offset %+d (%s)\n",
            cands[j8][6], cands[j8][2], cands[j8][3])
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
