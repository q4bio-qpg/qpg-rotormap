# =============================================================================
# harness.jl -- tiny end-to-end self-test scaffolding shared by the e2e
# experiments' `test` modes (extracted from legacy e2ehuman.jl; the fp16 and
# complex experiments keep their own quantization-specific variants locally):
#
#   _E2H_TEST_REF          path of the cached 1 Mbp two-record test reference
#   _ensure_e2h_test_ref   generate it once (deterministic; lowercase + IUPAC junk)
#   _e2h_ref_table         naive fasta record table (heads, byte spans, char counts)
#   _e2h_span_codes        scalar 2-bit scan of a record span (_LUT3 convention)
#   _e2h_sample_reads      sample n reads + sample provenance headers (+ mutate)
#   _e2h_build_index       miniature of the production fp8 index save (encode
#                          fwd+rc windows, transpose + e4m3 columns, serialize)
#   _ref_forward_pack      codes -> forward 2-bit packed UInt32 words
# REQUIRES (include beforehand, canonical order): common/dna.jl (mutate,
#   save_fasta, save), fasta/reader.jl (_next_rec3, _LUT3),
#   encode/ropeencoder.jl + encode/kernel.jl (encode_frag_real_batch!),
#   gemm/fp8_convert.jl (F8), index/load.jl (E2H_FP8_FORMAT).
# =============================================================================
using Mmap
using Random: Xoshiro

const E2H_TEST_BASES = 1_000_000 # tiny reference: two records, total bases
const E2H_TEST_N = 1000          # tiny test reads
const E2H_TEST_SEED = 42

const _E2H_TEST_REF = joinpath(_v3_data_dir(), "e2ehuman_ref1M.fasta")

# two records (~70/30 split) so the record-discrimination half of the location
# check is exercised; indexreal's builder style (lowercase + IUPAC junk,
# wrapped lines -- its wrapped() emits width+1 chars per line, so the file
# holds ~1.4% more than `total` bases).  Cached: regenerated only when
# missing/too small.
function _ensure_e2h_test_ref(; total::Int = E2H_TEST_BASES)
    isfile(_E2H_TEST_REF) && filesize(_E2H_TEST_REF) > total + 10_000 &&
        return _E2H_TEST_REF
    lenA = 7 * total ÷ 10
    lenB = total - lenA
    rs = MersenneTwister(2024)
    bases = collect("ACGT")
    iupac = collect("NRYSWKMBDHV")
    randseq(len) = String([bases[rand(rs, 1:4)] for _ in 1:len])
    function sprinkle(s)
        v = collect(s)
        for i in 1:97:length(v)
            v[i] = lowercase(v[i])
        end
        for i in 41:211:length(v)
            v[i] = rand(rs, iupac)
        end
        return String(v)
    end
    wrapped(s, width) = join((s[i:min(i + width, end)] for i in 1:width:length(s)), "\n")
    mkpath(dirname(_E2H_TEST_REF))
    @info "Generating the tiny $(total)-base test reference -> $_E2H_TEST_REF"
    open(_E2H_TEST_REF, "w") do io
        write(io, ">chrA e2ehuman tiny test record A\n",
              wrapped(sprinkle(randseq(lenA)), 70), "\n")
        write(io, ">chrB e2ehuman tiny test record B\n",
              wrapped(sprinkle(randseq(lenB)), 70), "\n")
    end
    return _E2H_TEST_REF
end

# record table of a fasta in the indexreal sense: (verbatim '>' heads,
# inclusive sequence byte spans, sequence-char counts -- every byte except
# \n/\r; junk/IUPAC -> G keeping its position)
function _e2h_ref_table(raw::Vector{UInt8})
    heads = String[]
    lo = Int[]
    hi = Int[]
    nchar = Int[]
    NL = UInt8('\n')
    CR = UInt8('\r')
    n = length(raw)
    s = _next_rec3(raw, 1, n) # junk before the first record is discarded
    while s <= n
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n)
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

# scalar scan of the record span [lo, hi] into 2-bit codes (the _LUT3
# convention; deliberately not the production SWAR path)
function _e2h_span_codes(raw::Vector{UInt8}, lo::Int, hi::Int)
    codes = Vector{UInt8}()
    @inbounds for p in lo:hi
        c = raw[p]
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        push!(codes, _LUT3[Int(c) + 1])
    end
    return codes
end

# codes -> forward 2-bit packed words (16 bases per UInt32), the production
# word convention (legacy fastareads_v3.jl's reference packer)
function _ref_forward_pack(codes::Vector{UInt8}, L::Int)
    words = fill(UInt32(0), cld(L, 16)) # 16 bases (32 bits) per word
    for pos in 1:L
        words[(pos - 1) >> 4 + 1] |= UInt32(codes[pos]) << (2 * ((pos - 1) & 15))
    end
    return words
end

"""
    _e2h_sample_reads(ref; n, k, err = 0.0, seed, out) -> out::String

Sample `n` reads of length `k` from the tiny reference (the sample flow's
:random draw: n iid uniform windows, with replacement), optionally mutate each
with mutate (err = fraction; err = 0 keeps the exact window), and save
as plain fasta with the sample flow's provenance headers.  Deterministic in
(ref, n, k, err, seed).  The output name mirrors the sample flow's parameter
record so the ground truth stays self-describing.
"""
function _e2h_sample_reads(ref::String; n::Int, k::Int, err::Real = 0.0,
                           seed::Int = E2H_TEST_SEED,
                           out::String = string(splitext(ref)[1],
                                                ".sample_n$(n)_k$(k)_e$(err)_s$(seed).fasta"))
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    w = [max(l - k + 1, 0) for l in rlen] # sampling weights: valid windows
    total = sum(w)
    total > 0 || error("reference too short for k = $k")
    cum = accumulate(+, w)
    rng = Xoshiro(seed)
    recs = Vector{Int}(undef, n)
    starts = Vector{Int}(undef, n)
    for i in 1:n # every choice up front: thread-schedule-free determinism
        r = rand(rng, Int64(1):total)
        j = searchsortedfirst(cum, r)
        recs[i] = j
        starts[i] = Int(r - (j > 1 ? cum[j-1] : Int64(0)))
    end
    rngs = [Xoshiro(rand(rng, Int64)) for _ in 1:n] # one child rng per read
    reads = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        codes = _e2h_span_codes(raw, rlo[recs[i]], rhi[recs[i]])
        win = codes[starts[i]:starts[i]+k-1]
        err > 0 && (win = mutate(win, err; rng = rngs[i]))
        reads[i] = win
    end
    heads = [">read_$(i) start=$(starts[i]) len=$(k) src=$(_rec_name(rheads[recs[i]]))"
             for i in 1:n] # src = the record NAME (first token, sans '>)
    save_fasta(reads, out; heads)
    @info "sampled $n reads (err = $err) -> $out"
    return out
end

"""
    _e2h_build_index(ref; k, kstep, normalize = 0, binfile) -> binfile::String

A miniature of indexreal's fp8 `save`: every k-window (step kstep, last
window fully inside the record) of every record of the tiny reference,
forward AND reverse complement, encoded with the PRODUCTION kernel
(`encode_frag_real_batch!`), transposed + quantized to e4m3 columns and
serialized under the production naming `<ref>.indexreal_fp8.bin`
(format "indexreal.fp8.v1") -- so the test exercises the same
load path as the human run.
"""
function _e2h_build_index(ref::String; k::Int = 20_000, kstep::Int = max(k ÷ 10, 1),
                          normalize::Int = 0,
                          binfile::String = string(splitext(ref)[1], ".indexreal_fp8.bin"))
    isfile(binfile) && return binfile # shared fixture: a previous run built it
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    rdim = 2 * re.m * 4^re.c
    W = cld(k, 16)
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    gheads = String[] # per-window location meta (file order)
    gstarts = Int[]
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        for wi in 0:((rlen[j] - k) ÷ kstep)
            push!(gheads, rheads[j])
            push!(gstarts, wi * kstep + 1)
        end
    end
    nwin = length(gheads)
    # a 1M reference holds ~nwin windows: materialize ALL fragment words
    # (a few MB) and encode both strands in one kernel call each
    fw = Vector{UInt32}(undef, W * nwin)
    rcw = Vector{UInt32}(undef, W * nwin)
    g = 0
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        codes = _e2h_span_codes(raw, rlo[j], rhi[j]) # full record, once
        @assert length(codes) == rlen[j]
        for wi in 0:((rlen[j] - k) ÷ kstep)
            g += 1
            s = wi * kstep
            copyto!(fw, (g - 1) * W + 1, _ref_forward_pack(codes[s+1:s+k], k), 1, W)
            rcc = [UInt8(0x03 - codes[s+k+1-t]) for t in 1:k] # revcomp codes
            copyto!(rcw, (g - 1) * W + 1, _ref_forward_pack(rcc, k), 1, W)
        end
    end
    @assert g == nwin
    dest_f = CUDA.zeros(Float16, nwin, rdim)
    dest_r = CUDA.zeros(Float16, nwin, rdim)
    dn_f = CUDA.zeros(Float32, re.m, nwin)
    dn_r = CUDA.zeros(Float32, re.m, nwin)
    encode_frag_real_batch!(dest_f, dn_f, re, cu(fw); normalize)
    encode_frag_real_batch!(dest_r, dn_r, re, cu(rcw); normalize)
    ef = Matrix{Float16}(undef, nwin, rdim)
    er = Matrix{Float16}(undef, nwin, rdim)
    nf = Matrix{Float32}(undef, re.m, nwin)
    nr = Matrix{Float32}(undef, re.m, nwin)
    copyto!(ef, dest_f)
    copyto!(er, dest_r)
    copyto!(nf, dn_f)
    copyto!(nr, dn_r)
    # transposed e4m3 columns -- the production save's layout and the exact
    # F8.(Float32.(.)) quantization (rows 1:nwin fwd, nwin+1:2nwin rc)
    embeds8 = Matrix{F8}(undef, rdim, 2 * nwin)
    @views embeds8[:, 1:nwin] .= F8.(Float32.(permutedims(ef)))
    @views embeds8[:, nwin+1:2*nwin] .= F8.(Float32.(permutedims(er)))
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    copyto!(view(norms, :, 1:nwin), nf)
    copyto!(view(norms, :, nwin+1:2*nwin), nr)
    nt = (format = E2H_FP8_FORMAT, source = abspath(ref), k, kstep, s = re.s,
          m = re.m, c = re.c, normalize, fp8 = true, n_frag = nwin, embeds8,
          norms, heads = vcat(gheads, gheads), starts = vcat(gstarts, gstarts),
          strand = vcat(fill(UInt8(0), nwin), fill(UInt8(1), nwin)))
    save(nt, binfile)
    @info "tiny index built" binfile k kstep nwin cols = 2 * nwin rdim
    return binfile
end
