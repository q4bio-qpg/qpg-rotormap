# =============================================================================
# gen_dna.jl -- the CLI for the random-DNA layer (common/dna.jl): generate a
# random ACGT reference and/or sample + mutate reads from a fasta, without
# writing any Julia.  CPU-only (no CUDA include), -t N speeds the mutation.
#
#   julia --project=RotorMap -t 8 RotorMap/tools/gen_dna.jl ref <N> [out.fasta] [--seed=S] [--head=NAME]
#   julia --project=RotorMap -t 8 RotorMap/tools/gen_dna.jl reads <ref.fasta> <K> <N> [out.fasta] [--err=F] [--seed=S]
#
# ref     generate_reference(N; seed) -> save_fasta: N bases drawn uniformly
#         from ACGT (default out ref_n<N>_s<SEED>.fasta, default record
#         header ">chr0", --head= overrides the WHOLE header line sans '>').
#
# reads   N reads of length K sampled from ref.fasta: iid uniform windows
#         WITH replacement over ALL records' valid K-windows (records
#         shorter than K contribute nothing -- the indexreal window rule),
#         each mutated at err (dna.jl's mutate: ceil(K*err) edits, ~1/3
#         substitutions + ~1/3 insertions + ~1/3 deletions; equal ins/del
#         counts keep every read exactly K letters long).  The selection AND
#         the mutations are driven by --seed (Xoshiro; per-read child rngs
#         drawn serially up front -- the generate_reads pattern -- so
#         threading never leaks into the result and a re-run with the same
#         seed is bitwise identical).  Headers carry the SAMPLE provenance
#             >read_<i> start=<1-based> len=<K> src=<record name>
#         (the reads/sample.jl contract), so tools/verify_maps.jl scores the
#         output directly after a tools/map.jl run.  Default out
#         <ref>.reads_n<N>_k<K>_e<ERR>_s<SEED>.fasta.
#
# The reads mode holds the whole reference in memory as codes -- for
# SYNTHETIC-scale files (the ref mode's output, multi-Mbp).  For genome-scale
# sampling use reads/sample.jl gen instead (mmap streaming, the :even/:random
# draws, the N-free window pool).
# =============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("common", "treearrays.jl") # dna.jl requires TreeArrays first
inc("common", "dna.jl")        # generate_reference / mutate / save_fasta

using Random
using Base.Threads
using Mmap

_no_gt(s::AbstractString) = startswith(s, ">") ? String(s[2:end]) : String(s)

# ------------------------------------------------------------------------------
# a small whole-record fasta reader: line-start '>' splits records, the
# header's FIRST WHITESPACE TOKEN is the record NAME (the provenance src=
# convention -- multi-token headers cannot break the sample format), every
# sequence byte maps through the toolchain's code LUT (ACGT/acgt -> 0..3,
# every other byte -> G = 0x02 keeping its position; \n/\r are not
# positions).  Returns Vector{(name, codes)}.
# ------------------------------------------------------------------------------
function _read_records(path::String)
    raw = Mmap.mmap(path)
    recs = Tuple{String,Vector{UInt8}}[]
    name = nothing
    codes = UInt8[]
    n = length(raw)
    i = 1
    while i <= n
        j = findnext(c -> c == UInt8('\n'), raw, i)
        j === nothing && (j = n + 1)
        if raw[i] == UInt8('>') && (i == 1 || raw[i-1] == UInt8('\n'))
            name === nothing || push!(recs, (name, codes))
            name = _no_gt(split(String(raw[i+1:j-1]))[1])
            codes = UInt8[]
        elseif name !== nothing # junk before the first record is discarded
            for p in i:j-1
                b = raw[p]
                b == UInt8('\r') && continue
                c = b == UInt8('A') || b == UInt8('a') ? 0x00 :
                    b == UInt8('C') || b == UInt8('c') ? 0x01 :
                    b == UInt8('G') || b == UInt8('g') ? 0x02 :
                    b == UInt8('T') || b == UInt8('t') ? 0x03 : 0x02
                push!(codes, c)
            end
        end
        i = j + 1
    end
    name === nothing || push!(recs, (name, codes))
    isempty(recs) && error("no fasta records in $path")
    return recs
end

# ------------------------------------------------------------------------------
# reads mode: iid uniform windows (with replacement) over all records' valid
# K-windows, mutate each (per-read child rngs, the generate_reads pattern),
# provenance headers -> save_fasta
# ------------------------------------------------------------------------------
function run_reads(ref::String, k::Int, n::Int; err::Real, seed::Int, out::String)
    recs = _read_records(ref)
    pool = Tuple{Int,Int}[] # (record index, 1-based window start)
    for j in eachindex(recs), s in 1:(length(recs[j][2]) - k + 1)
        push!(pool, (j, s))
    end
    isempty(pool) &&
        error("no valid k = $k windows in $ref (every record is shorter than k)")
    rng = Xoshiro(seed)
    sel = [pool[rand(rng, 1:length(pool))] for _ in 1:n]
    rngs = [Xoshiro(rand(rng, Int)) for _ in 1:n]
    reads = Vector{Vector{UInt8}}(undef, n)
    @threads for i in 1:n
        j, s = sel[i]
        @views reads[i] = mutate(recs[j][2][s:s+k-1], err; rng = rngs[i])
    end
    heads = [">read_$i start=$(sel[i][2]) len=$k src=$(recs[sel[i][1]][1])"
             for i in 1:n]
    save_fasta(reads, out; heads)
    nrec = length(unique(first.(sel)))
    @info "gen_dna reads: $n reads x $k (err = $err, seed = $seed) from " *
          "$nrec of $(length(recs)) record(s) -> $out " *
          "($(round(filesize(out) / 2^20; digits = 1)) MiB)"
    return out
end

# ------------------------------------------------------------------------------
# mode dispatch (positional args + --name=value flags, map.jl's spellings)
# ------------------------------------------------------------------------------
function main()
    getflag(name::String, default::String) =
        begin
            for a in ARGS, p in ("--" * name * "=", name * "=")
                startswith(a, p) && return String(a[(length(p) + 1):end])
            end
            return default
        end
    posargs = filter(a -> !startswith(a, "-"), ARGS)
    mode = isempty(posargs) ? "" : posargs[1]
    if mode == "ref"
        length(posargs) >= 2 ||
            error("usage: gen_dna.jl ref <N> [out.fasta] [--seed=S] [--head=NAME]")
        N = parse(Int, posargs[2])
        seed = parse(Int, getflag("seed", "1"))
        head = getflag("head", "chr0")
        out = length(posargs) > 2 ? posargs[3] : "ref_n$(N)_s$(seed).fasta"
        t = @elapsed ref = generate_reference(N; seed)
        save_fasta([ref], out; heads = [">" * head])
        @info "gen_dna ref: $N random ACGT bases (seed = $seed, header >$head) " *
              "-> $out ($(round(filesize(out) / 2^20; digits = 1)) MiB) in " *
              "$(round(t; digits = 2)) s"
    elseif mode == "reads"
        length(posargs) >= 4 ||
            error("usage: gen_dna.jl reads <ref.fasta> <K> <N> [out.fasta] [--err=F] [--seed=S]")
        ref = posargs[2]
        k = parse(Int, posargs[3])
        n = parse(Int, posargs[4])
        err = parse(Float64, getflag("err", "0.05"))
        seed = parse(Int, getflag("seed", "42"))
        out = length(posargs) > 4 ? posargs[5] :
              string(ref, ".reads_n$(n)_k$(k)_e$(err)_s$(seed).fasta")
        run_reads(ref, k, n; err, seed, out)
    else
        error("unknown mode '$mode' (use ref|reads; see the header of " *
              "RotorMap/tools/gen_dna.jl)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
