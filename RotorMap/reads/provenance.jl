# =============================================================================
# provenance.jl -- ground-truth provenance header parsing (UNIFIED).
# The legacy tree carried three copies of this parser (e2ehuman.jl /
# e2ehuman16.jl / e2ecomplex.jl); this is e2ehuman.jl's multi-format superset,
# which handles every format the shorter copies did (verified: the fp16 and
# complex copies only parsed format 1, covered below).
#
# REQUIRES: nothing.
# =============================================================================

# ------------------------------------------------------------------------------
# Ground truth: TWO provenance header formats
#
#   1. sample: >read_<i> start=<1-based> len=<k> src=<record name>
#      (`src` is the record header's FIRST TOKEN -- before the first
#      whitespace -- sans the leading '>'; older data may carry the verbatim
#      multi-token header there: everything after " src=" is free text and
#      never parsed, so both spellings parse identically)
#
#   2. colon eval (reads generated from GCF_000001405.40_GRCh38.p14_primary25
#      by maksym's simulator):
#      >record:start:revcomp       e.g. >NC_000001.11:80433546:false
#      (a re-drawn duplicate locus may carry a trailing `#<n>` suffix, e.g.
#      `>NC_000011.10:49059499:true#2` -- ignored: same provenance)
#      `record` is the record ACCESSION (the first whitespace-delimited token
#      of the '>' header, sans '>'), `start` is the 0-BASED offset of the
#      read's first base in the record (verified base-by-base against the
#      reference fasta), `revcomp` (true/false) marks the read as the reverse
#      complement of ref[start:start+readlen].  No length is carried: the
#      fasta reader keeps each record's FIRST k bases (records shorter than k
#      are skipped, longer ones are trimmed to exactly k), so the mapped
#      fragment covers ref[start:start+k] -- for a revcomp read its source
#      window sits at the END of ref[start:start+readlen], which always
#      overlaps ref[start:start+k] whenever readlen < 2k (true for every eval
#      batch: readlen <= 29,718 < 2*20,000).  The parser normalizes `start`
#      to 1-based and returns len = nothing; the scoring loop then uses the
#      dbk-long window at that start -- the db's rc columns are the SAME
#      locations, so no strand special-casing is needed.
# ------------------------------------------------------------------------------
# the record's NAME: the header's first whitespace token, sans a leading '>'
# (the naming convention for sample provenance src= values and for the match
# tables' record fields)
_no_gt(s::AbstractString) = startswith(s, ">") ? String(s[2:end]) : String(s)
_rec_name(h::AbstractString) = _no_gt(split(h)[1])

function parse_read_head(h::AbstractString)
    startswith(h, ">") || return nothing
    if startswith(h, ">read_") # format 1: sample
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
    # format 2: colon eval provenance >record:start:revcomp[#<dup>]; `start`
    # is 0-based -> normalized to 1-based, length unknown (the reader trims
    # every kept record to exactly the db window length k); the optional
    # trailing #<n> duplicate-locus marker is stripped
    f = split(h, ':')
    length(f) == 3 || return nothing
    start = tryparse(Int, f[2])
    start === nothing && return nothing
    rev = split(f[3], '#')[1]
    rev in ("true", "false") || return nothing
    return (id = String(h), start = start + 1, len = nothing,
            src = String(f[1][2:end]))
end
