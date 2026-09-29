# =============================================================================
# separate_provenance.jl -- split a provenance-bearing sampled reads FASTA
# (RotorMap/reads/sample.jl `gen` output: headers like
#   >read_<i> start=<1-based> len=<k> src=<record name>)
# into a PLAIN reads FASTA (headers stripped to the bare read id -- no
# provenance left anywhere in the file, like a real sequencer batch) and a
# standalone PROVENANCE TABLE (.tsv, one row per read) that
# tools/verify_maps.jl accepts in place of a reads FASTA:
#
#   julia --project=RotorMap RotorMap/tools/separate_provenance.jl \
#       <reads.fasta> [outprefix]
#
#   <reads.fasta>   the provenance-bearing sampled reads FASTA
#   [outprefix]     optional output prefix (default: <reads.fasta> minus its
#                   trailing `.fasta`).  Outputs (same directory as the
#                   prefix):
#                     <prefix>.plain.fasta     the SAME sequences, bare id
#                                              headers, original line
#                                              wrapping (bytes copied
#                                              verbatim)
#                     <prefix>.provenance.tsv  header row `id src start len`,
#                                              then one tab-separated row per
#                                              read (start 1-based, len 0 for
#                                              colon reads that carry none)
#
# CPU-only.  Streams the input record by record over an mmap'd file and
# NEVER decodes DNA: sequence bytes are copied verbatim, so the output
# sequences are bitwise identical to the input.  Both provenance formats of
# reads/provenance.jl are handled: the sample format keeps its first token
# as the id (>read_<i>); the colon eval format (>acc:start:rc) -- where the
# WHOLE header is the id -- gets a synthesized `r<i>` id (i = the record
# number in the file), its original header being the provenance itself
# (start/src survive in the table).  A header from which NO provenance
# parses is an error: this tool's input is a sampled batch, never an
# already-plain one.
#
# REQUIRES: reads/provenance.jl (parse_read_head).  CPU-only, no GPU.
# =============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))
isdefined(Main, :parse_read_head) || inc("reads/provenance.jl")

using Mmap
using Printf

_mmap_fasta(file::String) =
    filesize(file) == 0 ? UInt8[] : open(file, "r") do io
        Mmap.mmap(io)
    end

# ------------------------------------------------------------------------------
# Record walk -- the fastareads_v3/indexreal convention copied from
# reads/sample.jl: a '>' opens a record iff at position 1 or after \n/\r
# ------------------------------------------------------------------------------
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

function main()
    length(ARGS) < 1 &&
        error("usage: separate_provenance.jl <reads.fasta> [outprefix]")
    input = ARGS[1]
    isfile(input) || error("reads fasta not found: $input")
    prefix = length(ARGS) > 1 ? ARGS[2] :
             (occursin(r"\.fasta$", input) ?
                 input[1:(first(findlast(".", input)) - 1)] : input)
    plain = prefix * ".plain.fasta"
    tsv = prefix * ".provenance.tsv"

    raw = _mmap_fasta(input)
    n = length(raw)
    nreads = 0
    srcs = Set{String}()
    t0 = time()
    open(plain, "w") do p
        open(tsv, "w") do t
            println(t, "id\tsrc\tstart\tlen")
            i = 0
            s = _next_rec(raw, 1, n)
            while s <= n
                h = s
                while h <= n && raw[h] != UInt8('\n')
                    h += 1
                end
                hend = h - 1
                (hend >= s && raw[hend] == UInt8('\r')) && (hend -= 1)
                e = _next_rec(raw, s + 1, n)
                head = String(raw[s:hend])
                pr = parse_read_head(head)
                pr === nothing &&
                    error("record $(i + 1): no provenance parseable in header: $head")
                i += 1
                # sample format: the first token is the bare id; colon format:
                # the WHOLE header is the id (no whitespace in it) -> synthesize
                newid = pr.id === head ? "r$i" : _no_gt(pr.id)
                @printf(p, ">%s\n", newid)
                lo, hi = h + 1, e - 1     # the record's sequence span, verbatim
                lo <= hi && write(p, view(raw, lo:hi))
                @printf(t, "%s\t%s\t%d\t%d\n", newid, pr.src, pr.start,
                        pr.len === nothing ? 0 : pr.len)
                push!(srcs, pr.src)
                nreads += 1
                s = e
            end
        end
    end
    t1 = time()
    @printf("separated %d reads (from %d source records) in %.2f s\n",
            nreads, length(srcs), t1 - t0)
    @printf("  plain reads  %s  (%.1f MiB)\n", plain, filesize(plain) / 2^20)
    @printf("  provenance   %s  (%.1f MiB)\n", tsv, filesize(tsv) / 2^20)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
