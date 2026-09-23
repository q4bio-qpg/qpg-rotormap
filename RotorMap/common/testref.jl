# =============================================================================
# testref.jl -- synthetic test-data builders (deduplicated from the legacy
# experiment scripts; identical copies used to live in newfasta_v2.jl /
# fastareads_v3.jl, and near-identical ones in indexsample.jl / indexflowreal.jl):
#
#   _data_dir / _v3_data_dir      data directories (ENV-overridable; defaults /tmp)
#   _ensure_file(n)               mutated-reads fasta `newfasta_v2_<n>.fasta`   (2^13 / 2^17 reads)
#   _ensure_file3(n)              same generator, `fastareads_v3_<n>.fasta` naming
#   ensure_data() / ensure_data3()  (small, big) pairs
#   _ensure_test_fasta(path; k)   the sample flow's edge-case reference (N-gaps, CRLF, empty
#                                 record, mid-line '>', all-N record; seed 77)
#   _if_test_fasta_path() / _ensure_test_fasta(; k)
#                                 indexreal's synthetic reference (junk prefix,
#                                 mixed case + IUPAC + N + mid-line '>', CRLF record)
# The v2/v3 generator bodies were byte-identical apart from the directory ENV
# var and the filename prefix; both filename conventions are kept so caches on
# the GPU host stay valid. ENV names unchanged (NEWFASTA_V2_DIR/NEWFASTA_V3_DIR).
# REQUIRES: common/dna.jl (generate_reference, generate_reads, save_fasta).
# =============================================================================
using Random: MersenneTwister

const _SMALL_READS = 2^13  #  8192 reads -> 164 MB, whole-file comparisons legal
const _BIG_READS = 2^17    # 131072 reads -> ~2.6 GB, EXCEEDS the Int32 offset
                           # limit whole-file: benches must batch
const _READ_LEN = 20_000

const _V3_SMALL_READS = 2^13  #  8192 reads -> 164 MB
const _V3_BIG_READS = 2^17    # 131072 reads -> ~2.6 GB (streams in ONE pass)
const _V3_READ_LEN = 20_000

_data_dir() = get(ENV, "NEWFASTA_V2_DIR", "/tmp")
_v3_data_dir() = get(ENV, "NEWFASTA_V3_DIR", "/tmp")

# shared generator body (was duplicated verbatim as _ensure_file / _ensure_file3)
function _gen_mutated_reads_fasta(path::String, n_reads::Int, read_len::Int)
    isfile(path) && filesize(path) > n_reads * 19_000 && return path
    @info "Generating $(n_reads) mutated reads of ~$(read_len) bp -> $path"
    mkpath(dirname(path))
    ref = generate_reference(2^22; seed = 1234)
    reads, pos = generate_reads(ref, read_len, n_reads; err = 0.02, seed = 42)
    heads = [">read_$i pos=$(pos[i])" for i in eachindex(reads)]
    t = @elapsed save_fasta(reads, path, heads = heads)
    @info "Wrote $path ($(filesize(path)) bytes) in $(round(t, digits = 1)) s"
    return path
end

function _ensure_file(n_reads::Int)
    path = joinpath(_data_dir(), "newfasta_v2_$(n_reads).fasta")
    _gen_mutated_reads_fasta(path, n_reads, _READ_LEN)
end

function _ensure_file3(n_reads::Int)
    path = joinpath(_v3_data_dir(), "fastareads_v3_$(n_reads).fasta")
    _gen_mutated_reads_fasta(path, n_reads, _V3_READ_LEN)
end

ensure_data() = (_ensure_file(_SMALL_READS), _ensure_file(_BIG_READS))
ensure_data3() = (_ensure_file3(_V3_SMALL_READS), _ensure_file3(_V3_BIG_READS))

# -----------------------------------------------------------------------------
# the sample flow's edge-case reference: a small deterministic file sized around a
# test k of 50 -- including an N-gap record and an all-N record for the N-free
# pool tests. (legacy indexsample.jl, verbatim)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# indexreal's synthetic reference: deterministic edge-case content (junk
# prefix; chrA 3k+27 mixed case + IUPAC + N + a mid-line '>' junk byte, 70-col
# lines; chrB exactly k, ONE line; chrC k+1; mt too short; empty record;
# chrD CRLF). (legacy indexflowreal.jl, verbatim)
# -----------------------------------------------------------------------------
_if_test_fasta_path() = joinpath(_v3_data_dir(), "indexreal_ref.fasta")

function _ensure_test_fasta(; k::Int = 20_000)
    path = _if_test_fasta_path()
    isfile(path) && filesize(path) > 6 * k && return path
    @info "Generating the synthetic reference fasta -> $path"
    rs = MersenneTwister(77)
    bases = "ACGT"
    iupac = collect("NRYSWKMBDHV")
    function randseq(len)
        String([bases[rand(rs, 1:4)] for _ in 1:len])
    end
    function sprinkle(s) # lowercase + N + IUPAC junk at fixed strides
        v = collect(s)
        for i in 1:97:length(v)
            v[i] = lowercase(v[i])
        end
        for i in 41:211:length(v)
            v[i] = rand(rs, iupac)
        end
        return String(v)
    end
    function wrapped(s, width, eol = "\n")
        join((s[i:min(i + width, end)] for i in 1:width:length(s)), eol)
    end
    seqA = sprinkle(randseq(3k + 27))
    seqA = seqA[1:2k] * ">" * seqA[2k+1:end] # a mid-line '>' (junk -> G)
    seqB = randseq(k)
    seqC = randseq(k + 1)
    seqMt = randseq(k - 1)
    seqD = sprinkle(randseq(2k + 5))
    open(path, "w") do io
        write(io, "junk bytes before the first record\nACGTACGT\n")
        write(io, ">chrA synthetic mixed-case\n", wrapped(seqA, 70))
        write(io, "\n>chrB exactly_k\n", seqB, "\n")
        write(io, ">chrC k_plus_1\n", seqC, "\n")
        write(io, ">mt too_short\n", seqMt, "\n")
        write(io, ">empty\n")
        write(io, ">chrD crlf_record\n", replace(wrapped(seqD, 60), "\n" => "\r\n"))
    end
    return path
end
