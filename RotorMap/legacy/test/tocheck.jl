# using Revise

# using Base.Threads
# using LinearAlgebra
# using CUDA
# using ProgressMeter
using RotorMap
# using RotorMap.SplitComplexMatrices
using RotorMap.Utils
# using RotorMap.RopeEncoders
# using RotorMap.RopeScores
# using RotorMap.RopeIndexers
# using RotorMap.Mapper

# using NPZ
using StringDistances

refname = "/lustre/scratch127/qpg/mc47/rope_mapper/data/GCA_000001405.29_GRCh38.p14_genomic.fna"
# refname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.fna"
refseqs, refheads = load_fasta_mmap(refname, true)

k=20_000

# readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated5er/simulated-human.2X_5_all.fq"
# readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated10er/simulated-human.2X_10_all.fq"
# readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated5er/simulated-maize.2X_5_all.fq"
# readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated10er/simulated-maize.2X_10_all.fq"
# readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated15er/simulated-maize.2X_15_all.fq"
readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated15er/simulated-human.2X_15_all.fq"

seqs, heads = load_fastq_ignore_quality_fixed(readsname, true, trim=k, skipN = true)
seqs_rev = [UInt8(3).-reverse(a) for a in seqs]
seqs2 = deepcopy(seqs)
append!(seqs2, seqs_rev)

locs, vals = load("locs_vals_human_15.bin")
tocheck = load("tocheck_human_15.bin")

n = seqs |> length
ld = Levenshtein()

function ld_batch(N)
    no_match = Int[]

    ldna = deepcopy(seqs2[1])
    rdna = deepcopy(ldna)

    for it in 1:N
        i = tocheck[it]
        if vals[i] > vals[i+n]
            p, l = locs[i]
            # @show p, l, "+"
            ldna = refseqs[p][l:l+19999]    
            rdna = seqs2[i]
        else
            p, l = locs[i+n]
            # @show p, l, "-"
            ldna = refseqs[p][l:l+19999]    
            rdna = seqs2[i+n]
        end
        dist = ld(ldna, rdna)
        # @show dist
        if dist > 6000
            push!(no_match, i)
        end
    end
    save(no_match, "no_match_human_15.bin")
    # @show no_match
end
ld_batch(length(tocheck))

