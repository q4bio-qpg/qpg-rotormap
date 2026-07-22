using Revise

using Base.Threads
using LinearAlgebra
using CUDA
using ProgressMeter
using RotorMap
using RotorMap.SplitComplexMatrices
using RotorMap.Utils
using RotorMap.RopeEncoders
using RotorMap.RopeScores
using RotorMap.RopeIndexers
using RotorMap.Mapper

using NPZ
using StringDistances

# test of the sliding kernel
function test()
    # refname = "/lustre/scratch127/qpg/mc47/rope_mapper/data/GCA_000001405.29_GRCh38.p14_genomic.fna"
    # # refname = "ref_human.fna"
    # @time refseqs, refheads = load_fasta_mmap(refname)
    # CUDA.@time dnas = jagged_array(refseqs, refheads)

    # normalize = 0

    # re = RopeEncoder(k=20_000, s=8, m=4, c=4)
    # CUDA.@time cindex, norms, all_starts, all_stops, find_ind, ind_starts = index_cuda_final(re, dnas, normalize=normalize, stp=re.k÷20)

    # @show size(cindex)
    # @show size(norms)
    # @show size(find_ind)
    @info "Loading index"
    CUDA.@time begin 
        re, cindex_sp, find_ind, ind_starts = load("/lustre/scratch127/qpg/dy4/rotormap/newindex.bin")
        # cuts = load("/lustre/scratch127/qpg/dy4/rotormap/dev/cuts_human_015.bin") |> cu
        cuts = load("/lustre/scratch127/qpg/dy4/rotormap/cuts.bin") |> cu
    end
    @info "index loaded"

    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/data/SRR29497336_trimmed_20k.fastq" 
    readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/simulated-human.2X_15_all_trimmed_minus_dropped.fasta"
    # readsname = "reads_human.fastq"
    # seqs, heads = load_fastq_ignore_quality(readsname)
    seqs, heads = load_fasta_mmap(readsname)

    seqs_rev = [UInt8(3).-reverse(a) for a in seqs]
    seqs2 = deepcopy(seqs)
    append!(seqs2, seqs_rev)
    # seqs2=seqs
    jreads = jagged_array(seqs2)
    
    CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
    cropes = reshape(cropes, :, length(jreads.starts))
    cropes_sp = SplitComplexMatrix(ComplexF16.(cropes))

    # cuts .*= 0.8

    CUDA.@time ranges, locs = search_m1_cuda_split_all_batch(re::RopeEncoder, re.k÷20, cindex_sp, cropes_sp, find_ind, ind_starts, cuts = cuts, batch_prod_size=2^12, euc=false)
    # CUDA.@time ranges2 = search_m1_cuda_split_all_batch(re::RopeEncoder, re.k÷10, cindex, cropes_inv, cut = cut, batch_prod_size=2^12)
    @show size(ranges)
    lr = length.(ranges) 
    # @show lr[1:10]
    # @show lr[end-9:end]
    @show sum(lr)
    n = length(heads)
    # n = length(ranges)
    lrt = lr[1:n] .+ lr[n+1:2n]
    findall(iszero, lrt) |> size
    findall(isone, lrt) |> size

    # v = cropes.Re[:,2197] + im*cropes.Im[:,2197]
    # @show [Utils.dna_letters[l+1] for l in (seqs[2197] |> Array)] |> String


    # jreads2 = jagged_array([UInt8(3).-reverse(a) for a in seqs], heads)
    # CUDA.@time cropes2, _ = encode_batch_cuda_all_best(re, jreads2.vect, starts=cu(jreads2.starts), stops=cu(jreads2.stops))
    # cropes2 = reshape(cropes2, :, length(jreads2.starts))
    # cropes2_sp = SplitComplexMatrix(ComplexF16.(cropes2))
    # CUDA.@time ranges2 = search_m1_cuda_split_all_batch(re::RopeEncoder, re.k÷10, cindex_sp, cropes2_sp, cut = cut, batch_prod_size=2^12)

    # @show size(ranges2)
    # lr2 = length.(ranges2)
    # # @show lr[1:10]
    # # @show lr[end-9:end]
    # @show sum(lr2)

    # @show findall(iszero, lr.+lr2) |> size



    # ss1=dnas.vect[dnas.starts[44]+51396000:dnas.starts[44]+51396000+20000-1]
    # ss2 = jreads2.vect[jreads2.starts[17]:jreads2.stops[17]]
    # jtest = jagged_array(Array.([ss1,ss2]))

    # CUDA.@time cropestest, _ = encode_batch_cuda_all_best(re, jtest.vect, starts=cu(jtest.starts), stops=cu(jtest.stops))

    # v1 = reshape(cropestest[:,:,1], 256)
    # v2 = reshape(cropestest[:,:,2], 256)

    # v1'*v2 |> abs2 # 14.160427f0 --- Why not found ??     
    # # cut 4.9153113f0
end 