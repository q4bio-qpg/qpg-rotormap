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
    refname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.fna"
    # refname = "ref_human.fna"
    @time refseqs, refheads = load_fasta_mmap(refname, true) # randomize 
    # @time refseqs, refheads = load_fasta_mmap(refname, false)
    CUDA.@time dnas = jagged_array(refseqs, refheads)

    normalize = 0

    re = RopeEncoder(k=20_000, s=8, m=4, c=4)
    stp = re.k÷16

    CUDA.@time cindex, norms, all_starts, all_stops, find_ind, ind_starts = index_cuda_final(re, dnas, normalize=normalize, stp=stp)

    ni = size(cindex, 3)
    cuts = 100*ones(Float32, ni) |> cu
    
    dnas_mut_shift = deepcopy(dnas.vect) |> Array
    # @showprogress @threads 
    for j = 1:length(dnas.starts)
        if  dnas.stops[j] - dnas.starts[j] + 1 < re.k # the last one №709 in human is 16568 < 20000
            continue 
        end
        @views dnas_mut_shift[dnas.starts[j]:dnas.stops[j]] .= mutate_shift(dnas_mut_shift[dnas.starts[j]:dnas.stops[j]], 0.05, shift = stp÷2)
    end

    for N=1:8
        @show N 
        
        dnas_mut = deepcopy(dnas_mut_shift)

        # @showprogress @threads 
        for j = 1:10000:length(dnas_mut)-10000
            @views dnas_mut[j:j+10000-1] .= mutate(dnas_mut[j:j+10000-1], 0.15, mll=10000)
        end

        cindex2 = CUDA.zeros(ComplexF32, re.m, 4^re.c, ni)
        norms2 = CUDA.zeros(Float32, re.m, ni)

        CUDA.@time encode_batch_cuda_all_best!(cindex2, norms2, re, CuArray(dnas_mut), starts = all_starts[1:end], stops = all_stops[1:end], normalize=normalize)
        CUDA.@time fids = sum(conj.(cindex) .* cindex2, dims=(1,2)) .|> abs2 |> vec #|> Array #|> vec
        CUDA.@time cuts .= min.(cuts, fids)
    end

    save(cuts, "cuts_maize_015_rng.bin")
    # save...
end 