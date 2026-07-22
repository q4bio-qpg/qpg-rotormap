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
    refname = "/lustre/scratch127/qpg/mc47/rope_mapper/data/GCA_000001405.29_GRCh38.p14_genomic.fna"
    # refname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.fna"
    # refname = "ref_human.fna"
    # @time refseqs, refheads = load_fasta_mmap(refname, true) # randomized N 
    @time refseqs, refheads = load_fasta_mmap(refname, true)
    CUDA.@time dnas = jagged_array(refseqs, refheads)

    normalize = 0

    re = RopeEncoder(k=20_000, s=8, m=4, c=4)
    stp = re.k÷16

    CUDA.@time cindex, norms, all_starts, all_stops, find_ind, ind_starts = index_cuda_final(re, dnas, normalize=normalize, stp=stp)

    @show size(cindex)
    @show size(norms)
    @show size(find_ind)

    @info "index done"

    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/data/SRR29497336_trimmed_20k.fastq" 
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/simulated-maize.30X_all_filtered_20k.fasta"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/simulated-maize.30X_all_filtered_20k_minus_dropped.fasta"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/simulated-human.2X_15_all_trimmed_minus_dropped.fasta"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concated/simulated-human.2X_15_all.fq"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated5er/simulated-human.2X_5_all.fq"
    readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated15er/simulated-human.2X_15_all.fq"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated15er/simulated-maize.2X_15_all.fq"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated5er/simulated-maize.2X_5_all.fq"
    # readsname = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated10er/simulated-maize.2X_10_all.fq"
    # readsname = "reads_human.fastq"
    seqs, heads = load_fastq_ignore_quality_fixed(readsname, true, trim=re.k, skipN = true)    
    # seqs, heads = load_fasta_mmap_fixed(readsname, true, trim=re.k, skipN = true)
    n = seqs |> length

    seqs_rev = [UInt8(3).-reverse(a) for a in seqs]
    seqs2 = deepcopy(seqs)
    append!(seqs2, seqs_rev)
    # seqs2=seqs
    jreads = jagged_array(seqs2)
    
    CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
    # CUDA.@time cropes_inv, _ = encode_batch_cuda_all_best(re, jreads_inv.vect, starts=cu(jreads_inv.starts), stops=cu(jreads_inv.stops))

    cindex = reshape(cindex, :, size(cindex, 3))
    cindex_sp = SplitComplexMatrix(ComplexF16.(cindex))

    cropes = reshape(cropes, :, size(cropes, 3))
    cropes_sp = SplitComplexMatrix(ComplexF16.(cropes))

    # cropes = reshape(cropes, :, length(jreads.starts))
    # cropes_inv = reshape(cropes_inv, :, length(jreads_inv.starts))

    
    # cropes_inv = SplitComplexMatrix(ComplexF16.(cropes_inv))

    # cut,_,_ = cutting_threshold_new(re, 0.15, 0.05, n=10000, normalize=normalize, double=double, euc=false)
    # cut = 0.5
    cuts = load("cuts_human_015_rng.bin") |> cu
    # cuts = load("cuts_maize_015_rng.bin") |> cu
    
    # CUDA.@time carts = search_carts(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, cuts = cuts, batch_prod_size=2^12, euc=false)
    # cut = 8f0

    CUDA.@time ranges, locs = search_m1_cuda_split_all_batch(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, cuts = cuts, batch_prod_size=2^12, euc=false)        
    # CUDA.@time ranges2 = search_m1_cuda_split_all_batch(re::RopeEncoder, re.k÷10, cindex, cropes_inv, cut = cut, batch_prod_size=2^12)
    @show size(ranges)
    lr = length.(ranges) 
    # @show lr[1:10]
    # @show lr[end-9:end]
    @show sum(lr)
    # n = length(ranges) ÷ 2
    lrt = lr[1:n] .+ lr[n+1:2n]
    findall(iszero, lrt) |> size
    findall(isone, lrt) |> size

    # v = cropes.Re[:,2197] + im*cropes.Im[:,2197]
    # @show [Utils.dna_letters[l+1] for l in (seqs[2197] |> Array)] |> String

    # for ri in 1:length(ranges)
    for ri in 1:100
        r = ranges[ri]
        if length(r) != 1 
            continue
        end        
        @show ri
        ind = r[1][1]
        loc = r[1][2][1]
        ldna = refseqs[ind][loc:loc+19999]
        @show ld(ldna, seqs[ri])
    end

    CUDA.@time locs, vals = search_batch_max(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, batch_prod_size=2^12)
    # CUDA.@time locs = search_batch_max(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, batch_prod_size=2^12)

    vals = Array(vals)
    save((locs, vals), "locs_vals_maize_15.bin")
    # save((locs, vals), "locs_vals_maize_10.bin")
    # save((locs, vals), "locs_vals_maize_5.bin")
    # save((locs, vals), "locs_vals_human_15.bin")
    # save((locs, vals), "locs_vals_human_10.bin")
    # locs, vals = load("locs_vals_human_10.bin")

    n = seqs |> length
    ld = Levenshtein()
    function ld_batch(N)
        # @threads for i in 1:N
        for it in 1:N
            i = tocheck[it]
            # @show i
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
            # ldna = refseqs[p][l:l+19999]
            @show i, ld(ldna, rdna)
        end
    end
    @time ld_batch(10)

    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated15er/maize_d2_a85_read_true_positions_correct.json"
    readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated15er/human_d2_a85_read_true_positions_correct.json"
    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/maize_d2_a85_read_true_positions_correct.json"
    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated5er/human_d2_a95_read_true_positions_correct.json"
    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/human/concatenated10er/human_d2_a90_read_true_positions_correct.json"
    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated5er/maize_d2_a95_read_true_positions_correct.json"
    # readsinfo = "/lustre/scratch127/qpg/mc47/rope_mapper/generated_data/concatenated10er/maize_d2_a90_read_true_positions_correct.json"
    data = open(f -> JSON3.read(f, Dict{String, NamedTuple}), readsinfo)

    tocheck = Int[]
    function test_batch(N)
        for i = 1:N
            rname = heads[i][2:end]
            rtrueposindex = data[rname].ref_index + 1
            rtrueposloc = data[rname].pos + 1
            # rdir = data[rname].reverse            
            fdir = vals[i+n] > vals[i]
            if !fdir
                p, l = locs[i]    
            else
                p, l = locs[i+n]
            end 
            # if (rdir!=fdir) || (p-rtrueposindex !=0)
            if (p-rtrueposindex !=0)
                push!(tocheck, i)
            elseif abs(l-rtrueposloc) >= re.k 
                push!(tocheck, i)
            end 
        end
    end
    test_batch(n)
    save(tocheck, "tocheck_maize_15.bin")


    tocheck = Int[]
    function test_batch(N)
        for i = 1:N
            rname = heads[i][2:end]
            rtrueposindex = data[rname].ref_index + 1
            rtrueposloc = data[rname].pos + 1
            # rdir = data[rname].reverse            
            fnd = false
            for (p, l) in locs[i]
                if (p-rtrueposindex ==0) && abs(l-rtrueposloc) < re.k 
                    fnd = true                    
                end 
            end
            if !fnd 
                push!(tocheck, i)
            end
        end
    end
    test_batch(n)    
    save(tocheck, "tocheck_human_15_full_neg.bin")


    tocheck2 = Int[]
    function test_batch2()
        for i in tocheck
            rname = heads[i][2:end]
            rtrueposindex = data[rname].ref_index + 1
            rtrueposloc = data[rname].pos + 1
            # rdir = data[rname].reverse            
            fnd = false
            for (p, l) in locs[i]
                if (p-rtrueposindex ==0) && abs(l-rtrueposloc) < re.k 
                    fnd = true                    
                end 
            end
            if !fnd 
                push!(tocheck2, i)
            end
        end
    end
    test_batch2()  

    # save(tocheck, "tocheck_human_15.bin")
    # save(tocheck, "tocheck_maize_10.bin")
    # save(tocheck, "tocheck_maize_5.bin")
    # save(tocheck, "tocheck_human_10.bin")

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