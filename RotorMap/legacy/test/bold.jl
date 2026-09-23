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

using Parquet2, Tables, DataFrames
using Printf

using Statistics

function bold_load(filename="BOLD_Public.10-Apr-2026.parquet")
    ds = Parquet2.Dataset(filename)

    df_lazy = DataFrame(ds)[!, [:bin_uri, :nuc, :processid]] 

    df = DataFrame(
        bin_uri = Vector(df_lazy.bin_uri),
        nuc = Vector(df_lazy.nuc),
        processid = Vector(df_lazy.processid)
    )

    @info "All rows:"
    @show nrow(df)
    
    dropmissing!(df, [:bin_uri, :nuc])
    @info "Rows without missing:"
    @show nrow(df)

    bins = combine(groupby(df, :bin_uri), :nuc => (x -> [Vector(x)]) => :sequences)
    nbins = nrow(bins)
    @info "Number of BINs:"
    @show nbins

    return df, bins
end

function bold_stats(bins)
    nbins = nrow(bins)

    bin_uris = bins.bin_uri
    sequences_arrays = bins.sequences

    avg_lens = zeros(nbins)
    for i in 1:length(bin_uris)
        current_bin = bin_uris[i]
        current_seqs = sequences_arrays[i]

        lengths = length.(current_seqs)        
        avg_length = sum(lengths) / length(lengths)

        avg_lens[i] = avg_length
    end
    
    begin 
        data = sort(avg_lens)
        probabilities = 0:0.01:1
        percentile_values = quantile(data, probabilities)
        for (p, val) in zip(0:100, percentile_values)
            println("$p%: $val")
        end
    end
end

function consensus_all_bins(bins)
    # load cached:
    # bin_uris, seqs_bin = load("data/bin_seq_cons.bin")
    # nbins = bin_uris |> length

    barlen = 600 

    bin_uris = bins.bin_uri
    sequences_arrays = bins.sequences

    GGs = rpad("", barlen, 'G')
    sequences_cons = fill(GGs, nbins)

    @threads for i in 1:length(bin_uris)
        bin_i = bin_uris[i]
        seqs_i = sequences_arrays[i]

        sequences_cons[i] = get_consensus_sequence(seqs_i)
    end

    return sequences_cons
end

# Function to calculate the consensus sequence
function get_consensus_sequence(sequences::Vector{String}; barlen=600)
    if isempty(sequences)
        return ""
    end

    # 1. Handle unequal lengths (Pad shorter sequences with gaps '-')
    # Note: For this to be biologically accurate, sequences should be ALIGNED first.
    # max_len = maximum(length.(sequences))
    # if max_len < barlen 
    #     return "" 
    # end
    max_len = barlen 
    
    padded_seqs = [rpad(seq, max_len, '-') for seq in sequences]

    consensus_array = Char[]

    # 2. Iterate through each position (column by column)
    for i in 1:max_len
        # Dictionary to keep track of character counts in this column
        counts = Dict{Char, Int}()
        
        for seq in padded_seqs
            base = seq[i]
            # Increment the count for this base
            counts[base] = get(counts, base, 0) + 1
        end
        
        # 3. Find the most frequent character in this column
        most_frequent_base = 'G'
        max_count = -1
        
        for (base, count) in counts
            if base != '-' && base != 'N' && count > max_count
                max_count = count
                most_frequent_base = base
            end
        end
        
        push!(consensus_array, most_frequent_base)
    end

    # Convert the array of characters back into a single string
    return String(consensus_array)
end

const DNA_LOOKUP = zeros(UInt8, 256)
DNA_LOOKUP[Int('A')] = 0
DNA_LOOKUP[Int('C')] = 1
DNA_LOOKUP[Int('G')] = 2
DNA_LOOKUP[Int('T')] = 3

function seq_to_bytes!(seq::String, bytes::Vector{UInt8}; barlen=600)
    total = 0
    for b in codeunits(seq)
        if b == 0x2d continue end # skip '-'
        
        total += 1
        bytes[total] = DNA_LOOKUP[b]

        if total == barlen break end # trim
    end
    # no padding needed as bytes are filled with GGs
end

"""
Constructs the index of RoPE encodings for each BIN 
"""
function bins_rope_index(re::RopeEncoder, bins; save::String="")::SplitComplexMatrix
    bin_uris = bins.bin_uri
    sequences_arrays = bins.sequences

    seqs_bin = fill(fill(UInt8(2), barlen), nbins)
    @showprogress @threads for i = 1:nbins
        for seq in sequences_arrays[i]
            seq_to_bytes!(seq, seqs_bin[i])
        end        
    end

    CUDA.@time dnas = jagged_array(seqs_bin, bin_uris)
end

"""
Prepares all sequences in BOLD for encoding
"""
function prepare_all(df; barlen = 600)
    n = nrow(df)
    allseqs = [fill(UInt8(2), barlen) for _ in 1:n]

    @showprogress @threads for i = 1:n 
        seq = df.nuc[i]
        seq_to_bytes!(seq, allseqs[i])
    end

    return allseqs
end

"""
Encodes prepared sequences in batches
"""
function encode_all(re::RopeEncoder, df, bins, allseqs; batchsize = 4*10^5)
    n = nrow(df) # length(allseqs) 

    nbins = nrow(bins)
    bindex = CUDA.zeros(ComplexF16, dim(re), nbins)

    bintonum =  Dict(zip(bins.bin_uri, 1:nbins))

    stp = length(allseqs[1]) # barlen

    @showprogress for i=1:batchsize:n
        stop = min(i + batchsize - 1, n)    
        @views dnas = jagged_array(allseqs[i:stop])
        cindex, norms, _, _, _, _ = index_cuda_final(re, dnas, normalize=0, stp=stp)
        cindex = reshape(cindex, :, size(cindex, 3))
        for j = i:stop 
            binstr = df.bin_uri[j]
            binum = bintonum[binstr]
            @views bindex[:, binum] .+= cindex[:, j-i+1]
        end        
    end
    
    @showprogress @threads for i=1:nbins
        @views nrm = norm(bindex[:, i])+1f-16
        # @views bindex[:, i] ./= length(bins.sequences[i]) 
        @views bindex[:, i] ./= nrm
    end

    return bindex
end

function gen_query(bins; barlen = 600, err = 0.10)
    nbins = nrow(bins) 

    query = [fill(UInt8(2), barlen) for _ in 1:nbins]
    @showprogress @threads for i = 1:nbins 
        seq = rand(bins.sequences[i])
        seq_to_bytes!(seq, query[i])
        query[i] = mutate(query[i], err, mll = 2*barlen)
    end

    return query
end
    
# function search1(re::RopeEncoder, query, cindex_sp; nbatch = 2^16, ktopk = 10) 
#     nbins = length(query)
    
#     res = zeros(Int32, ktopk)
#     res_i = zeros(Int32, ktopk)
#     c=0
#     CUDA.@time for k=1:nbatch:nbins        
#         c+=1
#         @show k
#         @views jreads = jagged_array(query[k:min(k-1+nbatch,nbins)])
#         cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
#         cropes = reshape(cropes, :, size(cropes, 3))
#         cropes_sp = SplitComplexMatrix(ComplexF16.(cropes))

#         locs, vals = search_batch_topk(cindex_sp, cropes_sp, k=ktopk, batch_prod_size=2^12)
#         # res[c] = [locs[i][1]-k+1-i!=0 ? 1 : 0 for i=1:size(cropes,2)] |> sum
#         res_i *= 0 
#         for i=1:size(cropes,2)
#             for j=1:ktopk
#                 if locs[i,j]-k+1-i==0
#                     res_i[j] += 1                    
#                 end
#             end
#         end
#         res.+= res_i
#     end

#     return res 
# end

function search(re::RopeEncoder, query, cindex_sp; nbatch = 2^16, ktopk = 10) 
    nbins = length(query)

    @views jreads = jagged_array(query)
    cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
    cropes = reshape(cropes, :, size(cropes, 3)) 
    cropes = ComplexF16.(cropes)
    
    res = zeros(Int32, ktopk)
    res_i = zeros(Int32, ktopk)
    found = zeros(Bool, nbins)
    c=0
    CUDA.@time for k=1:nbatch:nbins        
        c+=1
        @show k
        @views cropes_sp = SplitComplexMatrix(cropes[:, k:min(k-1+nbatch,nbins)])

        CUDA.@time locs, vals = search_batch_topk(cindex_sp, cropes_sp, k=ktopk, batch_prod_size=2^12)
        # res[c] = [locs[i][1]-k+1-i!=0 ? 1 : 0 for i=1:size(cropes,2)] |> sum
        res_i *= 0 
        for i=1:size(cropes_sp.Re,2)
            for j=1:ktopk
                if locs[i,j]-k+1-i==0
                    res_i[j] += 1                    
                    found[k+i-1] = true
                end
            end
        end
        res.+= res_i
    end

    return res, found 
end

function test()

    barlen = 600 

    df, bins = bold_load("data/BOLD_Public.10-Apr-2026.parquet")
    allseqs = prepare_all(df)
    
    re = RopeEncoder(k=barlen, s=4, m=1, c=4)
    # bindex = encode_all(re, df, bins, allseqs)
    bindex = load("data/bindex.bin") |> CuArray
    cindex_sp = SplitComplexMatrix(ComplexF16.(bindex))

    found = ones(Bool, nbins)
    for _ in 1:10
        query = gen_query(bins)
        res, f = search(re, query, cindex_sp)
        found .&= f 
    end

    @show sum(found)/nbins


        
    # res = search(re, query, cindex_sp)

    # sres = [sum(res[1:i])/nbins for i=1:ktopk]
    for i in 1:length(sres)
        @printf("top-%d: %.2f%%\n", i, sres[i] * 100)
    end


    CUDA.@time dnas = jagged_array(seqs_bin, bin_uris)

    normalize = 0

    re = RopeEncoder(k=barlen, s=4, m=1, c=4)
    stp = re.k # one rope per seq

    CUDA.@time cindex, norms, all_starts, all_stops, find_ind, ind_starts = index_cuda_final(re, dnas, normalize=0, stp=stp)
    cindex = reshape(cindex, :, size(cindex, 3))
    cindex_sp = SplitComplexMatrix(ComplexF16.(cindex))
    @info "index done"

    seqs_mut = deepcopy(seqs_bin)
    @threads for i=1:nbins
        seqs_mut[i] .= mutate(seqs_bin[i], 0.10, mll = 2*barlen)
    end

    nbatch = 2^16
    ktopk = 10
    # res = [0 for k=1:nbatch:nbins]
    res = zeros(Int32, ktopk)
    res_i = zeros(Int32, ktopk)
    c=0
    CUDA.@time for k=1:nbatch:nbins        
        c+=1
        @show k
        jreads = jagged_array(seqs_mut[k:min(k-1+nbatch,nbins)], bin_uris[k:min(k-1+nbatch,nbins)])
        cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
        cropes = reshape(cropes, :, size(cropes, 3))
        cropes_sp = SplitComplexMatrix(ComplexF16.(cropes))

        # CUDA.@time locs, vals = search_batch_max(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, batch_prod_size=2^12)
        # CUDA.@time 
        locs, vals = search_batch_topk(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, k=ktopk, batch_prod_size=2^12)
        # res[c] = [locs[i][1]-k+1-i!=0 ? 1 : 0 for i=1:size(cropes,2)] |> sum
        res_i *= 0 
        for i=1:size(cropes,2)
            for j=1:ktopk
                if locs[i,j]-k+1-i==0
                    res_i[j] += 1                    
                end
            end
        end
        res.+= res_i
    end
    
    sres = [sum(res[1:i])/nbins for i=1:ktopk]
    for i in 1:length(sres)
        @printf("top-%d: %.2f%%\n", i, sres[i] * 100)
    end

    # CUDA.@time cropes_inv, _ = encode_batch_cuda_all_best(re, jreads_inv.vect, starts=cu(jreads_inv.starts), stops=cu(jreads_inv.stops))


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

    # CUDA.@time locs, vals = search_batch_max(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, batch_prod_size=2^12)
    CUDA.@time locs, vals = search_batch_topk(re::RopeEncoder, stp, cindex_sp, cropes_sp, find_ind, ind_starts, k=10, batch_prod_size=2^12)
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

