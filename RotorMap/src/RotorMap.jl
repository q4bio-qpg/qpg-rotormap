# __precompile__(true)

module RotorMap

export main

include(joinpath(@__DIR__,"TreeArrays.jl"))
include(joinpath(@__DIR__,"SplitComplexMatrices.jl"))
include(joinpath(@__DIR__,"Utils.jl"))
include(joinpath(@__DIR__,"RopeEncoders.jl"))
include(joinpath(@__DIR__,"RopeScores.jl"))
include(joinpath(@__DIR__,"RopeIndexers.jl"))
include(joinpath(@__DIR__,"Mapper.jl"))

# using Reexport 

# using Revise
using ArgParse
using ProgressMeter
using CUDA
# using Profile
# using Random
using Base.Threads
using .RopeEncoders
using .RopeIndexers
using .RopeScores
using .Mapper
using .Utils
using .SplitComplexMatrices


# @reexport 

# struct GPUIndexCache 
#     re::RopeEncoder
#     step::Int64
#     n::Int64
#     e::CuVector
#     cindex::CuArray
#     dna_ref::CuArray
# end 

# const server_cache = Ref{GPUIndexCache}()
const server_cache = Ref{RopeIndexer}()

using PrecompileTools: @setup_workload, @compile_workload 

@setup_workload begin
    include(joinpath(@__DIR__,"../test/full.jl"))
    @compile_workload begin 
        # "A small work needed for precompilation has started: " |> println
        # full(; k=5, N=10, step=4, s=1, n=1, err=0.1, show_trace=false)
    end    
end

function parse_commandline(args::Vector{String}=ARGS, server=false)
    s = ArgParseSettings(description = "RotorMap: A tool for DNA mapping",
                    version = "Version 0.1.5",
                    add_version = true) 

    function my_exc_handler(s::ArgParseSettings, err)
        # println(stderr, err.text)
        println(err.text)    
        println(usage_string(s))
    end
    if server
        s.exit_after_help = false 
        s.exc_handler = my_exc_handler
    end 

    @add_arg_table! s begin
        "index"
            help = "create an index for mapping"
            action = :command
        "search", "map"
            help = "map reads against an index or reference"
            action = :command
        "random", "gen"
            help = "generate DNA data"
            action = :command
        "mutate", "mut"
            help = "mutation routines"
            action = :command
        "convert", "conv"
            help = "convert formats"
            action = :command
        "server", "serve", "serv"
            help = "simple server mode"
            action = :command
    end

    @add_arg_table! s["random"] begin
        "reference", "ref"
            help = "generate reference sequence of length N"
            action = :command
        "reads"
            help = "generate reads from a reference sequence"
            action = :command
    end

    @add_arg_table! s["random"]["reference"] begin
        "N"
            help = "length of the reference"
            arg_type = Int
            required = true
        "-o"
            help = "name of the output file; should be either with .bin or .fastq extension which defines the format"
            arg_type = String
            default = "ref.bin"
            # required = true
        "--seed" 
            help = "seed for the random number generator"
            arg_type = Int
    end

    s["random"]["reads"].description = "Generate reads from a reference sequence."

    @add_arg_table! s["random"]["reads"] begin
        "-k", "--len"
            help = "length of each read"
            arg_type = Int
            required = true
        "-n", "--num"
            help = "number of reads to generate"
            arg_type = Int
            default = 100
        "-i", "--input"
            help = "name of the input reference file; should be either with .bin or .fastq extension"
            arg_type = String
            required = true
        "-f", "--format"
            help = "format of the output file: either julia or fastq; default matches the input file"
            arg_type = String
        "-o", "--output"
            help = "name of the output file; the default is either reads.bin or reads.fastq, depending on the format"
            arg_type = String
        "--error", "--err"
            help = "error rate to introduce in reads; float number"    
            arg_type = Float64
            default = 0.15
        "--seed" 
            help = "seed for the random number generator"
            arg_type = Int
    end

    s["index"].description = "Create an index for a given reference file"
    @add_arg_table! s["index"] begin
        "-i", "--input"
            help = "name of the input reference file; should be either with .bin or .fastq extension"
            arg_type = String
            required = true
        "-k", "--len"
            help = "minimal length of reads expected to map; recommended 100000"
            arg_type = Int
            required = true
        "-o", "--output"
            help = "name of the output file"
            arg_type = String
            default = "index.bin"
        "-s" 
            help = "length of the small s-mer in the Rope Encoder"
            arg_type = Int
            default = 5
        "-m" 
            help = "the multiplicative factor in the Rope Encoder"
            arg_type = Int
            default = 1
        "-c" 
            help = "the compactness factor in the Rope Encoder"
            arg_type = Int
            default = 0
        "--step"
            help = "step with which the sliding window slides; default is 10% of k"
            arg_type = Int
        "--error", "--err"
            help = "expected error in reads"
            arg_type = Float64
            default = 0.15            
    end

    s["search"].description = "Map reads using an index of a reference DNA"
    @add_arg_table! s["search"] begin
        "-r", "--reads"
            help = "name of the reads file"
            arg_type = String
            required = true
        # "--ref"
        #     help = "name of the reference file; cached will be used if not supplied"
        #     arg_type = String
        "-i", "--index"
            help = "name of the index file; if not supplied, in the server mode the cached version will be used" # todo recompute it instead? 
            arg_type = String
        "-C", "--cuts"
            help = "name of the cuts file; if not supplied, in the server mode the cached version will be used" # todo recompute it instead? 
            arg_type = String
        "-s" 
            help = "length of the small s-mer in the Rope Encoder for search Phase 2"
            arg_type = Int
            default = 7
        "-m" 
            help = "the multiplicative factor in the Rope Encoder for search Phase 2"
            arg_type = Int
            default = 4
        "-c" 
            help = "the compactness factor in the Rope Encoder for search Phase 2"
            arg_type = Int
            default = 0            
        "-o", "--output"
            help = "name of the output file of found positions"
            arg_type = String
            default = "pos.csv"
        # "--fp"
        #     help = "precision of floats, FP32 or FP16"
        #     arg_type = Int
        #     default = 32            
    end

    s["convert"].description = "Convert file formats"
    @add_arg_table! s["convert"] begin
        # "-t", "--type"
        #     help = "Type: "
        #     arg_type = String
        "-i", "--input"
            help = "name of the input file"
            arg_type = String
            required = true
        "-o", "--output"
            help = "name of the output file"
            arg_type = String
            required = true
    end

    s["server"].description = "Server mode. In the basic mode it continuously tracks the file args.jl. To supply input in args.jl, write args()=\"arg1 arg2 arg3\" into it."
    @add_arg_table! s["server"] begin
        "mode"
            help = "Only the basic mode is implemented. Write exit to close the server."
            arg_type = String
            default = "basic"
    end

    return parse_args(args, s)
end

function main(; args::Vector{String}=ARGS, server=false)
    # @show (args,server)
    parsed_args = parse_commandline(args,server)

    # println("Parsed args:")
    # @show parsed_args 
    # for (key, val) in parsed_args
    #     println("  $key  =>  $(repr(val))")
    # end

    if parsed_args != nothing # could be nothing if -h is supplied in server mode 
        if parsed_args["%COMMAND%"] == "random"
            if parsed_args["random"]["%COMMAND%"] == "reference"
                N = parsed_args["random"]["reference"]["N"]
                o = parsed_args["random"]["reference"]["o"]
                seed = parsed_args["random"]["reference"]["seed"]

                seed = seed!=nothing ? seed : rand(UInt64)                
                dna_ref = generate_reference(N,seed=seed)

                @info "Reference has been generated. Saving to disk"
                if endswith(o, ".bin")
                    save(([dna_ref], ["> $seed"]), o)
                elseif endswith(o, ".fasta")
                    save_fasta([dna_ref], o, heads=["> $seed"])
                else
                    error("wrong file format; should be either .bin or .fastq")
                end
                @info "Reference has been written into $o"
                if !server 
                    exit() 
                end 
            elseif parsed_args["random"]["%COMMAND%"] == "reads"
                k = parsed_args["random"]["reads"]["len"]
                n = parsed_args["random"]["reads"]["num"]
                i = parsed_args["random"]["reads"]["input"]
                f = parsed_args["random"]["reads"]["format"]
                o = parsed_args["random"]["reads"]["output"]
                err = parsed_args["random"]["reads"]["error"]
                seed = parsed_args["random"]["reads"]["seed"]
                
                ia = rsplit(i, ".", limit=2)
                refs_name = ia[1]
                refs_ext = ia[2]

                # todo: multiple dna_refs
                if refs_ext == "bin"
                    dna_refs, heads = load(i) 
                elseif refs_ext == "fasta"
                    dna_refs, heads = load_fasta(i)
                end

                # set reads format 
                if f != nothing
                    reads_ext = f
                else
                    reads_ext = refs_ext
                end

                if o!=nothing 
                    oa = rsplit(o, ".", limit=2)
                    reads_name = oa[1]
                    reads_ext = oa[2]
                else
                    reads_name = "reads"
                end

                reads_fn = reads_name * "." * reads_ext
                pos_fn = reads_name * "_pos.csv"                

                @info "Generating reads"
                @time reads, pos = generate_reads(dna_refs[1], k, n, err=err, seed=seed) # todo: use all refs
                @info "Reads have been generated. Saving to disk" 
                # save((reads,locs), o)            

                # save positions 
                open(pos_fn, "w") do io 
                    for p in pos
                        println(io, p)
                    end
                end

                # save reads
                if reads_ext == "bin"
                    save((reads, ["> " * repr(pos[i]) for i in eachindex(pos)]), reads_fn) 
                elseif reads_ext == "fasta"
                    save_fasta(reads, reads_fn, pos=pos)
                # elseif reads_ext == "fastq"
                end

                @info "Reads have been written into $reads_fn, their locations to $pos_fn" 
                if !server 
                    exit() 
                end 
            end
        elseif parsed_args["%COMMAND%"] == "index"
            i = parsed_args["index"]["input"]
            o = parsed_args["index"]["output"]
            k = parsed_args["index"]["len"]        
            s = parsed_args["index"]["s"]
            m = parsed_args["index"]["m"]
            c = parsed_args["index"]["c"]
            stp = parsed_args["index"]["step"]
            err = parsed_args["index"]["error"]


            stp = (stp==nothing) ? k÷10 : stp
            
            @info "Loading reference" 
            # dna_ref = load(i)
            dna_refs, heads = load_dnas(i)

            dna_ref = dna_refs[1] # todo: multiple refs

            re = RopeEncoder(k=k, s=s, m=m, c=c)
            ri = RopeIndexer(dna_ref; re=re, step=stp, err=err)
            
            @info "Constructing the index on GPU"     
            CUDA.@time index_cuda_best_new!(ri)               

            # @info "Transferring it to CPU" 
            # copyto!(ri.index, cindex) 
            @info "Saving the index to $o"

            server_cache[] = ri
            save(ri, o) 

            # caching 
            # cindex_fp16 = ComplexF16.(cindex)
            # e_fp16 = ComplexF16.(e)
            # server_cache[] = GPUIndexCache(ri.re, stp, ri.n, e_fp16, cindex_fp16)
            # server_cache[] = GPUIndexCache(ri.re, stp, ri.n, e, cindex, cu_dna_ref)
            
            @info "The index has been saved and cached"

            if !server 
                exit() 
            end 

            # todo: should also find the property of the RopeEncoder used, the cutting 30% err fidelity. 
            # this cutting fidelity should be approximated by both 30% ordinary and 10% shift mutation

        elseif parsed_args["%COMMAND%"] == "search"
            r = parsed_args["search"]["reads"]
            i = parsed_args["search"]["index"]
            cuts = parsed_args["search"]["cuts"]
            s = parsed_args["search"]["s"]
            m = parsed_args["search"]["m"]
            c = parsed_args["search"]["c"]
            o = parsed_args["search"]["output"]   

            skip_search = false
                        
            # if i == nothing 
            #     # try to use cache
            #     try 
            #         # re = server_cache[].re
            #         # stp = server_cache[].step
            #         # n = server_cache[].n
            #         # e = server_cache[].e
            #         # cindex = server_cache[].cindex
            #         # cu_dna_ref = server_cache[].dna_ref
            #         ri = server_cache[]
            #         @info "Using the cached index"
            #     catch
            #         @warn "No index in cache. Try to supply it from a file using -i option"
            #         @info "Skipping search" 
            #         skip_search = true
            #     end
            # else
            #     # @info "Loading reference"   
            #     # dna_refs, heads = load_dnas(ref)
            #     # dna_ref = dna_refs[1] # todo: multiple refs
            #     # cu_dna_ref = CuArray(dna_ref)
                
            #     @info "Loading index"   
            #     @time ri = load(i)

            #     # caching
            #     # cindex_fp16 = ComplexF16.(cindex)
            #     # e_fp16 = ComplexF16.(e)
            #     # server_cache[] = GPUIndexCache(re, stp, n, e_fp16, cindex_fp16)
            #     # server_cache[] = GPUIndexCache(re, stp, n, e, cindex, cu_dna_ref)
            #     server_cache[] = ri
            # end

            @info "Loading index"
            CUDA.@time begin 
                re, cindex_sp, find_ind, ind_starts = load(i)
                cuts = load(cuts) |> CuArray
            end

            # cu_dna_ref = ri.cu_dna_ref

            # re = ri.re
            # stp = ri.step
            # n = ri.n
            
            # e = ri.cu_e
            
            # cindex = ri.cu_index

            # re2 = RopeEncoder(k=re.k, s=s, m=m, c=c) # Encoder for Phase 2

            if !skip_search
                @info "Loading reads"   
                @time begin 
                    seqs, heads = load_dnas(r, trim=re.k)                   
    
                    seqs_rev = [UInt8(3).-reverse(a) for a in seqs]
                    seqs2 = deepcopy(seqs)
                    append!(seqs2, seqs_rev)
                    # seqs2=seqs
                    jreads = jagged_array(seqs2)
                    # pos = [parse(Int, head[3:end]) for head in heads] # todo: what about descriptions?
                    # reads = hcat(seqs...) # todo: it's more efficient to directly transfer to GPU

                    # @info "Sending reads to GPU"    
                    # creads = reads |> CuArray 
                    # # creads = transpose(reads) |> CuArray 
                end

                @info "Encoding reads"   
                CUDA.@time cropes, _ = encode_batch_cuda_all_best(re, jreads.vect, starts=cu(jreads.starts), stops=cu(jreads.stops), normalize=0)
                cropes = reshape(cropes, :, length(jreads.starts))
                cropes_sp = SplitComplexMatrix(ComplexF16.(cropes))

                # cu_dna_ref = CuArray(dna_ref)
                
                # @views cu_dna_ref_view = cu_dna_ref[1:10^7]
                
                # # CUDA.@time encode_reads_batch_m1_cuda(re, creads, 1:re.k, e=e)
                # # CUDA.@time cropes = encode_batch_m1_cuda(re, creads, 1:re.k, e=e)

                # CUDA.@time vals, locs = sliding_search_cuda(re, cu_dna_ref_view, creads, e)
                # @show typeof(vals)
                # @show vals[1:100]
                # @show locs[1:100]  
                # throw("yo")  

                @info "Main search"
                # locs = search_main(ri, re2, creads; pos=pos)  
                CUDA.@time ranges, _ = search_m1_cuda_split_all_batch(re, re.k÷20, cindex_sp, cropes_sp, find_ind, ind_starts, cuts = cuts, batch_prod_size=2^12, euc=false)

                @info "Saving results"
                open(o, "w") do io
                    # write(io, repr("text/plain", ranges))
                    # repr("text/plain", ranges, context=io)
                    show(io, "text/plain", ranges)
                end

                @info "The results have been saved to $o"

                # throw("yo3")  

                # @time begin                 
                #     @info "Encoding reads"
                #     # @time begin 
                #         # CUDA.@time cropes = encode_batch_m1_cuda(re, creads, 1:re.k-re.s+1, e=e)
                #         # CUDA.@time cropes2 = encode_reads_batch_m1_cuda(re, creads, 1:re.k-re.s+1, e=e)
                #         CUDA.@time cropes, _ = encode_batch_cuda_best(re, creads, e=e)
                        
                #         # @show (cropes .- cropes2) |> norm 
                #         # throw("yo")
                #     # end

                #     CUDA.@time begin 
                #         @info "Splitting complex matrices and converting to FP16"
                #         # cindex_split = SplitComplexMatrix(cindex)
                #         # cropes_split = SplitComplexMatrix(cropes)
                #         cindex_split = SplitComplexMatrix(ComplexF16.(cindex))
                #         cropes_split = SplitComplexMatrix(ComplexF16.(cropes))
                #     end 

                #     @info "Mapping reads, phase 1"
                #     # @time begin 
                #         # cinds = search_batch_m1_cuda(cindex, cropes)
                #         # cinds = search_batch_m1_cuda_split(cindex_split, cropes_split)
                #         cinds = search_cuda_split_max(cindex_split, cropes_split)
                #         # ccarts = search_batch_m1_cuda_split_all(cindex_split, cropes_split, cut = 0.02463403f0)

                #         # cinds_all = search_batch_m1_cuda_split_all(cindex_split, cropes_split, cut = 0.31544152f0)
                #         # ccarts = search_batch_m1_cuda_split_all(cindex_split, cropes_split, cut = 0.30544152f0)
                        
                #     # end
                                        
                #     @info "Getting found locations from GPU"
                #     # carts = ccarts |> Array            
                #     inds = Array(cinds)
                #     # inds_all = Array(cinds_all)                    
                # end

                # N = length(cu_dna_ref)

                # @info "Processing found locations"
                # @time locations = process_carts(carts, n=length(seqs), N=N, k=re.k, stp=stp)  

                # @info "Mapping phase 2"

                # @info "Computing more granular index and mapping to it"
                # stp2 = stp÷100


                
                # ret2 = [Int64[] for i=1:n]
                # nr = length(seqs) # == length(locations)
                # cindex2 = CUDA.zeros(ComplexF32, 4^re.s, 10*re.k ÷ stp2) # todo what max?
                # # rgs0 = hcat([[loc; loc+re.k-1] for loc in 0:stp2:10*re.k]...) |> CuArray # todo max? 
                # # rgs = deepcopy(rgs0)
                # @showprogress for i = 1:11 #nr÷100  
                #     ranges = locations[i]
                #     for range in ranges 
                #         # @show range
                #         N2 = length(range)
                #         num = length(0:stp2:N2-re.k)                        

                #         rgs = [loc+range.start:loc+range.start-1+re.k for loc in 0:stp2:N2-re.k] |> CuArray
                #         # rgs .+= range.start

                #         # rgs = hcat([[loc+range.start; loc+re.k-1+range.start] for loc in 0:stp2:N2-re.k]...) |> CuArray

                #         # @show rgs

                #         # CUDA.@time c
                #         @views encode_ranges_m1_cuda_v1!(cindex2[:,1:num], re, cu_dna_ref, rgs, e = e)
                #         # @views encode_ranges_m1_cuda!(cindex2[:,1:num], re, cu_dna_ref, rgs, e = e)
                #         # @show cindex2[:,1:num]
                #         # @show CUDA.@allowscalar rgs[1,1], rgs[2,1] 

                #         @views cindex2_split = SplitComplexMatrix(ComplexF16.(cindex2[:,1:num]))
                #         @views cropes2_split = SplitComplexMatrix(cropes_split.Re[:,i:i], cropes_split.Im[:,i:i])
                #         cinds2 = search_batch_m1_cuda_split(cindex2_split, cropes2_split) |> Array
                #         # @show cinds2

                #         # CUDA.@time 
                #         # @views cinds2 = search_batch_m1_cuda(cindex2[:,1:num], cropes[:,i:i]) |> Array
                #         push!(ret2[i], range.start + cinds2[1]*stp2)

                #         # copyto!(rgs, rgs0)


                #         # ri2 = RopeIndexer(N=N2, re=re, step=stp÷100)
                #         # dna_ref2 = @view dna_ref[range.+1]
                #         # CUDA.@time cindex2 = index_m1_cuda(ri2, dna_ref2; e=e)
                #         # CUDA.@time cindex2 = index_m1_cuda_nobatch(ri2, dna_ref2; e=e)
                #         # cindex2_split = SplitComplexMatrix(ComplexF16.(cindex2))
                #         # cropes2_split = @view cropes_split[:,i]
                #         # @views cropes2_split = SplitComplexMatrix(ComplexF16.(cropes[:,i]))
                #         # @views cropes2_split = SplitComplexMatrix(cropes_split.Re[:,i:i], cropes_split.Im[:,i:i])
                #         # CUDA.@time 
                #         # cinds2 = search_batch_m1_cuda_split(cindex2_split, cropes2_split) |> Array
                #         # push!(ret2[i], range.start + cinds2[1]*(stp÷100))
                #     end
                # end

                # @show ret2[1:10]
                # # @show ret2[end-100:end]

                # allnl = length.(locations) |> sort 
                # @show allnl[1:100]
                # @show allnl[end-100:end]

                # ranges_all = vcat(locations...)
                # nl = ranges_all |> length
                
                # @show minimum(length.(ranges_all))
                # @show minimum([r.start for r in ranges_all])
                # @show maximum([r.stop for r in ranges_all])
                # @show ranges_all[1:100]
                # @show ranges_all[end-100:end]
                # @show nl

                # @info "Mapping phase 2"

                # @info "Computing more granular index"
                # stp2 = stp ÷ 100
                # 0:self.step:self.N-self.k
                # cindex2 = CUDA.zeros(ComplexF32, 4^re.s, nl)
                # CUDA.@time encode_ranges_m1_cuda!(cindex2, re, CuArray(dna_ref), CuArray(ranges_all), e = e)

                # cindex2_split = SplitComplexMatrix(ComplexF16.(cindex2))

                # @info "More precise mapping"
                # ret2 = [Int64[] for i=1:n]
                # @showprogress for i=1:length(seqs)
                #     ranges = locations[i]
                #     for range in ranges 
                #         # @show range
                #         N2 = length(range)
                #         # ri2 = RopeIndexer(N=N2, re=re, step=stp÷100)
                #         # dna_ref2 = @view dna_ref[range.+1]
                #         # CUDA.@time cindex2 = index_m1_cuda(ri2, dna_ref2; e=e)
                #         # CUDA.@time cindex2 = index_m1_cuda_nobatch(ri2, dna_ref2; e=e)
                #         # cindex2_split = SplitComplexMatrix(ComplexF16.(cindex2))
                #         # cropes2_split = @view cropes_split[:,i]
                #         # @views cropes2_split = SplitComplexMatrix(ComplexF16.(cropes[:,i]))
                #         @views cropes2_split = SplitComplexMatrix(cropes_split.Re[:,i:i], cropes_split.Im[:,i:i])
                #         # CUDA.@time 
                #         cinds2 = search_batch_m1_cuda_split(cindex2_split, cropes2_split) |> Array
                #         push!(ret2[i], range.start + cinds2[1]*(stp÷100))
                #     end
                # end 


                # @info "Comparing with true locations"
                # found = 0
                # for i=1:length(seqs)
                #     ranges = locations[i]
                #     for range in ranges 
                #         if pos[i] in range 
                #             found +=1 
                #             break 
                #         end
                #     end
                # end

                # @show found, length(seqs)

                # preds = inds.*stp
                # diffs = preds.-pos .|> abs 

                # maxdiff = maximum(diffs) 
                # avgdiff = sum(diffs)/length(diffs)

                # mdp = maxdiff/re.k*100
                # adp = avgdiff/re.k*100
                
                # "Mapped precision in % of the read length (0.0 means exact match): " |> println
                # "Maximum: $mdp%" |> println
                # "Average: $adp%" |> println
                # "(Step size is $(stp*100/re.k)%)" |> println



                # @info "Saving positions"
                # open(o, "w") do io
                #     for p in locs
                #         println(io, p)
                #     end
                # end

                # open(o, "w") do io
                #     for i=1:length(seqs)
                #         ranges = locations[i]
                #         line = " "
                #         for range in ranges 
                #             line *= repr(range) * " "
                #         end
                #         println(io, line)
                #     end
                # end 

                # @info "The positions has been saved to $o"           

                if !server 
                    exit() 
                else
                    GC.gc(); CUDA.reclaim(); CUDA.memory_status()
                end 
            end
        elseif parsed_args["%COMMAND%"] == "convert"
            i = parsed_args["convert"]["input"]
            o = parsed_args["convert"]["output"]      

            @info "Reading file"
            seqs, heads = load_dnas(i)

            @info "Saving file"
            save_dnas(seqs, o, heads=heads)

            @info "Done"
        elseif parsed_args["%COMMAND%"] == "server"
            if server 
                if parsed_args["server"]["mode"] == "exit"
                    exit()
                else
                    @info "Already in the server mode."
                end
            else
                @info "Server mode started. Change args.jl to input commands. Write args()=\"arg1 arg2 arg3\" into it."
                @spawn while true 
                    sleep(1)
                    # @show Revise.revision_queue
                    # revise()
                end
            end
        end
    end
end

end
