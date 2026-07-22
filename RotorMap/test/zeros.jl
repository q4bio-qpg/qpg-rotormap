function get_name(head)
    split(head, " ")[1][2:end]
end

readsname = "reads.fastq"
seqs, heads = load_fastq_ignore_quality(readsname)

data = JSON3.read("levenshtein_dists.json")

n = length(data.distances)

names = [get_name(h) for h in heads]

dists = Any[]
cur = 1
found = 0
bestdist = 20000
i = 1
while i <= n
    dnm = data.distances[i].read_name    
    # @show i, cur, names[z[cur]], dnm, found
    if dnm == names[z[cur]]     
        # @show i, cur, names[z[cur]]   
        found = 1
        dist = data.distances[i].levenshtein_distance
        bestdist = min(dist, bestdist)
        i += 1
    elseif dnm != names[z[cur]] 
        if found == 1                    
            found = 0
            push!(dists, (cur,bestdist))
            bestdist = 20000
            cur += 1    
        else
            ind = parse(Int, split(dnm, ".")[2])
            cind = parse(Int, split(names[z[cur]], ".")[2])
            if ind < cind 
                i += 1       
            else                
                push!(dists, (cur,20000))
                cur += 1
            end
        end
    end        
end