"""
    count_4mers_base(dna::String)

Computes the total number of each distinct 4-mer in a DNA string
using only Julia's Base language.

# Arguments
- `dna::String`: A string containing DNA characters.

# Returns
- `Dict{String, Int}`: A dictionary of 4-mer counts.
"""
function count_4mers_base(dna::String, l=4)
    counts = Dict{String, Int}()
    len = length(dna)

    # If the string is too short, no 4-mers exist
    if len < 4
        return counts
    end

    # Iterate from the first possible 4-mer to the last
    for i in 1:(len - l+1)
        # Extract the 4-mer
        kmer = dna[i:i+l-1]
        
        # Increment its count in the dictionary.
        # `get(counts, kmer, 0)` returns the current count or 0 if it's new.
        counts[kmer] = get(counts, kmer, 0) + 1
    end
    
    return counts
end

# --- Example Usage ---
my_dna = "ATCGATCGAATCGATATCG"

# Compute the 4-mer counts
# kmer_counts_base = count_4mers_base(my_dna)
kmer_counts_base = count_4mers_base(ss, 6)

pairs = collect(kmer_counts_base)
sorted_pairs = sort(pairs, by=last, rev=true)

# # Print the results
# println("\n4-mer counts using pure Julia for '$my_dna':")
# for (kmer, count) in sort(collect(kmer_counts_base))
#     println("$kmer: $count")
# end