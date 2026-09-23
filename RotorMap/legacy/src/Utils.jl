# __precompile__(true)
module Utils

export save, load, generate_reference, generate_reads, mutate, mutate_shift, save_fasta, load_fasta
export load_dnas, save_dnas
export load_fasta_optimized, load_fasta_mmap, load_fastq_ignore_quality
export jagged_array, JaggedArray
export load_fastq_ignore_quality_fixed
export load_fasta_mmap_fixed

using CUDA
using Base.Threads
using Random 
using Serialization
using ProgressMeter
using Mmap
using ..TreeArrays
# using FASTX 

struct JaggedArray
	vect::AbstractVector # the combined linear array
	starts::AbstractVector # start positions of each piece
	stops::AbstractVector # stop positions of each piece
	descs::Union{AbstractVector, Nothing} # optional descriptions 
end

"""
Concatenates a vector of vectors into a single vector with saving of the start and stop positions of each piece. 
The result is saved in the JaggedArray structure. 
Descriptions of pieces can be provided. 

The combined vector is stored on GPU by default. 
"""
function jagged_array(data::AbstractVector{<:AbstractVector{T}}, descs=nothing; dev=:gpu)::JaggedArray where T
	num = data |> length # number of pieces
	total_length = sum(data .|> length)

	vect = Vector{T}(undef, total_length)
	starts = Vector{Int}(undef, num)
	stops = Vector{Int}(undef, num)
	
    total_current = 0
	for i = 1:num
		piece = data[i]
		len = piece |> length
		starts[i] = 1 + total_current
		stops[i] = len + total_current

		@views vect[1 + total_current : len + total_current] .= piece
		total_current += len
	end

    if dev==:gpu
	    vect = vect |> CuArray
    end

	return JaggedArray(vect, starts, stops, descs)
end

function load_dnas(file; trim = 0, skipN = false)
    fa = rsplit(file, ".", limit=2)
    file_name = fa[1]
    file_ext = fa[2]

	if file_ext == "bin"
		dna_refs, heads = load(file)
	elseif file_ext == "fasta"
		# dna_refs, heads = load_fasta(file)
		# dna_refs, heads = load_fasta_optimized(file)
		dna_refs, heads = load_fasta_mmap(file, trim=trim, skipN=skipN)
    elseif file_ext == "fastq"
        dna_refs, heads = load_fastq_ignore_quality(file, trim=trim, skipN=skipN)
	end

    return dna_refs, heads
end

function save_dnas(seqs::AbstractArray, file::String; heads=[])
	fa = rsplit(file, ".", limit=2)
    file_name = fa[1]
    file_ext = fa[2]

	if file_ext == "bin"
		save((seqs, heads), file)
	elseif file_ext == "fasta"
		save_fasta(seqs, file, heads=heads)
	end
	return 1
end

function save(data, file)
    open(file, "w") do f
        serialize(f, data)
    end
end

function load(file)
    data = open(file, "r") do f
        deserialize(f)
    end
    return data
end

const dna_letters = UInt8.(['A', 'C', 'G', 'T'])
# const dna_decodes = Dict(zip(dna_letters, UInt8.(0:3)))
# function dna_decodes(x)
# 	if x==UInt8('A')
# 		return UInt8(0)
# 	elseif x==UInt8('C')
# 		return UInt8(1)
# 	elseif x==UInt8('G')
# 		return UInt8(2)
# 	elseif x==UInt8('T')
# 		return UInt8(3)
# 	end

# 	throw("Unknown letter $x")
# 	return 0
# end 

function dna_hash(dna::AbstractVector, hash_size=8)::String
	# todo: convert dna to UInt8

	n = length(dna)
	if n < hash_size 
		padded = vcat(dna, UInt8.(255*ones(hash_size-n)))
		return bytes2hex(padded)
	end

	@views begin 
		hash1 = copy(dna[1:hash_size])
		i = hash_size
		while i+hash_size <= n
			hash1 .+= dna[i+1:i+hash_size]
			i += hash_size
		end

		if i+1 <= n 
			hash1[1:n-i] .+= dna[i+1:n]
		end
	end
    
    return bytes2hex(hash1)
end

# !!!important use only ASCII in the description, or reading might break
function save_fasta(dnas::AbstractArray, file::String; heads=[], pos=[], chunk_size=2^12)
	n = length(dnas)

	if heads==[]
		if pos != [] 
			heads = ["> " * repr(pos[i]) for i in eachindex(dnas)]
		else
			heads = ["> " for i in eachindex(dnas)]
		end 
    end

	write_buffer = Vector{UInt8}(undef, chunk_size)

	open(file, "w") do io
        # Use zip for cleaner, more direct iteration over sequences and their headers
        @showprogress for (dna, head) in zip(dnas, heads)
            # Write the header for the current sequence
            println(io, head)

            # Process the current DNA sequence in chunks
            for i_start in 1:chunk_size:length(dna)
                # Determine the end of the current chunk, correctly handling the last one
                i_end = min(i_start + chunk_size - 1, length(dna))
                
                # The number of bytes to actually write from the buffer
                chunk_len = i_end - i_start + 1

                # Fill the pre-allocated buffer in-place.
                # This is the core of the allocation reduction.
                @inbounds for k in 1:chunk_len
                    # dna is 0-indexed (0-3), so we add 1 for 1-based indexing.
                    # We directly map from the number to the corresponding ASCII byte.
                    write_buffer[k] = dna_letters[dna[i_start + k - 1] + 1]
                end
                
                # Write only the filled part of the buffer to the file.
                # `view` avoids creating another array for the slice.
                write(io, view(write_buffer, 1:chunk_len))
            end
            
            # FASTA format requires a newline at the end of each sequence
            println(io)
        end
    end
	return 1
end

# function load_fasta(file::String; as_matrix::Bool = true)
function load_fasta(file::String; trim = 0)
    dna_letters = UInt8.(['A', 'C', 'G', 'T'])
	dna_decodes = Dict(zip(dna_letters, UInt8.(0:3)))

    headers = String[]
    sequences = Vector{UInt8}[]
    current_sequence_buffer = IOBuffer()

    open(file, "r") do io
		seekend(io)
		fileSize = position(io)

		seekstart(io)
		p = Progress(fileSize; dt=1.0)   # minimum update interval: 1 second

        for line in eachline(io)

			update!(p, position(io))

            # line = strip(line)
            if isempty(line)
                continue
            end

            if startswith(line, ">")
			# if line[1] == UInt8('>')
                # It's a new header, so we finalize the previous sequence (if any)
                if !isempty(sequences) || position(current_sequence_buffer) > 0
                    seq_data = take!(current_sequence_buffer)                    
                    if trim > 0 
                        len = length(seq_data)
                        if len < trim 
                            # pass 
                        else
                            @views decoded_seq = [dna_decodes[b] for b in seq_data[1:trim]]
                            push!(sequences, decoded_seq)
                        end
                    else
                        decoded_seq = [dna_decodes[b] for b in seq_data]
                        push!(sequences, decoded_seq)
                    end
                end
                push!(headers, line)
            else
                # It's a sequence line, append it to the current buffer
                write(current_sequence_buffer, line)
            end
        end
    end

    # Don't forget to add the last sequence after the loop finishes
	# @show position(current_sequence_buffer)

    if position(current_sequence_buffer) > 0
        seq_data = take!(current_sequence_buffer)
        if trim > 0 
            len = length(seq_data)
            if len < trim 
                # pass 
            else
                @views decoded_seq = [dna_decodes[b] for b in seq_data[1:trim]]
                push!(sequences, decoded_seq)
            end
        else
            decoded_seq = [dna_decodes[b] for b in seq_data]
            push!(sequences, decoded_seq)
        end
    end

	return (sequences, headers)
end

function load_fasta_optimized(file::String)
    # 1. Create a Fast Lookup Table (0-255)
    # We use a fixed-size array instead of a Dict for O(1) access without hashing.
    # Initialize with 0xff (255) to represent "ignore/invalid" characters.
    lut = fill(0xff, 256)
    
    # Map 'A', 'C', 'G', 'T' to 0, 1, 2, 3
    # Note: Julia arrays are 1-based, ASCII is 0-based, so we use Int(char)+1
    lut[Int('A')+1] = 0x00
    lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02
    lut[Int('T')+1] = 0x03
    
    # Handle lowercase cases (optional but recommended)
    lut[Int('a')+1] = 0x00
    lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02
    lut[Int('t')+1] = 0x03

    headers = String[]
    sequences = Vector{Vector{UInt8}}()

    # 2. Read the whole file into memory
    # For files larger than RAM, replace read(file) with Mmap.mmap(open(file))
    raw_data = read(file)
    n = length(raw_data)
    
    p = Progress(n; dt=1.0)
    i = 1

    # 3. State Machine Loop
    while i <= n
        byte = raw_data[i]

        if byte == UInt8('>')
            # --- HEADER FOUND ---
            start_head = i
            # Scan forward until newline
            while i <= n && raw_data[i] != UInt8('\n')
                i += 1
            end
            
            # Extract header (handling Windows \r\n and Unix \n)
            end_head = i - 1
            if end_head >= start_head && raw_data[end_head] == UInt8('\r')
                end_head -= 1
            end
            push!(headers, String(raw_data[start_head:end_head]))
            
            i += 1 # Move past the newline

            # --- PARSE SEQUENCE ---
            # Pre-allocate a buffer for the sequence. 
            # We can use sizehint! if we can guess the length (e.g. from previous seq)
            current_seq = UInt8[]
            sizehint!(current_seq, 1024) 

            while i <= n && raw_data[i] != UInt8('>')
                b = raw_data[i]
                val = lut[b+1] # Instant array lookup
                
                # Only push valid DNA characters (skips \n, \r, and spaces automatically)
                if val != 0xff
                    push!(current_seq, val)
                end
                
                i += 1
            end
            push!(sequences, current_seq)
            
            # Update progress only occasionally to save time
            if i % 100_000 == 0
                update!(p, i)
            end
        else
            # Skip any bytes before the first header
            i += 1
        end
    end
    
    finish!(p)
    return (sequences, headers)
end

function load_fasta_mmap(file::String, randomize_N = false; trim=0, skipN=false)
    # 1. Setup Lookup Table (same as before)
    lut = fill(0xff, 256)
    lut[Int('A')+1] = 0x00; lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02; lut[Int('T')+1] = 0x03
    lut[Int('a')+1] = 0x00; lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02; lut[Int('t')+1] = 0x03

	# lut[Int('N')+1] = 0x04
	lut[Int('N')+1] = 0x02 # todo: we treat each N as G, since it is harder to implement the fair way

    headers = String[]
    sequences = Vector{Vector{UInt8}}()

    # 2. Open file and create the Memory Map
    open(file, "r") do io
        # Mmap.mmap returns an array-like object linked to the disk file
        # The OS handles the loading/unloading of data pages.
        raw_data = Mmap.mmap(io)
        n = length(raw_data)
        
        # Initialize loop variables
        p = Progress(n; dt=1.0)
        i = 1

        skip_seq = false

        # 3. Parsing Loop (Identical logic)
        while i <= n
            # If we are at the very end and there's a trailing newline, break
            if i == n && (raw_data[i] == UInt8('\n') || raw_data[i] == UInt8('\r'))
                break
            end

            byte = raw_data[i]

            if byte == UInt8('>')
                # --- HEADER FOUND ---
                start_head = i
                # Scan until newline
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                
                # Handle Windows \r\n
                end_head = i - 1
                if end_head >= start_head && raw_data[end_head] == UInt8('\r')
                    end_head -= 1
                end
                
                # Convert header bytes to String
                # Note: This does allocate a small String, which is usually fine
                push!(headers, String(raw_data[start_head:end_head]))
                
                i += 1 # Move past \n

                # --- PARSE SEQUENCE ---
                current_seq = UInt8[]
                sizehint!(current_seq, 1024) 

                while i <= n && raw_data[i] != UInt8('>')
                    b = raw_data[i]
                    val = lut[b+1] 

                    if b == Int('N') 
                        if skipN 
                            skip_seq = true
                        elseif randomize_N
                            val = rand(UInt8.(0:3))
                        end
                    end
                    if val != 0xff
                        push!(current_seq, val)
                    end
                    i += 1
                end

                # push!(sequences, current_seq)
                if trim > 0 
                    len = length(current_seq)
                    if len < trim || skip_seq
                        # pass 
                    else
                        @views push!(sequences, current_seq[1:trim])
                        skip_seq = false
                    end
                else
                    push!(sequences, current_seq)                
                end                
                
                if i % 100_000 == 0
                    update!(p, i)
                end
            else
                # Skip junk before first header
                i += 1
            end
        end
    end # 'io' closes here, and the mmap is automatically finalized

    return (sequences, headers)
end

using Mmap
using ProgressMeter

function load_fasta_mmap_fixed(file::String, randomize_N = false; trim=0, skipN=false)
    # 1. Setup Lookup Table
    lut = fill(0xff, 256)
    lut[Int('A')+1] = 0x00; lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02; lut[Int('T')+1] = 0x03
    lut[Int('a')+1] = 0x00; lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02; lut[Int('t')+1] = 0x03

    # Note: Treating N as G. Change to 0xff if you prefer to drop them.
    lut[Int('N')+1] = 0x02 
    lut[Int('n')+1] = 0x02 

    headers = String[]
    sequences = Vector{Vector{UInt8}}()

    # 2. Open file and create the Memory Map
    open(file, "r") do io
        raw_data = Mmap.mmap(io)
        n = length(raw_data)
        
        p = Progress(n; dt=1.0)
        i = 1
        last_progress_update = 1

        # 3. Parsing Loop
        while i <= n
            if i == n && (raw_data[i] == UInt8('\n') || raw_data[i] == UInt8('\r'))
                break
            end

            byte = raw_data[i]

            if byte == UInt8('>')
                # --- 1. PARSE HEADER ---
                start_head = i
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                
                end_head = i - 1
                if end_head >= start_head && raw_data[end_head] == UInt8('\r')
                    end_head -= 1
                end
                
                # Store header temporarily
                current_header = String(raw_data[start_head:end_head])
                i += 1 

                # --- 2. PARSE SEQUENCE ---
                current_seq = UInt8[]
                sizehint!(current_seq, 1024) 
                
                skip_seq = false # Reset for EACH new sequence

                while i <= n && raw_data[i] != UInt8('>')
                    b = raw_data[i]
                    val = lut[b+1] 

                    if b == UInt8('N') || b == UInt8('n')
                        if skipN 
                            skip_seq = true
                        elseif randomize_N
                            val = rand(UInt8.(0:3))
                        end
                    end
                    
                    if val != 0xff
                        push!(current_seq, val)
                    end
                    i += 1
                end

                # --- 3. FINALIZE & PUSH ---
                # Push both header and sequence together if it passes filters
                if !skip_seq
                    if trim > 0 
                        if length(current_seq) >= trim
                            push!(headers, current_header)
                            push!(sequences, current_seq[1:trim])
                        end
                    else
                        push!(headers, current_header)
                        push!(sequences, current_seq)                
                    end 
                end               
                
                # Update progress tracking reliably
                if i - last_progress_update >= 100_000
                    update!(p, i)
                    last_progress_update = i
                end
            else
                # Skip junk before first header
                i += 1
            end
        end
    end

    return (sequences, headers)
end

function load_fastq_ignore_quality(file::String, randomize_N=false; trim = 0, skipN=false)
    # 1. Setup Lookup Table (Same as FASTA)
    lut = fill(0xff, 256)
    lut[Int('A')+1] = 0x00; lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02; lut[Int('T')+1] = 0x03
    lut[Int('a')+1] = 0x00; lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02; lut[Int('t')+1] = 0x03
    # Note: 'N' or other IUPAC codes will be skipped/ignored with this LUT. 
    # If you need 'N' as a value (e.g., 4), add it here.
	lut[Int('N')+1] = 0x02 # todo: careful, as many N could cause many matches in the index 

    headers = String[]
    sequences = Vector{Vector{UInt8}}()

    open(file, "r") do io
        raw_data = Mmap.mmap(io)
        n = length(raw_data)
        p = Progress(n; dt=1.0)
        i = 1
        
        # We need a flag to track if we are at the start of a line
        # to correctly identify the '+' separator.
        is_line_start = true

        skip_seq = false 

        while i <= n
            byte = raw_data[i]

            # FASTQ records start with '@' at the beginning of a line
            if is_line_start && byte == UInt8('@')
                
                # --- 1. PARSE HEADER ---
                start_head = i
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                
                # Handle Windows \r\n
                end_head = i - 1
                if end_head >= start_head && raw_data[end_head] == UInt8('\r')
                    end_head -= 1
                end
                
                # Push header (excluding the @ if you prefer, here we keep it to match line)
                push!(headers, String(raw_data[start_head:end_head]))
                i += 1 # Move past \n
                is_line_start = true

                # --- 2. PARSE SEQUENCE ---
                current_seq = UInt8[]
                sizehint!(current_seq, 150) # FASTQ reads are often short (150bp)

                while i <= n
                    b = raw_data[i]

                    # Check for Separator '+'
                    # It must be at the start of a line.
                    if is_line_start && b == UInt8('+')
                        break # Done with sequence
                    end

                    if b == UInt8('\n')
                        is_line_start = true
                        i += 1
                        continue
                    elseif b == UInt8('\r')
                        i += 1
                        continue
                    end

                    # It's a sequence character
                    val = lut[b+1]

                    if b == Int('N') 
                        if skipN 
                            skip_seq = true
                        elseif randomize_N
                            val = rand(UInt8.(0:3))
                        end
                    end
                    if val != 0xff
                        push!(current_seq, val)
                    end
                    
                    is_line_start = false
                    i += 1
                end
                # push!(sequences, current_seq)
                if trim > 0 
                    len = length(current_seq)
                    if len < trim || skip_seq
                        # pass 
                    else
                        @views push!(sequences, current_seq[1:trim])
                        skip_seq = false
                    end
                else
                    push!(sequences, current_seq)                
                end   

                # --- 3. PARSE SEPARATOR ---
                # We are currently at the '+' line. Skip until newline.
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                i += 1 # Past the newline of the '+' line
                
                # --- 4. SKIP QUALITY SCORES ---
                # The quality block has exactly as many chars as the sequence length.
                # However, it might be split across newlines (multi-line FASTQ).
                rem_qual = length(current_seq)
                
                while i <= n && rem_qual > 0
                    b = raw_data[i]
                    if b != UInt8('\n') && b != UInt8('\r')
                        rem_qual -= 1
                    end
                    i += 1
                end
                
                # After skipping 'rem_qual' chars, we might be sitting on a trailing newline
                # before the next '@'. Consume it so the loop restarts clean.
                while i <= n && (raw_data[i] == UInt8('\n') || raw_data[i] == UInt8('\r'))
                    i += 1
                end
                
                # Update loop state
                is_line_start = true 
                
                # Update progress
                if i % 50_000 == 0
                    update!(p, i)
                end

            else
                # If we are not at an '@' (e.g., junk or whitespace), advance.
                if byte == UInt8('\n')
                    is_line_start = true
                elseif byte != UInt8('\r')
                    is_line_start = false
                end
                i += 1
            end
        end
    end

    return (sequences, headers)
end

function load_fastq_ignore_quality_fixed(file::String, randomize_N=false; trim=0, skipN=false)
    # 1. Setup Lookup Table
    lut = fill(0xff, 256)
    lut[Int('A')+1] = 0x00; lut[Int('C')+1] = 0x01
    lut[Int('G')+1] = 0x02; lut[Int('T')+1] = 0x03
    lut[Int('a')+1] = 0x00; lut[Int('c')+1] = 0x01
    lut[Int('g')+1] = 0x02; lut[Int('t')+1] = 0x03
    
    # Optional: Map N to G. Consider changing to 0xff to drop Ns by default.
    lut[Int('N')+1] = 0x02 
    lut[Int('n')+1] = 0x02 

    headers = String[]
    sequences = Vector{Vector{UInt8}}()

    open(file, "r") do io
        raw_data = Mmap.mmap(io)
        n = length(raw_data)
        p = Progress(n; dt=1.0)
        i = 1
        is_line_start = true
        last_progress_update = 1

        while i <= n
            byte = raw_data[i]

            if is_line_start && byte == UInt8('@')
                
                # --- 1. PARSE HEADER ---
                start_head = i
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                
                end_head = i - 1
                if end_head >= start_head && raw_data[end_head] == UInt8('\r')
                    end_head -= 1
                end
                
                # Temporarily store the header instead of pushing it immediately
                current_header = String(raw_data[start_head:end_head])
                i += 1
                is_line_start = true

                # --- 2. PARSE SEQUENCE ---
                current_seq = UInt8[]
                sizehint!(current_seq, 150)
                skip_seq = false
                raw_seq_len = 0 # Track RAW length for accurate quality skipping

                while i <= n
                    b = raw_data[i]

                    if is_line_start && b == UInt8('+')
                        break
                    end

                    if b == UInt8('\n')
                        is_line_start = true
                        i += 1
                        continue
                    elseif b == UInt8('\r')
                        i += 1
                        continue
                    end

                    raw_seq_len += 1 # Count every non-newline character
                    val = lut[b+1]

                    if b == UInt8('N') || b == UInt8('n')
                        if skipN 
                            skip_seq = true
                        elseif randomize_N
                            val = rand(UInt8.(0:3))
                        end
                    end
                    
                    if val != 0xff
                        push!(current_seq, val)
                    end
                    
                    is_line_start = false
                    i += 1
                end

                # --- 3. PARSE SEPARATOR ---
                while i <= n && raw_data[i] != UInt8('\n')
                    i += 1
                end
                i += 1 
                
                # --- 4. SKIP QUALITY SCORES ---
                # Use raw_seq_len, not length(current_seq)
                rem_qual = raw_seq_len 
                
                while i <= n && rem_qual > 0
                    b = raw_data[i]
                    if b != UInt8('\n') && b != UInt8('\r')
                        rem_qual -= 1
                    end
                    i += 1
                end
                
                while i <= n && (raw_data[i] == UInt8('\n') || raw_data[i] == UInt8('\r'))
                    i += 1
                end
                
                is_line_start = true 
                
                # --- 5. FINALIZE & PUSH ---
                # Push header and sequence AT THE SAME TIME to prevent desync
                if !skip_seq
                    if trim > 0 
                        if length(current_seq) >= trim
                            push!(headers, current_header)
                            push!(sequences, current_seq[1:trim])
                        end
                    else
                        push!(headers, current_header)
                        push!(sequences, current_seq)                
                    end
                end

                # Update progress tracking reliably
                if i - last_progress_update >= 50_000
                    update!(p, i)
                    last_progress_update = i
                end

            else
                if byte == UInt8('\n')
                    is_line_start = true
                elseif byte != UInt8('\r')
                    is_line_start = false
                end
                i += 1
            end
        end
    end

    return (sequences, headers)
end

# function save_fasta(dnas::AbstractArray, file::String; desc="", pos=[])
# 	sz = size(dnas)
# 	if length(sz) == 1 # a single DNA, treating as a reference 
# 		dna = dnas
# 		@time dnahash = dna_hash(dna)
# 		@time dnastr = join(dna_letters[dna.+1])
# 		@time FASTAWriter(open(file, "w")) do writer
# 			record = FASTARecord("$desc hash#$dnahash", dnastr)
# 			write(writer, record)
# 		end
# 	else
# 		n = sz[2]

# 		FASTAWriter(open(file, "w")) do writer
# 			for i = 1:n 				
# 				@views dna = dnas[:,i]
# 				dnahash = dna_hash(dna)
# 				dnastr = join(dna_letters[dna.+1])
# 				position = "" 
# 				description = ""
# 				try 
# 					position = "pos" * repr(pos[i])
# 				catch
# 				end
# 				try 
# 					description = desc[i]
# 				catch
# 				end
# 				record = FASTARecord("$description $position hash#$dnahash", dnastr)
# 				write(writer, record)
# 			end
# 		end
# 	end 
# 	return 1
# end

# function load_fasta(file)
# #  map(c -> dna_decodes[c], g)

# end

function generate_reference(N::Int64; seed::Union{Integer, Nothing} = nothing)
    rng = Random.default_rng()
    if seed != nothing
        rng = Xoshiro(seed)
    end

    dna_ref = rand(rng, UInt8.(0:3), N)
	
    return dna_ref
end

function generate_reads(dna_ref::Vector{T}, k::Int64, n::Int64=1; err::Float64=0.15, seed::Union{Int64, Nothing}=nothing) where T
    # reads = zeros(T, k, n)  
	reads = [zeros(T, k) for i = 1:n]
	
	rng = Random.default_rng()
    if seed != nothing
        rng = Xoshiro(seed)
    end

	N = length(dna_ref)
	pos = rand(rng, 0:N-k, n)

	rngs = [Xoshiro(rand(rng, Int64)) for i=1:n] # generate rngs for each execution thread for reproducibility

	@showprogress @threads for i in 1:n
        @views dna = dna_ref[pos[i]+1:pos[i]+k]
        mut = mutate(dna, err, rng=rngs[i])
		# @views reads[:,i] = mut
		reads[i] .= mut
	end
	
	return reads, pos
end

function mutate(dna::AbstractVector{T}, err=0.1; rng=Random.default_rng(), mll=1000) where T # dna = [2,1,0,3,1,0,...]
	n = length(dna)
	mut = TreeArray(dna, max_len_leaf=mll)	
	n_err = n*err |> ceil |> Int
	ins_rate = 1/3
	n_ins = n_err*ins_rate |> floor |> Int
	n_del = n_ins
	n_sub = n_err - n_ins - n_del

	inds_s = Array{Int}(undef, n_sub)
	inds_d = Array{Int}(undef, n_del)
	inds_i = Array{Int}(undef, n_ins)

	vals_s = Array{T}(undef, n_sub)
	vals_i = Array{T}(undef, n_ins)
	
	rngs = [Xoshiro(rand(rng, Int64)) for i=1:4] # generate rngs for each execution thread for reproducibility
	@sync begin
		@spawn begin 
			for t=1:n_sub # gen indices for substitutions
				inds_s[t] = n*rand(rngs[1]) |> ceil |> Int
			end
			sort!(inds_s)

            for i in 1:n_sub # gen values for substitutions
				rl = rand(rngs[1], [T(k) for k=1:3])
				l = (dna[inds_s[i]] + rl)%4 # mutated letter
				vals_s[i] = l
			end
		end
		@spawn begin # gen indices for deletions
			for t=0:n_del-1
				inds_d[t+1] = (n-t)*rand(rngs[2]) |> ceil |> Int 
				# inds_d[t+1] = (n-t)*rand(rngs[2])/2 |> ceil |> Int 
			end
			sort!(inds_d)

			# make inds_d unique (this code is equivalent to that)
			for i = 2:n_del
				inds_d[i] = max(inds_d[i], inds_d[i-1]+1)
			end
		end 
		@spawn begin 
			for t=0:n_ins-1 # gen indices for insertions
				inds_i[t+1] = (n-n_del)*rand(rngs[3]) |> ceil |> Int 
				# inds_i[t+1] = (n-n_del) - (n-n_del)*rand(rngs[3])/2 |> ceil |> Int 
			end
			sort!(inds_i)
        end
        @spawn begin
			for i in 1:n_ins # gen values for insertions
				rl = rand(rngs[4], [T(k) for k=0:3])
				vals_i[i] = rl
			end		
		end
	end
	
	subsat!(mut, inds_s, vals_s)
	deleteat!(mut, inds_d)
	insertat!(mut, inds_i, vals_i)

	ret = Vector(mut)
	return ret
end

"""
shift - the exact number of letters that we erase (and add in the end)
err - fraction of deleted+inserted letters; used if shift is not supplied
"""
function mutate_shift(dna::AbstractVector{T}, err=0.1; shift=nothing, rng=Random.default_rng()) where T # dna = [2,1,0,3,1,0,...]
	n = length(dna)
    if shift==nothing
	    shift = n*err/2 |> floor |> Int
    end
	tail = rand(rng, 0:3, shift)

	ret = vcat(view(dna, 1+shift:n), tail)
	return ret 
end

function count_kmers(dna::String, k=4)
    counts = Dict{String, Int}()
    len = length(dna)

    # If the string is too short, no k-mers exist
    if len < k
        return counts
    end

    # Iterate from the first possible k-mer to the last
    for i in 1:(len - k+1)
        # Extract the 4-mer
        kmer = dna[i:i+k-1]
        
        # Increment its count in the dictionary.
        # `get(counts, kmer, 0)` returns the current count or 0 if it's new.
        counts[kmer] = get(counts, kmer, 0) + 1
    end
    
    return counts
end


end