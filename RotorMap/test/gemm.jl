# --- Full Corrected Code ---

using CUDA
using StaticArrays

# Helper type for vectorized loads/stores
const Float4 = NTuple{4, Float32}

const BM = 128
const BN = 128
const BK = 32
const TM = 8
const TN = 8

"""
    sgemm_optimized_kernel(M, N, K, alpha, A, B, beta, C)

A high-performance SGEMM kernel for Julia/CUDA.
This version fixes critical performance flaws:
1.  Uses larger BK (32) for higher arithmetic intensity.
2.  Uses smaller thread tiles (8x8) to reduce register pressure and increase occupancy.
3.  Uses padded shared memory to avoid bank conflicts without transposing on-the-fly.
4.  Unrolls the compute loop for high instruction-level parallelism (ILP).
"""
function sgemm_optimized_kernel(
    M::Int32, N::Int32, K::Int32, alpha::Float32, 
    A::CuDeviceVector{Float32, 1}, B::CuDeviceVector{Float32, 1}, 
    beta::Float32, C::CuDeviceVector{Float32, 1}
)
    # --- COMPILE-TIME TILE CONSTANTS ---
    # These are the hyperparameters for the kernel.

    
    # --- 1. INDEXING SETUP (using 2D thread blocks for clarity) ---
    # blockIdx and threadIdx are 1-based in Julia, so we subtract 1 for 0-based logic.
    tid_x = threadIdx().x - 1 # spans BN/TN threads
    tid_y = threadIdx().y - 1 # spans BM/TM threads
    block_col = blockIdx().x - 1
    block_row = blockIdx().y - 1

    # --- 2. PADDED SHARED MEMORY ALLOCATION ---
    # Pad by 1 to avoid bank conflicts on shared memory access.
    # A_tile is (BM x BK), B_tile is (BK x BN). We store them as 2D arrays.
    As = CuStaticSharedArray(Float32, (BM, BK + 1))
    Bs = CuStaticSharedArray(Float32, (BK + 1, BN))

    # --- 3. REGISTERS FOR ACCUMULATION ---
    # Each thread computes a TMxTN output tile.
    accumulators = MArray{Tuple{TM, TN}, Float32}(undef)
    @inbounds for i in 1:TM, j in 1:TN
        accumulators[i, j] = 0.0f0
    end

    # --- 4. POINTERS TO A, B, C ---
    # Calculate the starting offset in the flat arrays for this block.
    a_offset = block_row * BM * K
    b_offset = block_col * BN

    # --- 5. MAIN LOOP OVER K DIMENSION ---
    for k in 0:BK:K-1
        # --- Load A tile from Global to Shared Memory ---
        # Each thread is responsible for loading one 4-element vector.
        for i in 0:TM-1
            row = tid_y * TM + i
            col = k + tid_x * 4
            global_a_idx = a_offset + row * K + col + 1 # 1-based index for pointer()
            
            # Bounds checking is crucial for non-multiple-of-tile-size matrices
            if block_row * BM + row < M && col < K
                @inbounds unsafe_store!(
                    reinterpret(Ptr{Float4}, pointer(As, row * (BK + 1) + (col - k) + 1)),
                    unsafe_load(reinterpret(Ptr{Float4}, pointer(A, global_a_idx)))
                )
            end
        end

        # --- Load B tile from Global to Shared Memory ---
        for i in 0:TN-1
            row = k + tid_y * 4
            col = tid_x * TN + i
            global_b_idx = row * N + (block_col * BN + col) + 1

            if row < K && block_col * BN + col < N
                @inbounds unsafe_store!(
                    reinterpret(Ptr{Float4}, pointer(Bs, (row - k) * (BN + 1) + col + 1)),
                    unsafe_load(reinterpret(Ptr{Float4}, pointer(B, global_b_idx)))
                )
            end
        end

        sync_threads() # Wait for all tiles to be loaded

        # --- Computation Stage (with ILP Unrolling) ---
        # Unroll the loop over the K-dimension tile by a factor of 4.
        for dot_idx in 0:4:BK-1
            # Load a 1x4 fragment from As and a 4x1 fragment from Bs into registers.
            # This prefetches data needed for the next 4 calculations.
            a_frag = @inbounds (
                As[tid_y * TM + 1, dot_idx + 1 + 1],
                As[tid_y * TM + 1, dot_idx + 2 + 1],
                As[tid_y * TM + 1, dot_idx + 3 + 1],
                As[tid_y * TM + 1, dot_idx + 4 + 1]
            )
            
            for i in 0:TM-1
                a_frag_local = @inbounds (
                    As[tid_y * TM + i + 1, dot_idx + 1 + 1],
                    As[tid_y * TM + i + 1, dot_idx + 2 + 1],
                    As[tid_y * TM + i + 1, dot_idx + 3 + 1],
                    As[tid_y * TM + i + 1, dot_idx + 4 + 1]
                )
                for j in 0:TN-1
                    b_frag_local = @inbounds (
                        Bs[dot_idx + 1 + 1, tid_x * TN + j + 1],
                        Bs[dot_idx + 2 + 1, tid_x * TN + j + 1],
                        Bs[dot_idx + 3 + 1, tid_x * TN + j + 1],
                        Bs[dot_idx + 4 + 1, tid_x * TN + j + 1]
                    )
                    
                    # Compute 4 FMAs and accumulate. This is the high-ILP part.
                    @inbounds accumulators[i + 1, j + 1] += 
                        a_frag_local[1] * b_frag_local[1] +
                        a_frag_local[2] * b_frag_local[2] +
                        a_frag_local[3] * b_frag_local[3] +
                        a_frag_local[4] * b_frag_local[4]
                end
            end
        end
        sync_threads() # Wait for computation to finish before next tile is loaded
    end

    # --- 6. WRITE BACK RESULTS ---
    for i in 0:TM-1
        for j in 0:TN-1
            row = block_row * BM + tid_y * TM + i
            col = block_col * BN + tid_x * TN + j
            
            # Bounds check for C matrix
            if row < M && col < N
                global_c_idx = row * N + col + 1
                
                # C = alpha * (A*B) + beta * C
                old_c = @inbounds C[global_c_idx]
                @inbounds C[global_c_idx] = alpha * accumulators[i + 1, j + 1] + beta * old_c
            end
        end
    end
    return nothing
end


# --- Host Wrapper ---

function launch_sgemm_optimized(M, N, K, alpha, A, B, beta, C)
    # These must match the constants in the kernel
    BM = 128
    BN = 128
    TM = 8
    TN = 8

    # Calculate Grid/Block dimensions
    threads_per_block = (BN ÷ TN, BM ÷ TM) # e.g., (16, 16) -> 256 threads
    grid_dims = (cld(N, BN), cld(M, BM))  # Grid spans the output matrix C
    
    # Calculate shared memory size (with padding)
    shmem_size = (BM * (BK + 1) + (BK + 1) * BN) * sizeof(Float32)

    @cuda threads=threads_per_block blocks=grid_dims shmem=shmem_size sgemm_optimized_kernel(
        Int32(M), Int32(N), Int32(K), Float32(alpha), A, B, Float32(beta), C
    )
end

# --- Testing ---
M, N, K = 2048, 2048, 2048
# Use column-major layout to match Julia's standard and CUDA.jl's mul!
A = CUDA.rand(Float32, K, M)
B = CUDA.rand(Float32, N, K)
C_opt = CUDA.zeros(Float32, N, M)
C_ref = CUDA.zeros(Float32, N, M)

alpha = 1.0f0
beta = 0.0f0

println("Running optimized kernel...")
CUDA.@time begin
    launch_sgemm_optimized(M, N, K, alpha, A, B, beta, C_opt)
    CUDA.synchronize()
end

println("Running cuBLAS...")
# Use CUDA.jl's highly optimized wrapper to cuBLAS for comparison
CUDA.@time begin
    mul!(C_ref, B, A, alpha, beta) # Note: mul! computes C=alpha*B*A+beta*C
    CUDA.synchronize()
end

# Verify correctness
if all(isapprox.(Array(C_opt), Array(C_ref), rtol=1e-4))
    println("\n✅ Success: Optimized kernel results match cuBLAS.")
else
    println("\n❌ Failure: Results do not match.")
end