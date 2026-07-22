using CUDA
using LinearAlgebra
using StaticArrays

const WARPSIZE = 32

# Main SGEMM kernel with warptiling, rewritten to be compiler-friendly
function sgemm_warptiling_kernel(
    A::CuDeviceMatrix{Float32}, B::CuDeviceMatrix{Float32},
    C::CuDeviceMatrix{Float32}, alpha::Float32, beta::Float32
)
    # Tile parameters are now compile-time constants for this kernel
    BM, BN, BK = 128, 128, 16
    WM, WN = 64, 32
    TM, TN = 4, 4
    NUM_THREADS = 256

    # Derived constants calculated at compile time
    WNITER = 2
    WMITER = (WM * WN) ÷ (WARPSIZE * TM * TN * WNITER)
    WSUBM = WM ÷ WMITER
    WSUBN = WN ÷ WNITER

    # === Thread and Block Indexing ===
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x

    # Warp positioning within the threadblock tile
    warpIdx = tx ÷ WARPSIZE
    warpCol = warpIdx % (BN ÷ WN)
    warpRow = warpIdx ÷ (BN ÷ WN)

    # Thread positioning within the warp
    threadIdxInWarp = tx % WARPSIZE
    threadColInWarp = threadIdxInWarp % (WSUBN ÷ TN)
    threadRowInWarp = threadIdxInWarp ÷ (WSUBN ÷ TN)

    # === Shared Memory ===
    As = @cuDynamicSharedMem(Float32, BM * BK)
    Bs = @cuDynamicSharedMem(Float32, BK * BN, BM * BK * sizeof(Float32))

    # === Matrix Dimensions and Pointer Arithmetic ===
    M, N = size(C)
    K = size(A, 2)

    # Base pointers for this block's computation
    A_block_start_row = by * BM
    B_block_start_col = bx * BN
    
    # Note: pointer(A) gets the raw memory address. Offsets are in bytes.
    A_ptr = pointer(A) + A_block_start_row * K * sizeof(Float32)
    B_ptr = pointer(B) + B_block_start_col * sizeof(Float32)

    # === Thread's Loading Indices ===
    innerRowA = tx ÷ (BK ÷ 4)
    innerColA = tx % (BK ÷ 4)
    rowStrideA = (NUM_THREADS * 4) ÷ BK
    innerRowB = tx ÷ (BN ÷ 4)
    innerColB = tx % (BN ÷ 4)
    rowStrideB = NUM_THREADS ÷ (BN ÷ 4)

    # === Register Allocation (The Critical Fix) ===
    # The @MArray zeros(...) macro ensures the compiler knows the exact type and size
    # and can place these arrays directly into the register file.
    threadResults = @MArray zeros(Float32, WMITER * TM * WNITER * TN)
    regM = @MArray zeros(Float32, WMITER * TM)
    regN = @MArray zeros(Float32, WNITER * TN)

    # === Main K-dimension Loop ===
    num_tiles = cld(K, BK)
    for bkIdx in 0:num_tiles-1
        # --- Gmem to Smem: Load A tile (with float4) ---
        for offset in 0:rowStrideA:BM-1
            rowA = innerRowA + offset
            colA4 = innerColA * 4
            # Calculate if this thread's load is within the block tile bounds
            if rowA < BM && colA4 + 3 < K
                # Global memory index for the start of this float4
                global_idx_a = rowA * K + colA4
                # Use unsafe_load with reinterpret for an efficient float4 load
                tmp = unsafe_load(reinterpret(Ptr{NTuple{4, Float32}}, A_ptr + global_idx_a * sizeof(Float32)))
                # Store to shared memory, transposing for efficient access in compute loop
                As[(colA4 + 1) * BM + rowA + 1] = tmp[1]
                As[(colA4 + 2) * BM + rowA + 1] = tmp[2]
                As[(colA4 + 3) * BM + rowA + 1] = tmp[3]
                As[(colA4 + 4) * BM + rowA + 1] = tmp[4]
            end
        end

        # --- Gmem to Smem: Load B tile (with float4) ---
        for offset in 0:rowStrideB:BK-1
            rowB = innerRowB + offset
            colB4 = innerColB * 4
            if rowB < BK && colB4 + 3 < N
                global_idx_b = rowB * N + colB4
                tmp = unsafe_load(reinterpret(Ptr{NTuple{4, Float32}}, B_ptr + global_idx_b * sizeof(Float32)))
                # Store to shared memory (no transposition)
                Bs[rowB * BN + colB4 + 1] = tmp[1]
                Bs[rowB * BN + colB4 + 2] = tmp[2]
                Bs[rowB * BN + colB4 + 3] = tmp[3]
                Bs[rowB * BN + colB4 + 4] = tmp[4]
            end
        end
        
        # Synchronize to ensure all tiles are loaded before computation
        sync_threads()

        # --- Compute from Smem to Registers ---
        for dotIdx in 0:BK-1
            # Load M-dimension fragment for this warp into registers
            for wSubRowIdx in 0:WMITER-1
                for i in 0:TM-1
                    as_idx = (dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i) + 1
                    regM[wSubRowIdx * TM + i + 1] = As[as_idx]
                end
            end

            # Load N-dimension fragment for this warp into registers
            for wSubColIdx in 0:WNITER-1
                for i in 0:TN-1
                    bs_idx = (dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i) + 1
                    regN[wSubColIdx * TN + i + 1] = Bs[bs_idx]
                end
            end

            # Perform the outer product of the register fragments
            for wSubRowIdx in 0:WMITER-1
                for wSubColIdx in 0:WNITER-1
                    for resIdxM = 0:TM-1
                        m_val = regM[wSubRowIdx * TM + resIdxM + 1]
                        for resIdxN = 0:TN-1
                            result_idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN + 1
                            threadResults[result_idx] += m_val * regN[wSubColIdx * TN + resIdxN + 1]
                        end
                    end
                end
            end
        end
        
        # Synchronize before loading the next tiles
        sync_threads()
        
        # Advance pointers to the next K-tile
        A_ptr += BK * sizeof(Float32)
        B_ptr += BK * N * sizeof(Float32)
    end

    # === Write Results from Registers back to Global Memory ===
    for wSubRowIdx = 0:WMITER-1
        for wSubColIdx = 0:WNITER-1
            # This thread's contribution to the C matrix starts here
            base_row = by * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM
            base_col = bx * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN
            
            # Write out the thread's TMxTN tile
            for resIdxM = 0:TM-1
                row = base_row + resIdxM
                if row < M
                    for resIdxN = 0:TN-1
                        col = base_col + resIdxN
                        if col < N
                            # Calculate the index into our register file accumulator
                            result_idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN + 1
                            # Perform the alpha*C + beta*A*B update
                            old_val = C[row + 1, col + 1]
                            C[row + 1, col + 1] = alpha * threadResults[result_idx] + beta * old_val
                        end
                    end
                end
            end
        end
    end

    return
end


# High-level interface function
function sgemm_warptiling!(C, A, B, alpha=1.0f0, beta=0.0f0)
    M, N = size(C)
    K = size(A, 2)
    
    # Launch configuration
    BM, BN, BK = 128, 128, 16
    block_size = 256
    
    # Calculate grid dimensions (use `cld` for ceiling division)
    blocks = (cld(N, BN), cld(M, BM))
    threads = block_size
    
    # Calculate shared memory size for both A and B tiles
    shared_mem_size = (BM * BK + BK * BN) * sizeof(Float32)
    
    # Launch the kernel using the correct CUDA.jl syntax
    @cuda threads=threads blocks=blocks shmem=shared_mem_size sgemm_warptiling_kernel(
        A, B, C, Float32(alpha), Float32(beta)
    )
    
    return C
end

# === Test and Benchmark Functions ===

# Test the implementation for correctness
function test_warptiling_sgemm()
    M, N, K = 512, 512, 512  # Use a reasonably small size for testing
    
    # Create test matrices
    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)
    C_ref = CUDA.zeros(Float32, M, N)
    
    α = 1.234f0
    β = 0.567f0
    
    # Compute reference using the built-in (and highly optimized) BLAS
    mul!(C_ref, A, B, α, β)
    CUDA.synchronize()
    
    # Compute using our kernel
    sgemm_warptiling!(C, A, B, α, β)
    CUDA.synchronize()
    
    # Compare results
    max_error = Array(maximum(abs.(C - C_ref)))
    println("Maximum error: ", max_error)
    
    return max_error < 1e-3
end

# Performance benchmark
function benchmark_warptiling_sgemm()
    M, N, K = 2048, 2048, 2048  # Larger size for benchmarking
    
    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)
    
    # Warmup run to eliminate initialization overhead
    sgemm_warptiling!(C, A, B)
    CUDA.synchronize()
    
    # Benchmark
    println("Benchmarking warptiling SGEMM...")
    t = @elapsed for i in 1:10
        sgemm_warptiling!(C, A, B)
        CUDA.synchronize()
    end
    t_avg = t / 10
    
    gflops = (2.0 * M * N * K) / (t_avg * 1e9)
    println("Average time: ", t_avg, " seconds")
    println("Performance: ", gflops, " GFLOPS")
    
    return gflops
end


# --- Run the tests ---
println("Testing warptiling SGEMM...")
# @testset "SGEMM Correctness" begin
    # @test 
    test_warptiling_sgemm()
# end

println("\nBenchmarking warptiling SGEMM...")
benchmark_warptiling_sgemm()