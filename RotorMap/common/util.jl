# =============================================================================
# util.jl -- shared timing / system helpers, deduplicated from the legacy
# experiment scripts (each used to carry its own copy):
#   _timed_min      min-of-reps wall/GC/alloc measurement with GB/s printout
#                   (legacy newfasta_v2.jl)
#   _warm_cache     page a file into the page cache (legacy newfasta_v2.jl)
#   _majflt         major-fault counter via getrusage (legacy newfasta_v2.jl)
#   _fadvise_drop!  POSIX_FADV_DONTNEVE drop of a file's page cache (legacy newfasta_v2.jl)
#   _timed_gpu      CUDA-synchronized steady-state GPU timing (was duplicated,
#                   identically, in legacy flowtopkfp8.jl and e2ecomplex.jl)
#   _comma          integer thousands-separator (legacy nruns.jl and others)
#   _SINK           result sink so benchmarks cannot be optimized away
# NO dependencies (only Base/Printf/CUDA).
# =============================================================================
using Printf
using CUDA

const _SINK = Ref(0)

function _timed_min(f, label::String; bytes::Int, reps::Int)
    best_t = Inf; best_gc = 0.0; best_alloc = 0
    for r in 1:reps
        GC.gc(); GC.gc()
        g0 = Base.gc_num()
        t0 = time_ns()
        out = f()
        dt = (time_ns() - t0) / 1e9
        g1 = Base.gc_num()
        r == 1 && (_SINK[] += out) # keep/observe the result
        if dt < best_t
            best_t = dt
            best_gc = (g1.total_time - g0.total_time) / 1e9
            best_alloc = g1.total_allocd - g0.total_allocd
        end
    end
    @printf("  %-48s %7.3f s  %7.2f GB/s  alloc %8.1f MiB  gc %5.1f%%\n",
            label, best_t, bytes / best_t / 1e9, best_alloc / 2^20, 100 * best_gc / best_t)
    return best_t
end

function _warm_cache(file)
    open(file, "r") do io
        buf = Vector{UInt8}(undef, 1 << 22)
        sz = filesize(file)
        while !eof(io)
            nb = min(length(buf), sz - position(io))
            read!(io, view(buf, 1:nb))
        end
    end
    return nothing
end

function _majflt()
    v = Vector{Clong}(undef, 18)
    ccall(:getrusage, Cint, (Cint, Ptr{Clong}), 0, v)
    return Int(v[10]) # ru_majflt (2+2 timevals, maxrss, ixrss, idrss, isrss, minflt)
end

function _fadvise_drop!(file)
    io = open(file, "r")
    fd = Base.fd(io)
    ccall(:posix_fadvise, Cint, (Cint, Clonglong, Clonglong, Cint), fd, 0, 0, 4) # DONTNEED
    close(io)
    return nothing
end

function _timed_gpu(f, iters::Integer)
    CUDA.device_synchronize()
    t0 = time_ns()
    for _ in 1:iters
        f()
    end
    CUDA.device_synchronize()
    return (time_ns() - t0) / 1e9 / iters
end

# benchmark repetition count (legacy newfasta_v2.jl; ENV-overridable)
_reps() = parse(Int, get(ENV, "NEWFASTA_V2_REPS", "3"))

function _comma(n::Integer)
    s = string(n)
    out = IOBuffer()
    for (i, c) in enumerate(s)
        print(out, c)
        from_end = length(s) - i
        (from_end > 0 && from_end % 3 == 0) && print(out, ',')
    end
    return String(take!(out))
end
