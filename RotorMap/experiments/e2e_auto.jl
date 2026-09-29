# ==============================================================================
# e2e_auto.jl -- END-TO-END MAPPING, AUTO NUMERIC STACK: one entry point over
# the two compact-flow variants, picked from the GPU at startup --
#   cc >= 8.9 (Ada RTX 40xx, Hopper, Blackwell RTX 50xx) -> the FP8 flow
#       (experiments/e2e_compact.jl verbatim: e4m3 database, cuBLASLt/mma
#       GEMM, w = 2^16);
#   anything older -> the FP16 flow (experiments/e2e_compact16.jl verbatim:
#       fp16 cuBLAS, host fp16 database, fp8-layout indexes widened on load,
#       w = 2^15).
# The decision MUST precede the includes: search/engine_fp8.jl and
# search/engine_fp16.jl are NOT co-includable (the one-variant rule, see
# RotorMap/ARCHITECTURE.md) -- the same stage-0 dispatch selftest.jl uses.  And
# exactly like selftest.jl, the whole variant script is included WHOLESALE:
# it owns its canonical layered include list AND its run entry points
# (run_e2e_compact{,16}{,_test}; its own main() stays dormant -- the
# PROGRAM_FILE guard only fires for its own file).
#
# Everything else follows whichever variant was picked, exactly as if its
# script had been run:
#   modes       test / run / all (default all) -- identical semantics
#   flags       --k=20  --batch=8192  --segs=8  --resident=auto|on|off
#               --w=65536 (fp8) | 32768 (fp16) -- each variant's optimum
#               --engine=lt|mma on the fp8 path ONLY (rejected on fp16: the
#               fp16 cuBLAS flow has no engine choice)
#   env         the unified E2AUTO_INDEX / E2AUTO_READS applies to whichever
#               variant was picked (they override it); the variant's own
#               E2COMPACT_INDEX / E2COMPACT_READS (fp8) or
#               E2COMPACT16_INDEX / E2COMPACT16_READS (fp16) still work too,
#               preamble-of-origin style -- each variant's consts own their
#               defaults -- the maize too-big-for-VRAM index vs the human
#               k20000 index
# A one-line @info right after the include names the picked stack and the
# env prefix in effect, so a misread default cannot go unnoticed.
#
# USAGE
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_auto.jl test
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_auto.jl run --resident=off
#   julia --project=RotorMap -t 8 RotorMap/experiments/e2e_auto.jl run --engine=mma  # fp8 GPUs
# ==============================================================================

using CUDA # stage 0 needs the device; the layered stack re-`using`s in util.jl

# ------------------------------------------------------------------------------
# stage 0: hardware detection -> numeric path (BEFORE the includes; verbatim
# selftest.jl's rule: e4m3 fp8 tensor cores exist on sm_89, sm_90, sm_120+,
# everything older takes the fp16 cuBLAS path)
# ------------------------------------------------------------------------------
const _E2AUTO_FP8 = let
    CUDA.functional() ||
        error("no functional CUDA device -- the compact mapping flows are GPU-only")
    cap = CUDA.capability(device())
    cap.major > 8 || (cap.major == 8 && cap.minor >= 9)
end

# the picked variant, wholesale: full layered stack (canonical order) + its
# run entry points + env consts; identical re-includes of layers are no-ops
# on julia >= 1.12, and the variant's own main() is NOT run (PROGRAM_FILE)
inc(p...) = include(joinpath(@__DIR__, p...))
if _E2AUTO_FP8
    inc("e2e_compact.jl")
else
    inc("e2e_compact16.jl")
end

@info "e2e_auto: numeric stack picked from the GPU (before the includes)" stack =
    _E2AUTO_FP8 ? "fp8 (e2e_compact flow)" : "fp16 (e2e_compact16 flow)"

# ------------------------------------------------------------------------------
# Unified data paths: one env pair for auto mode, overriding the picked
# variant's consts (E2C_INDEX/E2C_READS vs E2C16_INDEX/E2C16_READS) when set;
# unset, the variant's own default stands (its own E2COMPACT* env vars keep
# working, at a lower precedence than the E2AUTO ones).  Redefining a `const`
# binding is legal on julia >= 1.12.
# ------------------------------------------------------------------------------
if _E2AUTO_FP8
    const E2C_INDEX = get(ENV, "E2AUTO_INDEX", E2C_INDEX)
    const E2C_READS = get(ENV, "E2AUTO_READS", E2C_READS)
else
    const E2C16_INDEX = get(ENV, "E2AUTO_INDEX", E2C16_INDEX)
    const E2C16_READS = get(ENV, "E2AUTO_READS", E2C16_READS)
end

# ------------------------------------------------------------------------------
# Library dispatch: the picked variant's run entry points under fixed names
# (the underlying run_e2e_compact{,16}{,_test} remain available too)
# ------------------------------------------------------------------------------
run_e2e_auto_test(; kw...) =
    _E2AUTO_FP8 ? run_e2e_compact_test(; kw...) : run_e2e_compact16_test(; kw...)
run_e2e_auto(; kw...) =
    _E2AUTO_FP8 ? run_e2e_compact(; kw...) : run_e2e_compact16(; kw...)

# ------------------------------------------------------------------------------
# Mode dispatch -- both parents' mains are identical except the --w default
# (2^16 fp8 vs 2^15 fp16) and the --engine flag (fp8 only); `test`/`run` take
# the same kwargs within each variant
# ------------------------------------------------------------------------------
function main()
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    ktop = parse(Int, getflag("k", "20"))
    w = parse(Int, getflag("w", string(_E2AUTO_FP8 ? 2^16 : 2^15)))
    batch = parse(Int, getflag("batch", string(2^13)))
    segs = parse(Int, getflag("segs", "8"))
    resident = let r = getflag("resident", "auto")
        r == "auto" ? nothing : r == "on" ? true : r == "off" ? false :
        error("bad --resident=$r (use auto|on|off)")
    end
    engine = Symbol(getflag("engine", "lt"))
    kw = _E2AUTO_FP8 ?
         (; ktop, w, batch_size = batch, segs, engine, resident) :
         begin # the fp16 flow has no engine choice; reject the flag outright
             getflag("engine", "") != "" && error("--engine is an fp8-path flag " *
                                                 "(the fp16 flow has no engine choice)")
             (; ktop, w, batch_size = batch, segs, resident)
         end
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    if mode == "test"
        run_e2e_auto_test(; kw...)
    elseif mode == "run"
        run_e2e_auto(; kw...)
    elseif mode == "all"
        run_e2e_auto_test(; kw...)
        run_e2e_auto(; kw...)
    else
        error("unknown mode $mode (use test|run|all)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
