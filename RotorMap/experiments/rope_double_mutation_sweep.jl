# ==============================================================================
# rope_double_mutation_sweep.jl -- inner-product decay under DOUBLY mutated
# copies: one fixed 0.3 pre-mutation, then the 1000-point error sweep on top.
#
# PROBLEM
#   rope_mutation_sweep.jl measures the pristine-vs-mutated decay.  Here the
#   fragment is pre-damaged ONCE (a single mutate at err = 0.3), and the same
#   1000-point sweep runs ON TOP of the pre-mutated sequence -- while the
#   scores are still taken against the PRISTINE encoding.  Question: how much
#   similarity survives when 0.3 worth of edit ops is already spent before the
#   sweep's err_i is added on top.
#
# SETUP
#   * fragment: k = 20,000 random ACGT bases (seed 42, generate_reference);
#   * pre-mutation: pre = mutate(frag, 0.3; rng = Xoshiro(42)) -- ONE fixed
#     draw for the whole sweep (ceil(k*0.3) = 6,000 edit ops split 1/3 sub,
#     1/3 ins, 1/3 del; length kept at k);
#   * sweep: mutation i = mutate(pre, err_i; rng = Xoshiro(42 + i)) with
#     err_i = 0.5*(i-1)/999 -- identical mechanics to rope_mutation_sweep.jl
#     ("as before"), just applied to `pre` instead of the pristine fragment;
#   * encoder: RopeEncoder(k = 20,000, s = 5, m = 1, c = 5) -- the production
#     s5m1c5 config; float64 CPU reference _ref_rope_frag_real
#     (encode/reference.jl), normalize = 0 (unit complex energy);
#   * scores: rope_inner against the PRISTINE encoding e0 -- Re<z0, zmut>
#     (the GEMM/search statistic) and the complex modulus |<z0, zmut>|.
#
# OUTPUT (experiments/out/, gitignored; copied back to the dev checkout)
#   rope_double_mutation_sweep.csv   err,ip_re,ip_abs -- err is the SECOND
#                                    mutation's rate; the err = 0 row is the
#                                    pre-mutation-only anchor
#   rope_double_mutation_sweep.svg   same swapped-axes layout as
#                                    rope_mutation_sweep (IP on x, err on y),
#                                    x domain tightened to the data
#
# NOTES
#   * includes rope_mutation_sweep.jl wholesale: the include chain (TreeArrays,
#     dna.jl's mutate/generate_reference, encode/reference.jl), rope_inner and
#     the SVG writer; that script's main() is PROGRAM_FILE-guarded.
#   * the pre-mutation is a single fixed draw, so the plot shows ONE channel
#     through composition space, not an average over pre-mutations.
#
# Run:  julia --project=. RotorMap/experiments/rope_double_mutation_sweep.jl
# ==============================================================================

include(joinpath(@__DIR__, "rope_mutation_sweep.jl")) # chain + rope_inner + SVG writer

using Random
using LinearAlgebra
using Printf

# ---- configuration -----------------------------------------------------------
const RDMS_K = 20_000                  # fragment length (== the encoder's k)
const RDMS_S, RDMS_M, RDMS_C = 5, 1, 5 # the s5m1c5 encoder config
const RDMS_N = 1000                    # mutated copies (== sweep resolution)
const RDMS_PRE = 0.3                   # the single pre-mutation's error rate
const RDMS_SEED = 42                   # fragment + mutation rng seeds
const RDMS_OUT = joinpath(@__DIR__, "out")

function main()
    mkpath(RDMS_OUT)
    re = RopeEncoder(k = RDMS_K, s = RDMS_S, m = RDMS_M, c = RDMS_C)
    frag = generate_reference(RDMS_K; seed = RDMS_SEED)
    e0, _ = _ref_rope_frag_real(frag, re; normalize = 0)

    pre = mutate(frag, RDMS_PRE; rng = Xoshiro(RDMS_SEED)) # the single 0.3 hit
    ep, _ = _ref_rope_frag_real(pre, re; normalize = 0)
    pre_re, pre_abs = rope_inner(e0, ep)

    errs = range(0.0, 0.5; length = RDMS_N)
    ipre = Vector{Float64}(undef, RDMS_N)
    ipabs = Vector{Float64}(undef, RDMS_N)

    @info "double sweep" k = RDMS_K s = RDMS_S m = RDMS_M c = RDMS_C n = RDMS_N pre_err = RDMS_PRE seed = RDMS_SEED
    @printf("%5s  %12s  %12s\n", "err", "Re<IP>", "|IP|")
    @printf("%5.3f  %12.6f  %12.6f  (0.3 pre-mutation anchor, no 2nd mutation)\n",
            RDMS_PRE, pre_re, pre_abs)
    t = @elapsed Threads.@threads for i in 1:RDMS_N
        mut = mutate(pre, errs[i]; rng = Xoshiro(RDMS_SEED + i))
        em, _ = _ref_rope_frag_real(mut, re; normalize = 0)
        ipre[i], ipabs[i] = rope_inner(e0, em)
    end
    @info "sweep done" seconds = round(t; digits = 1)

    for i in 1:50:RDMS_N # every 50th point + the err = 0.5 endpoint
        @printf("%5.3f  %12.6f  %12.6f\n", errs[i], ipre[i], ipabs[i])
    end
    @printf("%5.3f  %12.6f  %12.6f\n", errs[end], ipre[end], ipabs[end])

    csv = joinpath(RDMS_OUT, "rope_double_mutation_sweep.csv")
    open(csv, "w") do io
        println(io, "err,ip_re,ip_abs")
        for i in 1:RDMS_N
            println(io, errs[i], ",", ipre[i], ",", ipabs[i])
        end
    end

    svg = joinpath(RDMS_OUT, "rope_double_mutation_sweep.svg")
    write_sweep_svg(svg, collect(errs), ipre, ipabs;
                    xmax = ceil(1.05 * maximum(ipabs) * 20) / 20,
                    title = "RoPE-DNA inner product after a 0.3 pre-mutation vs further mutation error rate (s5m1c5)",
                    subtitle = "random k=20,000 fragment &#183; one 0.3 mutation (Re = $(round(pre_re; digits = 3)), |&#183;| = $(round(pre_abs; digits = 3))), then 1000 further mutations, err evenly spaced on [0.0, 0.5] &#183; seed $(RDMS_SEED)",
                    legend_text = "doubly mutated copy (N=1000)")

    @info "outputs" csv svg
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
