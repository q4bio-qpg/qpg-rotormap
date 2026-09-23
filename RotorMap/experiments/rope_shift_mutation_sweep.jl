# ==============================================================================
# rope_shift_mutation_sweep.jl -- inner-product decay under SHIFT-premutated
# copies: one fixed 0.05 shift pre-mutation, then the 1000-point error sweep
# on top.
#
# PROBLEM
#   rope_double_mutation_sweep.jl pre-damages the fragment with an ordinary
#   0.3 sub/ins/del mutation.  Here the pre-mutation is instead a pure SHIFT
#   (mutate_shift at err = 0.05 -> floor(k*0.05/2) = 500 bases: the head is
#   dropped, a random tail is appended, length kept at k).  A pure shift only
#   rotates the rope phases (every s-mer moves by the same offset), which the
#   encoding is supposed to tolerate -- the modulus |<z0, zpre>| should stay
#   high while the real part pays the fixed cos(2*pi*shift/k) rotation.  The
#   sweep then measures how the shifted fragment decays under ordinary noise.
#
# SETUP
#   * fragment: k = 20,000 random ACGT bases (seed 42, generate_reference);
#   * pre-mutation: pre = mutate_shift(frag, 0.05; rng = Xoshiro(42)) -- ONE
#     fixed draw for the whole sweep (shift = 500 bases);
#   * sweep: mutation i = mutate(pre, err_i; rng = Xoshiro(42 + i)) with
#     err_i = 0.5*(i-1)/999 -- identical to rope_mutation_sweep.jl /
#     rope_double_mutation_sweep.jl, applied to the shifted `pre`;
#   * encoder: RopeEncoder(k = 20,000, s = 5, m = 1, c = 5) -- the production
#     s5m1c5 config; float64 CPU reference _ref_rope_frag_real
#     (encode/reference.jl), normalize = 0 (unit complex energy);
#   * scores: rope_inner against the PRISTINE encoding e0 -- Re<z0, zmut>
#     (the GEMM/search statistic) and the complex modulus |<z0, zmut>|.
#
# OUTPUT (experiments/out/, gitignored; copied back to the dev checkout)
#   rope_shift_mutation_sweep.csv   err,ip_re,ip_abs -- err is the SECOND
#                                   mutation's rate; the err = 0 row is the
#                                   shift-only anchor
#   rope_shift_mutation_sweep.svg   same swapped-axes layout as
#                                   rope_mutation_sweep (IP on x, err on y),
#                                   x domain tightened to the data
#
# NOTES
#   * includes rope_mutation_sweep.jl wholesale: the include chain (TreeArrays,
#     dna.jl's mutate/mutate_shift/generate_reference, encode/reference.jl),
#     rope_inner and the SVG writer; that script's main() is
#     PROGRAM_FILE-guarded.
#   * the shift is a single fixed draw, so the plot shows ONE channel through
#     composition space, not an average over shifts.
#
# Run:  julia --project=. RotorMap/experiments/rope_shift_mutation_sweep.jl
# ==============================================================================

include(joinpath(@__DIR__, "rope_mutation_sweep.jl")) # chain + rope_inner + SVG writer

using Random
using LinearAlgebra
using Printf

# ---- configuration -----------------------------------------------------------
const RSMS_K = 20_000                  # fragment length (== the encoder's k)
const RSMS_S, RSMS_M, RSMS_C = 5, 1, 5 # the s5m1c5 encoder config
const RSMS_N = 1000                    # mutated copies (== sweep resolution)
const RSMS_PRE = 0.05                  # the single shift pre-mutation's err
const RSMS_SEED = 42                   # fragment + mutation rng seeds
const RSMS_OUT = joinpath(@__DIR__, "out")

function main()
    mkpath(RSMS_OUT)
    re = RopeEncoder(k = RSMS_K, s = RSMS_S, m = RSMS_M, c = RSMS_C)
    frag = generate_reference(RSMS_K; seed = RSMS_SEED)
    e0, _ = _ref_rope_frag_real(frag, re; normalize = 0)

    pre = mutate_shift(frag, RSMS_PRE; rng = Xoshiro(RSMS_SEED)) # the single 0.05 shift
    @assert length(pre) == RSMS_K
    shift = floor(Int, RSMS_K * RSMS_PRE / 2)
    ep, _ = _ref_rope_frag_real(pre, re; normalize = 0)
    pre_re, pre_abs = rope_inner(e0, ep)

    errs = range(0.0, 0.5; length = RSMS_N)
    ipre = Vector{Float64}(undef, RSMS_N)
    ipabs = Vector{Float64}(undef, RSMS_N)

    @info "shift sweep" k = RSMS_K s = RSMS_S m = RSMS_M c = RSMS_C n = RSMS_N pre_err = RSMS_PRE shift = shift seed = RSMS_SEED
    @printf("%5s  %12s  %12s\n", "err", "Re<IP>", "|IP|")
    @printf("%5.3f  %12.6f  %12.6f  (0.05 shift anchor, no 2nd mutation)\n",
            RSMS_PRE, pre_re, pre_abs)
    t = @elapsed Threads.@threads for i in 1:RSMS_N
        mut = mutate(pre, errs[i]; rng = Xoshiro(RSMS_SEED + i))
        em, _ = _ref_rope_frag_real(mut, re; normalize = 0)
        ipre[i], ipabs[i] = rope_inner(e0, em)
    end
    @info "sweep done" seconds = round(t; digits = 1)

    for i in 1:50:RSMS_N # every 50th point + the err = 0.5 endpoint
        @printf("%5.3f  %12.6f  %12.6f\n", errs[i], ipre[i], ipabs[i])
    end
    @printf("%5.3f  %12.6f  %12.6f\n", errs[end], ipre[end], ipabs[end])

    csv = joinpath(RSMS_OUT, "rope_shift_mutation_sweep.csv")
    open(csv, "w") do io
        println(io, "err,ip_re,ip_abs")
        for i in 1:RSMS_N
            println(io, errs[i], ",", ipre[i], ",", ipabs[i])
        end
    end

    svg = joinpath(RSMS_OUT, "rope_shift_mutation_sweep.svg")
    write_sweep_svg(svg, collect(errs), ipre, ipabs;
                    xmax = ceil(1.05 * maximum(ipabs) * 20) / 20,
                    title = "RoPE-DNA inner product after a 0.05 shift pre-mutation vs further mutation error rate (s5m1c5)",
                    subtitle = "random k=20,000 fragment &#183; one 0.05 shift pre-mutation (500 bases; Re = $(round(pre_re; digits = 3)), |&#183;| = $(round(pre_abs; digits = 3))), then 1000 mutations, err on [0.0, 0.5] &#183; seed $(RSMS_SEED)",
                    legend_text = "shift + mutated copy (N=1000)")

    @info "outputs" csv svg
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
