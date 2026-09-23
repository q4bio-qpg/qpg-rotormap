# ==============================================================================
# ropeperiod.jl -- INNER-PRODUCT DECAY of the s5m1c5 rope encoding under
# Utils.mutate noise, on a PERIODIC fragment.
#
# SETUP
#   * fragment: k = 20,000 bases with PERIOD P = 4,002 -- a random 4,002-mer
#     motif (seed 42) tiled 4.99x to exactly k (the last copy is cut off);
#   * encoder: RopeEncoder(k = 20_000, s = 5, m = 1, c = 5) -- the production
#     config of the s5m1c5 index sweep; the COMPLEX histogram (nbin = 4^5 =
#     1,024 bins, G-bin zeroed, each of the k - s + 1 = 19,996 s-mers adding
#     exp(+i*2pi*p/k)) is normalized to unit complex energy (normalize = 0,
#     the flow's mode) and shipped as the [Re; Im] real vector of length
#     2,048 -- here via the float64 CPU reference _ref_rope_frag_real
#     (ropeflowreal.jl), the same math the GPU kernel + index save mirror;
#   * noise: Utils.mutate(frag, err) -- the sampler's exact mutation model
#     (err = edit-OPERATION fraction: 1/3 insertions, 1/3 deletions, 1/3
#     substitutions, ceil(k*err) ops total, length kept exactly k);
#     ONE mutated fragment per err, child rng seeded deterministically;
#   * score: the real dot product of the two unit-energy [Re; Im] vectors,
#     i.e. Re(<z0, zmut>) -- exactly what the fp8 GEMM + top-1 computes.
#     Also printed: |<z0, zmut>| (complex modulus -- phase-rotation tolerant).
#
# Run:  julia --project=. test/ropeperiod.jl
#
# RESULTS (kau, julia -t 8, fragment seed 42; err 0.00 row = the fragment
# itself, must be 1.0):
#
#   err   Re<IP>     |IP|        err   Re<IP>     |IP|
#   0.00  1.000000   1.000000    0.26  -0.028885   0.051100
#   0.01  0.044436   0.045367    0.27  -0.015345   0.031869
#   0.02  0.040190   0.043040    0.28   0.014189   0.042637
#   0.03  0.075496   0.075496    0.29  -0.032966   0.039422
#   0.04  0.083003   0.083056    0.30   0.021402   0.053422
#   0.05 -0.021400   0.021790    0.31   0.034600   0.035843
#   0.06  0.037029   0.061808    0.32   0.011095   0.024769
#   0.07  0.025153   0.047196    0.33   0.033101   0.046135
#   0.08  0.029801   0.030006    0.34   0.006016   0.014987
#   0.09  0.007541   0.061845    0.35  -0.001827   0.032437
#   0.10  0.065349   0.070996    0.36  -0.050564   0.052081
#   0.11  0.023016   0.027515    0.37  -0.003823   0.046587
#   0.12  0.007918   0.025456    0.38   0.008195   0.020074
#   0.13  0.018234   0.033639    0.39  -0.020261   0.021310
#   0.14  0.009800   0.010164    0.40  -0.030180   0.033487
#   0.15  0.039327   0.039475    0.41   0.036447   0.037680
#   0.16 -0.002624   0.010602    0.42   0.014994   0.078958
#   0.17  0.013177   0.036758    0.43  -0.025439   0.028326
#   0.18  0.053306   0.059401    0.44   0.007534   0.050464
#   0.19 -0.028798   0.046365    0.45  -0.002247   0.011000
#   0.20 -0.015104   0.023175    0.46  -0.011323   0.039613
#   0.21  0.020509   0.030896    0.47  -0.009526   0.023395
#   0.22 -0.004276   0.009149    0.48  -0.016516   0.071570
#   0.23  0.031059   0.050405    0.49  -0.005269   0.026340
#   0.24  0.030266   0.047591    0.50   0.013820   0.025372
#   0.25  0.031893   0.037396
#
# INTERPRETATION (verified by a probe): the inner product collapses to noise
# (~0.02-0.08) ALREADY at err = 0.01 -- a property of THIS period choice, not
# of the encoder.  P = 4,002 is adversarial for k = 20,000: the rope phase
# advance per repeat copy is 2pi*P/k = 72.036 deg, so the 5 copies (5P =
# 20,010 = k + 10) form a near-regular PENTAGON and their phasors cancel:
# |sum_t exp(i*2pi*P*t/k)| = 0.0027.  The pristine fragment's histogram is
# therefore NOT dominated by its periodic bulk but by its ~100 aperiodic
# "defect" windows (the 5 motif-wrap boundaries x 4 straddling windows x 5
# copies + the partial 5th copy): norm 3.70, median |bin| 0.0099, but max
# |bin| ~ 0.995.  Any mutation adds hundreds-to-thousands of O(1) defect
# phasors, swamping that tiny defect-only reference -> the mutated and
# pristine histograms share only noise.  CONTROL on a RANDOM (genome-like)
# fragment, same encoder/sweep: norm 138.6 (coherent bulk, median |bin|
# 3.55), Re<IP> decays SMOOTHLY -- 0.956 @ 0.01, 0.802 @ 0.05, 0.644 @ 0.10,
# 0.419 @ 0.20, 0.279 @ 0.30 (8 reps each) -- consistent with the e2ehuman
# sweeps (top-1 ~98% to err 0.2-0.3, collapsed at 0.5).  The mapping flow's
# robustness lives in the coherent bulk + the dense kstep grid, neither of
# which a single self-cancelling periodic fragment exercises.
# ==============================================================================

using Random
using LinearAlgebra
using Printf
using RotorMap
using RotorMap.Utils # mutate
include(joinpath(@__DIR__, "ropeflowreal.jl")) # _ref_rope_frag_real (the
# float64 CPU reference encoder: complex histogram -> normalize -> [Re; Im])

# ---- configuration -----------------------------------------------------------
const RP_K = 20_000   # fragment length (== the encoder's k)
const RP_P = 4_002    # motif period
const RP_S, RP_M, RP_C = 5, 1, 5 # the s5m1c5 encoder config
const RP_SEED = 42    # motif + mutation rng seed
const RP_ERRS = 0.01:0.01:0.50

# random P-mer motif tiled to exactly K bases (last copy cut off)
function periodic_fragment(k::Int, p::Int, seed::Int)
    rng = Xoshiro(seed)
    motif = [UInt8(rand(rng, 0x00:0x03)) for _ in 1:p]
    return UInt8[motif[(i - 1) % p + 1] for i in 1:k]
end

function main()
    re = RopeEncoder(k = RP_K, s = RP_S, m = RP_M, c = RP_C)
    frag = periodic_fragment(RP_K, RP_P, RP_SEED)
    e0, n0 = _ref_rope_frag_real(frag, re; normalize = 0)
    @info "periodic fragment" k = RP_K period = RP_P s = RP_S m = RP_M c = RP_C
    @info "reference encoding" norm0 = n0[1] energy = dot(e0, e0)

    @printf("%5s  %12s  %12s\n", "err", "Re<IP>", "|IP|")
    @printf("%5.2f  %12.6f  %12.6f\n", 0.0, dot(e0, e0), abs(dot(e0, e0)))
    for (i, err) in enumerate(RP_ERRS)
        mut = Utils.mutate(frag, err; rng = Xoshiro(RP_SEED + i))
        em, _ = _ref_rope_frag_real(mut, re; normalize = 0)
        ipc = dot(e0, em) # real dot == Re of the complex inner product
        @printf("%5.2f  %12.6f  %12.6f\n", err, ipc,
                abs(ComplexF64(ipc, dot(@view(e0[1:1024]), @view(em[1025:2048])) -
                               dot(@view(e0[1025:2048]), @view(em[1:1024])))))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
