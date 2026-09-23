# ==============================================================================
# rope_decay.jl -- INNER-PRODUCT DECAY of the s5m1c5 rope encoding under
# mutation noise, on a PERIODIC fragment.
#
# PROBLEM
#   How fast does the flow's similarity statistic decay under the sampler's
#   exact mutation model, and what does a PERIODIC sequence do to it?  (The
#   legacy results: on this adversarial period the inner product collapses to
#   noise already at err = 0.01 -- the near-regular pentagon of the 5 motif
#   copies self-cancels -- while a random genome-like fragment decays
#   smoothly, 0.956 @ 0.01 -> 0.279 @ 0.30.  Full tables: the source file.)
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
#     (encode/reference.jl), the same math the GPU kernel + index save mirror;
#   * noise: mutate(frag, err) (common/dna.jl) -- the sampler's exact mutation
#     model (err = edit-OPERATION fraction: 1/3 insertions, 1/3 deletions,
#     1/3 substitutions, ceil(k*err) ops total, length kept exactly k);
#     ONE mutated fragment per err, child rng seeded deterministically;
#   * score: the real dot product of the two unit-energy [Re; Im] vectors,
#     i.e. Re(<z0, zmut>) -- exactly what the fp8 GEMM + top-1 computes.
#     Also printed: |<z0, zmut>| (complex modulus -- phase-rotation tolerant).
#
# SOURCE: legacy/test/ropeperiod.jl (verbatim; `RotorMap.mutate` is now
#         the bare `mutate` of common/dna.jl; ropeflowreal's CPU reference
#         encoder is now encode/reference.jl).
#
# INCLUDES (canonical order; the full encode chain so the CPU reference and
#   the encoder config it mirrors are the production ones)
#   common/dna.jl, fasta/{pack,loader,reader}.jl,
#   encode/{ropeencoder,encoder_v3,reference,kernel,stream}.jl
#
# Run:  julia --project=. RotorMap/experiments/rope_decay.jl
# ==============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

inc("common", "treearrays.jl") # TreeArrays module (dna.jl's mutate needs it)
inc("common", "dna.jl") # mutate (the sampler's exact mutation model)
inc("fasta", "pack.jl")
inc("fasta", "loader.jl")
inc("fasta", "reader.jl")
inc("encode", "ropeencoder.jl") # RopeEncoder
inc("encode", "encoder_v3.jl")
inc("encode", "reference.jl") # _ref_rope_frag_real (float64 CPU reference)
inc("encode", "kernel.jl")
inc("encode", "stream.jl")

using Random
using LinearAlgebra
using Printf

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
        mut = mutate(frag, err; rng = Xoshiro(RP_SEED + i)) # was mutate
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
