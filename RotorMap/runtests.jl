# ==============================================================================
# runtests.jl -- the package-less GPU SMOKE TEST (kau, RTX 5090 / sm_120).
#
# Includes the fp8 end-to-end mapping experiment (experiments/e2e_fp8.jl --
# its include list pulls in the whole layered stack: common, fasta, encode,
# gemm fp8, search/engine_fp8, index, reads/provenance, common/harness) and
# runs its tiny self-test `run_e2e_test()` with the tiny defaults:
#
#   a 1M-base synthetic two-record reference (E2H_TEST_BASES = 1e6, generated
#   once by common/harness.jl's fixture builders and cached in the data dir)
#   -> an fp8-format index built by the miniature production save -> 1000
#   err = 0 sampled reads -> the FULL load -> fp8 database -> rope stream ->
#   GEMM+top-k -> provenance-score path.  At err = 0 every read window shares
#   >= 95% of its bases with an index window -- a margin no quantization noise
#   can close -- so the test asserts 1000/1000 correctly mapped (rank
#   histogram dominated by rank 1).
#
# This needs the GPU + the matching driver userland on kau (LD_LIBRARY_PATH
# set up by ~/.bashrc) and single-process runs the production :lt engine.
#
# Usage:
#   julia --project=. -t 16 RotorMap/runtests.jl
# ==============================================================================

include(joinpath(@__DIR__, "experiments", "e2e_fp8.jl"))

# tiny defaults: kfrag = 20,000, n = E2H_TEST_N = 1000 err = 0 reads, ktop 20,
# w capped to the tiny index width, engine :lt (see run_e2e_test's signature)
run_e2e_test()
