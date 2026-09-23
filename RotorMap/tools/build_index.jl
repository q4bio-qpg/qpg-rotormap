# =============================================================================
# build_index.jl -- the standalone ENTRY SCRIPT for the index flow: includes
# the canonical prefix that index/build.jl REQUIRES (a pure layer file since
# the workbench reorganize -- running it directly dies with
# `UndefVarError: RopeEncoder` in run_save) and forwards its mode dispatch
# verbatim.  All INDEX_* env overrides of build.jl apply unchanged.
#
# Usage (kau, repo root):
#
#   julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save [fasta]
#   julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl verify [fasta] [--bin8]
#   julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl test|bench|gen|all
#
# e.g. the maize index at k = 1000, kstep = 100, s4m1c4 (rdim 512, fp8
# mapping layout):
#
#   INDEX_FASTA=<fna> INDEX_K=1000 INDEX_KSTEP=100 \
#   INDEX_S=4 INDEX_M=1 INDEX_C=4 INDEX_FP8BIN=<out.bin> \
#   julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save
#
# Includes (build.jl's header REQUIRES, canonical order): treearrays (dna.jl
# dep), dna (save/load), testref, util, fasta pack/loader/reader, encode
# ropeencoder/encoder_v3/reference/kernel/stream, gemm/fp8_convert, then
# index/build.jl itself; finally index/load.jl + common/harness.jl — the
# test mode's window checker uses harness.jl's _ref_forward_pack (harness
# REQUIRES index/load.jl for its format tags; runtime mapping calls are
# never exercised here).  NO search engines: not co-includable.
# =============================================================================

inc(p...) = include(joinpath(@__DIR__, "..", p...))

inc("common", "treearrays.jl")
inc("common", "dna.jl")
inc("common", "testref.jl")
inc("common", "util.jl")
inc("fasta", "pack.jl")
inc("fasta", "loader.jl")
inc("fasta", "reader.jl")
inc("encode", "ropeencoder.jl")
inc("encode", "encoder_v3.jl")
inc("encode", "reference.jl")
inc("encode", "kernel.jl")
inc("encode", "stream.jl")
inc("gemm", "fp8_convert.jl")
inc("index", "build.jl")
inc("index", "load.jl")
inc("common", "harness.jl") # last: the test mode's _ref_forward_pack# mode dispatch (build.jl's bottom block, verbatim -- its @__FILE__ guard
# fires only when build.jl is run directly, which the reorganized tree no
# longer supports)
posargs = filter(a -> !startswith(a, "--"), ARGS)
mode = isempty(posargs) ? "all" : posargs[1]
fasta2 = length(posargs) > 1 ? posargs[2] : IF_FASTA
mode == "gen" && _ensure_test_fasta()
mode == "test" && run_rope_index_test()
mode == "save" && run_save(fasta = fasta2)
mode == "verify" && run_verify(fasta = fasta2,
                               binfile = "--bin8" in ARGS ? _if_fp8file(fasta2) :
                                         _if_binfile(fasta2))
mode == "bench" && bench_index(fasta = fasta2)
mode == "all" && (_ensure_test_fasta(); run_rope_index_test(); isfile(IF_FASTA) && bench_index())
mode in ("gen", "test", "save", "verify", "bench", "all") ||
    error("unknown mode $mode (use gen|test|save|verify|bench|all)")
