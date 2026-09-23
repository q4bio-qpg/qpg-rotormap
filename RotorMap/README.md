# RotorMap — research workbench

Layered library of GPU (CUDA.jl, RTX 5090 / sm_120) experiment code for RoPE-based DNA read
mapping: FASTA parsing, rope encoding, batched GEMM + top-k search engines, reference index
construction, read sampling with provenance, and end-to-end mapping experiments.

The directory layout mirrors the **data flow** of the pipeline — each layer only depends on the
layers above it:

```
 reference.fasta ─┬─► tools/       (fasta utilities: subset, N-run stats, packed/rope .bin conversion)
                  └─► index/       (build: windows + revcomp → rope-real matrix → index .bin;
                                    load: quantize to fp16/fp8 GPU databases; complex split format)
 reads.fasta    ◄── reads/        (sampling with provenance headers; provenance parsing/scoring)
    │
    ├─ fasta/     DNA text → 2-bit packed fragments (parallel reader, batched loader)
    ├─ encode/    packed fragments → rope vectors (GPU kernel + channel stream, CPU golden refs)
    ├─ gemm/      GEMM engines: fp16 cuBLAS, fp8 PTX mma.sync, fp8 cuBLASLt; top-k kernels
    ├─ search/    streaming search engines over a database: fp16 / fp8 / complex / cuts / compact
    └─ experiments/  thin runnable entry points (ENV/ARGS → run_*, mode dispatch)
 common/   shared foundation: vendored Utils subset (mutate/save_fasta/serialization),
           synthetic test data builders, timing utils, tiny-e2e harness
 lab/      low-level CUDA kernel playgrounds (SGEMM warp/tile studies, fp8 cuTile / C++ PTX)
 legacy/   the pre-reorganization code (old package src/, bin/, test/) — frozen reference
```

## Layout

| Path | Contents |
|---|---|
| `common/` | `treearrays.jl` (vendored rope-tree module), `dna.jl` (mutate, save_fasta, save/load, generate_* — the old `Utils.` subset, now bare names), `util.jl` (timing/system helpers), `testref.jl` (synthetic fasta builders), `harness.jl` (tiny-e2e self-test scaffolding) |
| `fasta/` | `pack.jl` (2-bit pack/unpack + GPU pack kernel), `loader.jl` (`FastaBatchesV2` batched loader), `reader.jl` (`fasta_reads` multi-part parallel reader) |
| `encode/` | `ropeencoder.jl` (`RopeEncoder` struct, vendored), `encoder_v3.jl` (v3 batch encoder for packed reads), `reference.jl` (CPU golden encoders), `kernel.jl` (`kernel_rope_frag_real`), `stream.jl` (`rope_encode_real_stream`) |
| `gemm/` | `fp16.jl` (cuBLAS `gemm_topk!`), `fp8_convert.jl` (`F8`, quantization), `fp8_ptx.jl` (PTX mma.sync engine), `fp8_lt.jl` (cuBLASLt engine), `topk_kernels.jl` (row/segmented/complex top-k kernels — single deduplicated copies) |
| `search/` | one file per engine: `engine_fp16.jl`, `engine_fp8.jl`, `engine_complex.jl`, `engine_cuts.jl`, `engine_compact.jl`, `engine_compact16.jl` (the compact ring in fp16, for pre-fp8 GPUs) — `TopKEngine` family + flows + per-engine provenance scoring (`map_and_count` variants) |
| `index/` | `build.jl` (the indexreal builder + `run_save`/`run_verify`), `load.jl` (`load_index`, `index_to_f8/f16`, `index_database`), `complex.jl` (split complex format) |
| `reads/` | `sample.jl` (provenance read sampling), `provenance.jl` (`parse_read_head`) |
| `tools/` | standalone CLI tools: `build_index.jl` (the index-flow entry: includes the prefix `index/build.jl` REQUIRES + its mode dispatch), `fastasubset.jl`, `nruns.jl`, `to_packed_bin.jl`, `to_rope_bin.jl` |
| `experiments/` | `e2e_fp8.jl`, `e2e_fp16.jl`, `e2e_complex.jl`, `e2e_cuts.jl`, `e2e_compact.jl`, `e2e_compact16.jl`, `rope_decay.jl`, `bench_gemm_fp8.jl`, `eval_human1x.jl` (+ `eval_human1x.md`: chrX eval how-to + results) |
| `lab/` | `sgemm_warp.jl`, `sgemm_tiled.jl`, `fp8_cutile.jl`, `fp8_cpp.jl` |

## Include contract

There is **no Julia package** here anymore (the old `RotorMap` package lives in `legacy/src/`).
Code is assembled by plain `include` in a **fixed topological order** — every experiment/tools
file starts with the prefix it needs, e.g.:

```julia
inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("common/treearrays.jl"); inc("common/dna.jl"); inc("common/util.jl"); inc("common/testref.jl")
inc("fasta/pack.jl"); inc("fasta/loader.jl"); inc("fasta/reader.jl")
inc("encode/ropeencoder.jl"); inc("encode/encoder_v3.jl"); inc("encode/reference.jl")
inc("encode/kernel.jl"); inc("encode/stream.jl")
inc("gemm/fp8_convert.jl"); inc("gemm/fp8_ptx.jl"); inc("gemm/fp8_lt.jl"); inc("gemm/topk_kernels.jl")
inc("search/engine_fp8.jl"); inc("index/build.jl"); inc("index/load.jl")
inc("reads/provenance.jl"); inc("common/harness.jl")
```

Rules baked into the layout:

- **Include each file at most once per session** — files define `const`s (LUTs, format tags,
  cuBLASLt handles); re-inclusion would error. The canonical order above satisfies all
  const-to-const dependencies.
- **Never co-include `search/engine_fp16.jl` with `engine_fp8.jl`** (both define
  `TopKEngine`, `topk_flow`, ...). `engine_complex.jl` is likewise not co-includable with the
  fp8/fp16 engines, and `engine_compact16.jl` requires `engine_fp16.jl` (so it must never meet
  the fp8/complex stacks either). Every other combination is fine.
- Layer files never include across layers; the entry script owns the full list.
- `Project.toml` remains for dependency resolution (`julia --project=RotorMap`), but nothing
  `using RotorMap` anymore — the old package modules were moved to `legacy/src/`.

## Running

Run on the GPU host (kau) from the repo root:

```bash
# build the reference index from a fasta (ENV-overridable: INDEX_FASTA, INDEX_K, ...)
julia --project=RotorMap -t 16 RotorMap/tools/build_index.jl save   # via run_save (see index/build.jl)

# end-to-end mapping experiments (mode dispatch at the bottom of each file)
julia --project=RotorMap RotorMap/experiments/e2e_fp8.jl test     # tiny synthetic self-test
julia --project=RotorMap RotorMap/experiments/e2e_fp8.jl human    # full human-genome mapping
julia --project=RotorMap RotorMap/experiments/e2e_fp16.jl test
julia --project=RotorMap RotorMap/experiments/e2e_complex.jl bench
julia --project=RotorMap RotorMap/experiments/e2e_cuts.jl human
julia --project=RotorMap RotorMap/experiments/e2e_compact.jl human

# tools
julia --project=RotorMap RotorMap/tools/fastasubset.jl in.fna out.fna '^NC_000001' ...
julia --project=RotorMap RotorMap/tools/to_packed_bin.jl          # fasta → packed-DNA .bin
julia --project=RotorMap RotorMap/tools/to_rope_bin.jl            # fasta → rope encodings .bin

# GPU smoke test (tiny synthetic e2e)
julia --project=RotorMap RotorMap/runtests.jl
```

The ENV interface is unchanged from the pre-reorganization scripts (`E2HUMAN_*`,
`E2COMPLEX_*`, `E2CUTS_*`, `E2COMPACT_*`, `IF_*`, `IS_*` — see each file's header).

## Legacy

`legacy/` is the frozen pre-reorganization state: `legacy/src/` (the old `RotorMap` package:
RopeEncoders, RopeIndexers, Mapper, RopeScores, SplitComplexMatrices, TreeArrays, Utils),
`legacy/bin/`, and `legacy/test/` (all historical experiment scripts, kept for reference and
git archaeology). It is not included by any new code.
