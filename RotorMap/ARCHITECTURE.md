# RotorMap — architecture

Layered workbench of GPU (CUDA.jl, developed on an RTX 5090 / sm_120, Julia 1.12, CUDA
runtime 13.x pinned via `LocalPreferences.toml`) experiment code for RoPE-based DNA read
mapping: FASTA parsing, rope encoding, batched GEMM + top-k search engines, reference index
construction, read sampling with provenance, and end-to-end mapping experiments. There is no
Julia package — code is assembled by plain `include` in a fixed topological order (see
"Include contract"). The user-facing how-to (CLIs, env tables, walkthrough, measured
numbers) lives in `README.md` at the repo root; this file is the structure map.

The directory layout mirrors the **data flow** of the pipeline — each layer only depends on
the layers above it:

```
 reference.fasta ─┬─► tools/gen_dna.jl  (synthetic random reference + reads; CPU-only)
                  ├─► reads/sample.jl   (provenance sampling → >read_i start= len= src= headers)
                  └─► index/            (build: windows + revcomp → rope-real matrix → index .bin,
                                         via tools/build_index.jl; load: quantize to fp16/fp8 GPU
                                         databases; complex split format)
 reads.fasta    ◄── tools/separate_provenance.jl  (provenance-free fasta + ground-truth .tsv)
    │
    ├─ fasta/     DNA text → 2-bit packed fragments (parallel reader, batched loader)
    ├─ encode/    packed fragments → rope vectors (GPU kernel + channel stream, CPU golden refs)
    ├─ gemm/      GEMM engines: fp16 cuBLAS, fp8 PTX mma.sync, fp8 cuBLASLt; top-k kernels
    ├─ search/    streaming engines over the database: fp8 / fp16 TopKEngine, compact ring
    │             (auto VRAM residency), complex, cuts — provenance scoring via map_and_count
    └─ tools/map.jl   THE mapping CLI: index .bin + reads → top-k match table (cc-driven
                      numeric path, auto residency); experiments/ = e2e/eval/probe/sweep entries
 then  tools/verify_maps.jl  scores the match table against the provenance (fasta or .tsv)
 root  selftest.jl (hardware-aware 4-stage self-test) · runtests.jl (fp8 GPU smoke test)
 common/   shared foundation: vendored Utils subset (mutate/save_fasta/serialization),
           synthetic test data builders, timing utils, tiny-e2e harness
 lab/      low-level CUDA kernel playgrounds (SGEMM warp/tile studies, fp8 cuTile / C++ PTX)
 legacy/   the pre-reorganization code (old package src/, bin/, test/) — frozen reference
```

## Layout

| Path | Contents |
|---|---|
| `common/` | `treearrays.jl` (vendored rope-tree module — `dna.jl` requires it included first), `dna.jl` (`mutate`, `mutate_shift`, `save_fasta`, `save`/`load` serialization, `generate_reference`/`generate_reads` — the old `Utils.` subset, now bare names), `util.jl` (timing/system helpers), `testref.jl` (synthetic fasta builders), `harness.jl` (tiny-e2e self-test scaffolding) |
| `fasta/` | `pack.jl` (2-bit pack/unpack + GPU pack kernel), `loader.jl` (`FastaBatchesV2` batched loader), `reader.jl` (`fasta_reads` multi-part parallel reader) |
| `encode/` | `ropeencoder.jl` (`RopeEncoder` struct, vendored), `encoder_v3.jl` (v3 batch encoder for packed reads), `reference.jl` (CPU golden encoders — the probe scripts reconcile GPU scores against these), `kernel.jl` (`kernel_rope_frag_real`, the production kernel), `stream.jl` (`rope_encode_real_stream` channel encoder) |
| `gemm/` | `fp16.jl` (cuBLAS `gemm_topk!`), `fp8_convert.jl` (`F8` e4m3 type + quantization), `fp8_ptx.jl` (PTX `mma.sync` engine — the optional `:mma` engine, needs `nvcc`), `fp8_lt.jl` (cuBLASLt engine — the production `:lt` engine), `topk_kernels.jl` (row/segmented/complex top-k kernels — single deduplicated copies) |
| `search/` | two mutually exclusive engine stacks — `engine_fp8.jl` and `engine_fp16.jl` each define `TopKEngine`, `topk_flow`, `rope_topk_stream` and the provenance-scoring `map_and_count` — plus `engine_compact.jl` / `engine_compact16.jl` (host-RAM databases + triple-buffered device rings, auto residency; the `16` variant requires `engine_fp16.jl` and widens fp8-layout bins for pre-fp8 GPUs), `engine_complex.jl` (`ComplexTopKEngine`, split-complex GEMM), `engine_cuts.jl` (`map_and_count_cuts`, calibrated per-column score cuts) |
| `index/` | `build.jl` (the indexreal builder; ARGS modes `gen|test|save|verify|bench|all` via `run_save`/`run_verify`/`run_rope_index_test`/`bench_index`), `load.jl` (`load_index`, `index_to_f8`/`index_to_f16`, `index_database`/`index_database_f16`; format tags `indexreal.v1` fp16-rows / `indexreal.fp8.v1` transposed-e4m3 columns; legacy `indexflowreal.*` tags still load), `complex.jl` (split complex format) |
| `reads/` | `sample.jl` (provenance read sampler; modes `gen|verify|test|gen!`; `:even`/`:random` draws over the N-free window pool), `provenance.jl` (`parse_read_head` — both formats: sample `>read_<i> start=<1-based> len=<k> src=<record>` and colon eval `>acc:<0-based start>:<rc>`) |
| `tools/` | user-facing CLIs: `map.jl` (**the** mapping CLI: index `.bin` + reads fasta → top-k match table `<read header> <+|-> <score> <0-based pos> <record>`; no provenance required; cc-driven numeric path, auto residency, `-v` verbose budget report), `build_index.jl` (the index-flow entry: includes `index/build.jl`'s REQUIRES + its mode dispatch), `verify_maps.jl` (scores match tables against provenance — fasta headers or `separate_provenance.jl`'s TSV; stdlib-only, no includes), `separate_provenance.jl` (splits a sampled batch into a plain reads fasta + `id src start len` TSV), `gen_dna.jl` (random reference / mutated iid-window reads with provenance headers; CPU-only), `fastasubset.jl`, `nruns.jl`, `maizenfree.jl` (N-free k-window pool geometry report; deliberately self-contained), `to_packed_bin.jl`, `to_rope_bin.jl`, `paf/header_stats.jl` (top-1/any accuracy of minimap2 PAF output whose query names carry `seq:pos:rc` ground truth; stdlib-only) |
| `experiments/` | e2e mapping entries: `e2e_auto.jl` (cc-driven dispatch over the two compact flows; modes `test|run|all`, unified `E2AUTO_*` env), `e2e_compact.jl` (fp8) / `e2e_compact16.jl` (fp16, pre-fp8 GPUs), `e2e_fp8.jl` (always-resident baseline + bench), `e2e_fp16.jl` (fp16 reference accuracy), `e2e_complex.jl` (split-complex GEMM), `e2e_cuts.jl` (calibrated cuts); eval harness `eval_human1x.jl` (+ `eval_human1x.md`: chrX eval how-to + results, warm-rep `EVALRESULT` rows); CPU probes (golden-encoder forensics): `probe_argmax_read.jl` (single-read workup), `probe_locus_windows.jl` (index columns + stored fp8 values around one true locus), `probe_true_locus_scores.jl` (edit% → fidelity curve on real data), `probe_levenshtein.jl` (include-free Levenshtein check of external eval reads), `probe_top1_scores.jl` (the one GPU probe — score distributions + per-rank profile through the compact flow); rope sweeps (write CSV + hand-rolled SVG to gitignored `experiments/out/`): `rope_mutation_sweep.jl` (the shared base — included wholesale by the other two), `rope_double_mutation_sweep.jl` (fixed 0.3 pre-mutation), `rope_shift_mutation_sweep.jl` (pure-shift pre-mutation); studies/bench: `rope_decay.jl`, `bench_gemm_fp8.jl` |
| `lab/` | `sgemm_warp.jl`, `sgemm_tiled.jl`, `fp8_cutile.jl`, `fp8_cpp.jl` |
| root | `selftest.jl` (hardware-aware 4-stage self-test: A random DNA → B production index build → C provenance sampling → D mapping with pass floors; picks the numeric path at include time), `runtests.jl` (minimal GPU smoke test — the fp8 tiny e2e, hardcoded `:lt`) |

## Numeric-path dispatch (an include-time decision)

`engine_fp8.jl` and `engine_fp16.jl` (and the complex/compact stacks) intentionally define
clashing names, so exactly **one** numeric stack is included per session. The selector is
the GPU's compute capability, evaluated **before** any include: cc ≥ 8.9 → the fp8 e4m3
flow (production engine: fixed cuBLASLt `:lt`; the PTX `:mma` engine is optional), anything
older → the fp16 cuBLAS flow. Entry points implementing the dispatch: `selftest.jl` and
`experiments/e2e_auto.jl` (choose between the fp8 and fp16 compact flows) and
`tools/map.jl` (fp8 path always `:lt`). `runtests.jl` is hardcoded fp8 (the kau smoke
test).

## Residency: resident engine vs device ring

The compact engines (`search/engine_compact.jl`, `engine_compact16.jl`) wrap their
`TopKEngine` stack with **auto VRAM residency**: `resident_fits` / `resident_fits16`
compare the database plus the engine's working set against free VRAM. If it fits, the
host-RAM database (`index_database_host`/`_f16`) is uploaded once and the resident engine
runs at full speed; otherwise the database stays in host RAM and is swept through a
triple-buffered device ring, `w` columns at a time — which runs indexes of any size. Both
paths are bitwise identical, and the decision is logged (`tools/map.jl -v` restates it with
the PCIe/VRAM budget). The compact wrappers add no scoring of their own —
`map_and_count` is the inner engine's, verbatim.

## Include contract

There is **no Julia package** here anymore (the old `RotorMap` package lives in `legacy/src/`).
Code is assembled by plain `include` in a **fixed topological order** — every experiment/tools
file starts with the prefix it needs, e.g. the fp8 compact stack (`experiments/e2e_compact.jl`):

```julia
inc(p...) = include(joinpath(@__DIR__, "..", p...))
inc("common", "treearrays.jl"); inc("common", "dna.jl"); inc("common", "util.jl"); inc("common", "testref.jl")
inc("fasta", "pack.jl"); inc("fasta", "loader.jl"); inc("fasta", "reader.jl")
inc("encode", "ropeencoder.jl"); inc("encode", "encoder_v3.jl"); inc("encode", "reference.jl")
inc("encode", "kernel.jl"); inc("encode", "stream.jl")
inc("gemm", "fp8_convert.jl"); inc("gemm", "fp8_ptx.jl"); inc("gemm", "fp8_lt.jl"); inc("gemm", "topk_kernels.jl")
inc("search", "engine_fp8.jl"); inc("search", "engine_compact.jl")   # provides CompactEngine, index_database_host, …
inc("index", "build.jl"); inc("index", "load.jl")
inc("reads", "provenance.jl"); inc("common", "harness.jl")          # last: CALLS index/encode at runtime
```

Rules baked into the layout:

- **Include each file at most once per session** — files define `const`s (LUTs, format tags,
  cuBLASLt handles); re-inclusion would error. `common/dna.jl` requires
  `common/treearrays.jl` first; the canonical order above satisfies all const-to-const
  dependencies.
- **Respect the non-co-includable sets**: `engine_fp16.jl` vs `engine_fp8.jl` (both define
  `TopKEngine`, `topk_flow`, `rope_topk_stream`, `map_and_count`); `engine_complex.jl`
  vs either; `engine_compact.jl` belongs to the fp8 stack, and `engine_compact16.jl`
  requires `engine_fp16.jl` (so it must never meet the fp8/complex stacks either).
- Layer files never include across layers — the entry script owns the full list. Entry
  scripts *may* include another experiment wholesale when they wrap it:
  `e2e_auto.jl` → `e2e_compact.jl` | `e2e_compact16.jl`, `tools/map.jl` → the cc-chosen
  `e2e_compact*.jl`, `selftest.jl` → `e2e_fp8.jl` | `e2e_fp16.jl`,
  `probe_top1_scores.jl` → `e2e_compact.jl`, the sweep pair → `rope_mutation_sweep.jl`.
- A few tools are deliberately include-free and stdlib-only (no RotorMap code at all):
  `tools/verify_maps.jl`, `tools/paf/header_stats.jl`, `tools/maizenfree.jl`,
  `experiments/probe_levenshtein.jl`.
- `Project.toml` remains for dependency resolution (`julia --project=RotorMap`), but nothing
  `using RotorMap` anymore — the old package modules were moved to `legacy/src/`.

## Running

From the repo root on the GPU host (kau). The canonical synthetic loop — no external data
needed (`README.md` §3.4 has the full walkthrough with env tables):

```bash
# 0) hardware-aware self-test (cc ≥ 8.9 ⇒ fp8 stack, else fp16; exit code = pass/fail)
julia --project=RotorMap -t 8 RotorMap/selftest.jl

# 1) a random 2 Mbp reference (CPU-only)
julia --project=RotorMap -t 8 RotorMap/tools/gen_dna.jl ref 2000000 synth_ref.fasta --seed=1

# 2) sample 5000 mutated reads with provenance headers (CPU-only)
SAMPLE_FASTA=synth_ref.fasta SAMPLE_N=5000 \
julia --project=RotorMap -t 8 RotorMap/reads/sample.jl gen

# 3) build the index (ENV-overridable: INDEX_FASTA, INDEX_K, INDEX_KSTEP, INDEX_S/M/C, ...)
julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save

# 4) map (cc-dispatch + auto residency; writes <READS>.maps)
INDEX=synth_ref.indexreal_fp8.bin READS=synth_ref.sample_n5000_k20000_e0.05_s42.fasta \
julia --project=RotorMap -t 8 RotorMap/tools/map.jl -v

# 5) verify the match table against the sampled provenance
julia --project=RotorMap RotorMap/tools/verify_maps.jl \
      synth_ref.sample_n5000_k20000_e0.05_s42.fasta synth_ref.sample_n5000_k20000_e0.05_s42.fasta.maps
```

Further entry points:

```bash
julia --project=RotorMap RotorMap/runtests.jl                   # minimal fp8 GPU smoke test
julia --project=RotorMap RotorMap/experiments/e2e_auto.jl test  # cc-chosen compact flow self-test
julia --project=RotorMap RotorMap/experiments/e2e_fp8.jl human  # full human-genome mapping
julia --project=RotorMap RotorMap/experiments/e2e_complex.jl bench
julia --project=RotorMap RotorMap/experiments/eval_human1x.jl <index.bin>...   # chrX eval
julia --project=RotorMap RotorMap/tools/fastasubset.jl in.fna out.fna '^NC_000001' ...
julia --project=RotorMap RotorMap/tools/separate_provenance.jl reads.fasta [prefix]
julia --project=RotorMap RotorMap/tools/paf/header_stats.jl minimap2.paf       # vs header truth
```

The ENV interface is unchanged from the pre-reorganization scripts (`E2HUMAN_*`,
`E2COMPLEX_*`, `E2CUTS_*`, `E2COMPACT_*`, `IF_*`, `IS_*` — see each file's header); the
probes/sweeps are env- or constant-driven and print reports (sweeps also write CSV + SVG
into `experiments/out/`).

## Legacy

`legacy/` is the frozen pre-reorganization state: `legacy/src/` (the old `RotorMap` package:
RopeEncoders, RopeIndexers, Mapper, RopeScores, SplitComplexMatrices, TreeArrays, Utils),
`legacy/bin/`, and `legacy/test/` (all historical experiment scripts, kept for reference and
git archaeology). It is not included by any new code.
