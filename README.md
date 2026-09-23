# RotorMap

RotorMap indexes a reference genome and maps (possibly heavily mutated) DNA
reads back to it with a **RoPE-based** encoding (rotary position embeddings
applied to s-mer histograms) and a **batched GPU GEMM + top-k** search,
instead of classic seed-and-extend alignment. Everything here is Julia +
CUDA.jl, organized as a layered *workbench* (there is no package to
`using RotorMap` — scripts `include` the layers they need in a fixed order,
see `RotorMap/README.md`).

```
 reference.fasta ─┬─► index/      rope-encode sliding windows (fwd + revcomp) → index .bin
                  │
 reads.fasta    ◄─┤ reads/        sample reads (with mutations) + provenance headers
    │             ├─ fasta/       2-bit packing, batched/parallel FASTA readers
    │             ├─ encode/      rope encoder (production CUDA kernel + CPU golden refs)
    │             ├─ gemm/        GEMM engines: fp16 cuBLAS, fp8 cuBLASLt (:lt), fp8 PTX (:mma)
    │             ├─ search/      streaming search engines over the resident index database
    └─────────────┴─ experiments/ end-to-end experiment entry points (fp8/fp16/
                                  complex/cuts variants; see each file's header)
    tools/map.jl  ◄─ the mapping CLI: index .bin + reads -> match table
```

All commands below run **from the repository root**.

---

## 1. Requirements & Installation

### Requirements

| Requirement | Details |
|---|---|
| Julia | **1.12.x** (developed and tested on 1.12.2) |
| OS | Linux x86-64 |
| GPU | NVIDIA, CUDA.jl-supported driver. The **numeric path is picked from the compute capability**: cc ≥ 8.9 (Ada RTX 40xx), 9.0 (Hopper) or 12.x (Blackwell RTX 50xx) → **fp8 (e4m3)** flows, including the default `map.jl` mapping CLI; anything older → the **fp16** flows. Tested on an RTX 5090 (sm_120). |
| CUDA runtime | 13.x, pinned via `RotorMap/LocalPreferences.toml` (`CUDA_Runtime_jll`/`CUDA_Compiler_jll` = 13.2 — committed, so artifacts precompile without a visible driver) |
| VRAM | trivial for synthetic work; the full human-genome fp8 index database is ≈ 6.3 GiB resident on the GPU (the classic fp16 row layout is ≈ 11.9 GiB) → 16 GB+ VRAM recommended for human-scale mapping |
| RAM / disk | 32 GB RAM comfortable at human scale; the GRCh38.p14 fasta is 3.1 GB, its fp8 index .bin ≈ 6.3 GB |
| Threads | pass `-t 8` (or more): sampling, mutation and the encode/GEMM pipelines are multithreaded |

The optional fp8 **`:mma`** engine additionally needs `nvcc` 13.3 at
`/usr/local/cuda-13.3` (`gemm/fp8_ptx.jl` compiles a PTX mma.sync kernel);
the default production **`:lt`** engine only needs cuBLASLt (ships with the
pinned runtime). A GPU is **required** for index building and mapping; the
DNA utilities and the read sampler (§3.1, §3.2) are CPU-only.

### Installation

```bash
git clone <this-repo> rotormap && cd rotormap

# resolve + install every dependency (first run downloads CUDA artifacts)
julia --project=RotorMap -e 'using Pkg; Pkg.instantiate()'

# verify the GPU is visible and the driver userland matches
julia --project=RotorMap -e 'using CUDA; @show CUDA.functional(); println(CUDA.name(device()))'
```

Notes:

* **Driver/userland mismatch** (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, code
  803): the CUDA runtime artifact must match the *kernel* driver's userland.
  On the kau container this is already handled — `~/.bashrc` prepends the
  matching Nix driver libs to `LD_LIBRARY_PATH`, and non-interactive
  `ssh kau '...'` picks it up. The `LD_LIBRARY_PATH` must be set **before
  Julia starts**.
* No `Pkg.develop`/`using RotorMap`: `RotorMap/Project.toml` exists only for
  dependency resolution (`julia --project=RotorMap …`). Code is assembled by
  plain `include` in a fixed topological order; every experiment/tool file
  starts with the include prefix it needs. Two rules from
  `RotorMap/README.md`: include each layer file **at most once per session**,
  and **never co-include** `search/engine_fp16.jl` with `search/engine_fp8.jl`
  (nor `engine_complex.jl` with either) — they intentionally define clashing
  names.
* The pre-reorganization CLI package (ArgParse subcommands `random
  reference`, `random reads`, `index`, `search`, `convert`, `server`) is
  **frozen** under `RotorMap/legacy/` — not part of the current workflow.

---

## 2. Self-test

One script exercises every main capability and picks the optimal numeric path
for the detected hardware **before** any layer is included (the fp8 and fp16
engines are not co-includable, so the branch is an include-time decision):

```bash
julia --project=RotorMap -t 8 RotorMap/selftest.jl
```

What it does:

| Stage | Capability | How |
|---|---|---|
| 0 | **Hardware detection** | `CUDA.capability` → cc ≥ 8.9 ⇒ fp8 path (`e2e_fp8.jl` stack, cuBLASLt `:lt` engine); otherwise fp16 path (`e2e_fp16.jl` stack). Prints GPU name, VRAM, capability, chosen path. No CUDA ⇒ clean error (index + mapping are GPU-only). |
| A | **Random DNA creation** | `generate_reference` (seeded), FASTA write → independent re-parse roundtrip, `generate_reads` at err = 0 must return *exact* windows, `mutate` (err = 0 no-op, err > 0 length-preserving + deterministic in its rng). |
| B | **RoPE index creation** | The **production** encode kernel builds an index .bin (fwd + revcomp windows, s=8 m=4 c=4) from a cached 1 Mbp two-record fixture; `load_index` re-reads and validates it. |
| C | **Index sampling** | The production `sample_reads` (`:even` draw over the N-free window pool): provenance headers parse, err = 0 reads **bitwise equal** their claimed windows (two independent FASTA parsers agree), deterministic across calls. |
| D | **Mapping** | Full load → resident database → rope stream → GEMM + top-k → provenance scoring: 1000 err = 0 reads must map **100 %**; 200 reads mutated at `SELFTEST_ERR = 0.05` must hit their true locus at ≥ `SELFTEST_MUTATED_MIN_RATE = 0.90` within the top-20. |

Fixtures are cached under `NEWFASTA_V3_DIR` (default `/tmp`) and shared with
`RotorMap/runtests.jl`. Expected output (kau, RTX 5090):

```
==============================================================================
RotorMap selftest
hardware: NVIDIA GeForce RTX 5090, 31.4 GiB, compute capability 12.0
numeric path: fp8 (e4m3 database, cuBLASLt :lt engine)
==============================================================================
  A. random DNA creation (generate_reference / generate_reads / mutate / fasta) OK
  B. rope index creation OK: 488 windows (fwd + rc) -> /tmp/e2ehuman_ref1M.indexreal_fp8.bin
  C. index sampling (sample_reads :even, N-free pool, provenance) OK
  D. mapping OK: err = 0 -> 1000/1000; err = 0.05 -> 200/200 (100.0%) top-20 hits on the true locus
     first-hit rank histogram: [200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
==============================================================================
SELFTEST PASSED (all stages) in 8.6 s -- fp8 path on NVIDIA GeForce RTX 5090
```

The first-ever run pays one-time CUDA.jl JIT (~1–2 minutes total); later
runs with warm fixtures take seconds. Exit code 0 = all stages passed; a
failed assert aborts with a nonzero code.

Tunables (env): `SELFTEST_ERR` (stage-D mutation rate, default `0.05`),
`SELFTEST_MUTATED_MIN_RATE` (pass floor, default `0.90`),
`NEWFASTA_V3_DIR` (fixture cache dir).

`RotorMap/runtests.jl` remains the minimal GPU smoke test (fp8 tiny e2e
only); `selftest.jl` is the full capability check.

---

## 3. The four flows

Every flow can be driven two ways: **CLI scripts** (env-parameterized, below)
or **library calls** (the same functions the scripts call — signatures in
§3.5). A complete synthetic walkthrough is at the end of this section.

DNA is stored as `Vector{UInt8}` codes **0=A, 1=C, 2=G, 3=T**; FASTA files
may contain lowercase and IUPAC ambiguity codes (every non-ACGT byte decodes
to `G` but *keeps its position*; `\n`/`\r` are not positions).

### 3.1 Random DNA generation

CLI (`RotorMap/tools/gen_dna.jl`, CPU-only — no library calls needed):

```bash
# a random 2 Mbp reference (default out ref_n2000000_s1.fasta, header >chr0)
julia --project=RotorMap -t 8 RotorMap/tools/gen_dna.jl ref 2000000 --seed=1
```

`ref` draws `N` bases uniformly from ACGT (`--seed` reproducible, `--head=`
overrides the record header line sans `>`). The same script also has a
`reads` mode — iid uniform windows over a fasta + mutations at a given err,
with sample provenance headers (run it with no args for its usage) — but for
reference-sampling use §3.2's provenance-preserving sampler instead.

Library calls (the same functions the CLI wraps):

```julia
# any script: the common layer is self-contained
include(joinpath("RotorMap", "common", "treearrays.jl"))
include(joinpath("RotorMap", "common", "dna.jl"))

using Random: Xoshiro

ref = generate_reference(2_000_000; seed = 1)     # 2 Mbp random ACGT, codes 0:3
save_fasta([ref], "ref.fasta"; heads = [">chr0 synthetic"])   # wrapped FASTA

reads, pos = generate_reads(ref, 20_000, 100; err = 0.05, seed = 2)
# 100 reads x 20,000 bases: iid uniform windows (with replacement),
# each mutated at rate err; pos[i] = 0-BASED start of read i's window.

mut = mutate(ref[1:20_000], 0.05; rng = Xoshiro(3))
# mutate(dna, err; rng): n_err = ceil(n*err) edits = ~1/3 substitutions +
# ~1/3 insertions + ~1/3 deletions; equal ins/del counts keep the length
# exactly n.  err = 0 is a no-op.  Deterministic in rng.

shifted = mutate_shift(ref[1:20_000], 0.05)       # left-shift by n*err/2 + random tail
```

| Parameter | Meaning |
|---|---|
| `generate_reference(N; seed)` | `N` bases drawn uniformly from ACGT; `seed` makes it reproducible |
| `generate_reads(ref, k, n; err, seed)` | sample `n` reads of length `k` (0-based true starts returned), mutate each at fraction `err` |
| `mutate(dna, err; rng)` | `err` = mutation **fraction** of the read length (`ceil(k*err)` single-base edits; substitution/insertion/deletion ≈ ⅓ each) |
| `save(data, file)` / `load(file)` | Julia-serialization of any `.bin` (index files, sequences) |
| `save_fasta(dnas, file; heads)` | codes → FASTA text (4 KB lines) |

### 3.2 Read sampling with mutations (provenance-preserving)

The **sample** flow samples reads straight from a reference FASTA —
its output headers carry each read's *ground truth*, which the mapping flow
scores against:

```
>read_7 start=5402311 len=20000 src=NC_000001.11 ...
```

`start` is the 1-based window start in the record's sequence-character
space, `src` the record name (the header's first whitespace token, without `>`).

```bash
# sample 1000 reads x 20,000 bases, 5 % mutations (defaults shown)
SAMPLE_FASTA=ref.fasta \
SAMPLE_N=1000 SAMPLE_K=20000 SAMPLE_ERR=0.05 \
julia --project=RotorMap -t 8 RotorMap/reads/sample.jl gen      # -> ref.sample_n1000_k20000_e0.05_s42.fasta

julia --project=RotorMap -t 8 RotorMap/reads/sample.jl verify   # re-validate the saved batch
julia --project=RotorMap -t 8 RotorMap/reads/sample.jl test     # CPU-only correctness suite
julia --project=RotorMap -t 8 RotorMap/reads/sample.jl gen!     # force regeneration
```

(CPU-only and self-contained — no GPU needed. `ARGS[2]` overrides the fasta:
`… sample.jl gen ref.fasta`.)

| Env | Default | Meaning |
|---|---|---|
| `SAMPLE_FASTA` | `…/data/GCF_000001405.40_GRCh38.p14_primary25.fna` | reference FASTA to sample from |
| `SAMPLE_N` | `1000` | number of reads |
| `SAMPLE_K` | `20000` | read length (valid windows = fully inside one record) |
| `SAMPLE_ERR` | `0.05` | mutation fraction per read (same semantics as `mutate`; ins = del ⇒ read length exactly `k`) |
| `SAMPLE_SEED` | `42` | RNG seed (drives the mutations; the `:even` *selection* is seed-free) |
| `SAMPLE_DRAW` | `even` | `even`: deterministic quotas ∝ record pool size, reads at equal-strata midpoints · `random`: iid uniform draws with replacement (output name gets `_drandom`) |
| `SAMPLE_NFREE` | `1` | only sample windows with **no** N/IUPAC/junk byte (recommended — those bytes decode to fake Gs); `0` restores the raw pool (`_withn` name tag) |
| `SAMPLE_OUT` | parameter-record name | output FASTA (see below) |
| `SAMPLE_ROWS` / `SAMPLE_LEV` | `24` / `256` | `verify` spot-check rows / Levenshtein DP cap |
| `SAMPLE_FORCE` | `0` | `1` = regenerate even if the output exists |

The **output name doubles as the parameter record** —
`<fasta>.sample_n<N>_k<K>_e<ERR>_s<SEED>[_drandom][_withn].fasta` —
and `verify` parses it back: it checks every header against the reference,
re-extracts spot rows independently (err = 0: bitwise equal; err > 0:
different + 16-mer containment + Levenshtein edit rate just under `err`),
and regenerates the whole batch bitwise when all parameters are known.

Library call: `sample_reads(fasta; n, k, err, seed, draw, nfree) →
(reads, heads, recs, starts)`.

### 3.3 Index generation

The **indexreal** flow rope-encodes every `k`-window (slid by `kstep`,
last window fully inside the record; records shorter than `k` contribute
nothing) of every reference record, **forward and reverse-complement**, with
the production GPU kernel, and saves an index `.bin` next to the FASTA:

```bash
# the human default (k = 20,000, kstep = 2,000, s5 m1 c5 -> row dim 2048)
julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save [fasta]

# a custom config, e.g. maize at k = 1000, kstep = 100, s4 m1 c4 (row dim 512):
INDEX_FASTA=maize.fna INDEX_K=1000 INDEX_KSTEP=100 \
INDEX_S=4 INDEX_M=1 INDEX_C=4 INDEX_FP8BIN=maize.indexreal_fp8.bin \
julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save

julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl verify [fasta] [--bin8]
#                            test | bench | gen | all are also available
```

Reference speed (RTX 5090, 16 threads, GRCh38.p14 = 3.1 Gbp): full stream +
encode ≈ 18 s, ≈ 1.55 M windows ×2 strands, `.bin` written in ≈ 2 s.

| Env | Default | Meaning |
|---|---|---|
| `INDEX_FASTA` | `…/data/GCF_000001405.40_GRCh38.p14_primary25.fna` | reference to index |
| `INDEX_K` | `20000` | window (= fragment/read) length — the **index granularity**; mapping reads must use the same `k` |
| `INDEX_KSTEP` | `k/10` | slide step — smaller ⇒ denser index (better recall, bigger `.bin`) |
| `INDEX_S` | `5` | s-mer size (`4^s` buckets before compaction) |
| `INDEX_M` | `1` | multiplicity (rotations per strand; `2^q` targets a qubit system) |
| `INDEX_C` | `= INDEX_S` | compactness: keep only `4^c` buckets via a golden-ratio scramble (`c ≤ s`; row dim = `2·m·4^c`; defaults to `s` when unset) |
| `INDEX_FP8` | `1` | save the **fp8 mapping layout** (transposed e4m3 database columns — a pure H2D at load time); `0` = classic fp16 row layout |
| `INDEX_FP16` | `1` | stream eltype while encoding (`0` = fp32) |
| `INDEX_NORM` | `0` | normalization mode — the mapping flow asserts `0` (unit-energy rows ⇒ normalized database columns) |
| `INDEX_BATCH` | `8192` | fragments per encode batch |
| `INDEX_BIN` / `INDEX_FP8BIN` | `<fasta>.indexreal{,_fp8}.bin` | output paths |

Each index column carries its **location meta**: the record's verbatim
header (`heads`), the 1-based window start (`starts`) and the strand
(`strand`, 0 = fwd / 1 = rc). The encoder config (`k, s, m, c, normalize`)
is saved along — every consumer reads it back, so a mapping run always
follows the index it loads.

### 3.4 Mapping

**`RotorMap/tools/map.jl`** is the mapping CLI: it loads the index `.bin`,
streams the reads through rope encoding and the batched fp8 GEMM + per-row
top-k, and writes every read's top-k matches to a table. It does NOT parse
any provenance — the reads may be plain fasta with arbitrary headers. At
startup the GPU's compute capability picks the numeric stack (BEFORE any
include — the fp8 and fp16 engines are not co-includable): cc ≥ 8.9 runs
the fp8 flow (e4m3 database, the fixed cuBLASLt `:lt` engine), anything
older the fp16 cuBLAS flow. Residency is AUTO: when the database plus the
engine's working set fits the free VRAM it is uploaded once and the
resident engine runs (full speed); otherwise it stays in host RAM and is
swept through a triple-buffered device ring (`w` columns at a time) — the
ring runs indexes of ANY size. Both paths are bitwise identical; the decision is logged
either way.

```bash
INDEX=<index .bin> READS=<reads fasta> \
julia --project=RotorMap -t 8 RotorMap/tools/map.jl -topk=20 --batch=8192

INDEX=ref.indexreal_fp8.bin READS=reads.fasta MAPS=matches.txt \
julia --project=RotorMap -t 8 RotorMap/tools/map.jl --w=65536 --segs=8
```

| Env / Flag | Default | Meaning |
|---|---|---|
| `INDEX` | — | index `.bin` path (required; build with §3.3) |
| `READS` | — | reads fasta (required; records shorter than the index window `k` are skipped, records **longer than `k` are trimmed to their first `k` bases** — silently, no warning) |
| `MAPS` | `<READS>.maps` | output table path (overwritten) |
| `-topk` | `20` | top-k matches kept (and emitted) per read |
| `--w` | `65536` fp8 / `32768` fp16 | database column chunk width = ring slot width |
| `--batch` | `8192` | reads per encode/GEMM batch — larger batches amortize the ring sweep's H2D |
| `--segs` | `8` | column segments per row in the fused top-k kernel |
| `-v` | off | verbose: stage timings (index load, host database build), batches/s + reads/s, the ring sweep's PCIe GB/s or the resident upload GB, GPU memory occupancy around the pipeline, and a **JIT-aware flow-time split** (see below) |

**JIT-aware scoring:** Julia's first pipeline call pays a one-time JIT
compile. With `-v`, `map.jl` splits the flow wall time at the **first
batch**: the first batch's time is reported separately as the JIT/first-call
warm-up, and the remaining batches give the steady-state (warm) rate. The
printed total is therefore the cold end-to-end time — for benchmark
numbers, subtract the first batch or just run twice. The eval harness
`RotorMap/experiments/eval_human1x.jl` does the latter: it always runs
`reps = 2` `map_and_count` passes per index (rep1 pays the process's
one-time pipeline JIT — and, on the resident path, the one-time database
upload — while rep2 is fully warm) and reports **only the warm rep** in its
summary metrics and machine-readable `EVALRESULT` row (`map2_s`,
`e2e_warm_s = load + build + warm map`).

**Output format** (`MAPS`): one whitespace-separated line per (read, rank),
`topk` lines per read, matches sorted by inner-product score DESCENDING:

```
<read header> <'+'|'-'> <score> <0-based position> <record name>
```

* read header — the reads fasta's header, sans the leading `>` (multi-token
  headers pass through; the `+`/`-` token separates them from the fields)
* direction — the indexed column's strand: `+` = forward window, `-` =
  reverse-complement window
* score — the inner product of the read's embedding (e4m3-quantized in fp8
  mode) with the database column
* position — the record-relative 1-based window start minus 1, in the
  ORIGINAL record coordinates (both strands)
* record name — the index record header's FIRST whitespace token, sans `>`
  (multi-token record headers cannot break the table)

#### Verification (reads sampled by this toolchain)

When the reads were produced by §3.2's sampler (provenance headers
`>read_<i> start=<1-based> len=<k> src=<record name>`),
`RotorMap/tools/verify_maps.jl` cross-checks the match table against that
provenance: for every read it tests the same-record + k-long-overlap
condition against the header's `(src, start)` and reports how many reads'
top-1 match — and how many table hits in all — point at the true locus.
Unprovenanced reads are skipped quietly.

```bash
julia --project=RotorMap RotorMap/tools/verify_maps.jl \
      <reads.fasta> <out.maps> [K]    # K defaults to a header's len=
```

Measured on kau (RTX 5090, k = 20,000, kstep = 2,000 fixture): 500 mutated
(err = 0.05) reads map with every top-1 hit intersecting the true locus
(fp8 resident path, ≈ 7 s wall).

Further experimental flows (`RotorMap/experiments/`) keep the provenance-
scoring e2e harness with stats: `e2e_compact.jl` (fp8 compact),
`e2e_compact16.jl` (fp16 compact, pre-fp8 GPUs), `e2e_auto.jl`
(compute-capability-driven dispatch over the two), `e2e_fp8.jl`
(always-resident baseline + bench), `e2e_fp16.jl` (fp16 reference
accuracy), `e2e_complex.jl` (split-complex GEMM), `e2e_cuts.jl`
(calibrated score cuts). See each file's header for modes/flags/env.
Measured RTX 5090 sweeps (error-rate matrices, human and maize results) are
recorded there and in the `eval_*.log` files.

The encoder config
always follows the loaded index.

#### End-to-end example on synthetic data (no external data needed)

```bash
# 1. a random 2 Mbp reference
julia --project=RotorMap -t 8 RotorMap/tools/gen_dna.jl ref 2000000 synth_ref.fasta --seed=1

# 2. index it (k = 20,000, kstep = 2,000)
INDEX_FASTA=synth_ref.fasta julia --project=RotorMap -t 8 RotorMap/tools/build_index.jl save

# 3. sample 5,000 mutated reads with provenance (err = 0.05)
SAMPLE_FASTA=synth_ref.fasta SAMPLE_N=5000 \
julia --project=RotorMap -t 8 RotorMap/reads/sample.jl gen

# 4. map them back (auto residency: a small index auto-runs the resident engine)
INDEX=synth_ref.indexreal_fp8.bin \
READS=synth_ref.sample_n5000_k20000_e0.05_s42.fasta \
julia --project=RotorMap -t 8 RotorMap/tools/map.jl

# 5. verify against the sampled provenance
julia --project=RotorMap RotorMap/tools/verify_maps.jl \
      synth_ref.sample_n5000_k20000_e0.05_s42.fasta \
      synth_ref.sample_n5000_k20000_e0.05_s42.fasta.maps
```

On an fp8-capable GPU step 4 auto-runs resident (a small index never needs
the ring); on pre-fp8 hardware the same command runs the fp16 flow. Measured
on kau (RTX 5090): steps 4+5 report all 5,000 reads' top-1 intersecting
their true loci.

### 3.5 Library API (the functions behind the CLIs)

Layers are included once, in the canonical order (see the include list at
the top of any `experiments/e2e_*.jl`), then:

```julia
re = RopeEncoder(k = 20_000, s = 8, m = 4, c = 4)   # k: window length; s: s-mer size;
                                                    # m: multiplicity; c: compact buckets
dim_real = 2 * re.m * 4^re.c                        # real row dim ([Re; Im] stacked)

d  = load_index("ref.indexreal_fp8.bin")        # NamedTuple: embeds8/norms/heads/starts/strand/k/s/m/c/…
B  = index_database(d)                              # fp8 e4m3 database (CuMatrix{F8}, pure H2D)
B  = index_database_f16(d)                          # fp16 database (widens fp8 or transposes fp16 rows)

res = map_and_count(re, reads_fasta, B, d.heads, d.starts, d.k;
                    ktop = 20, w = 2^16, batch_size = 2^13, segs = 8,
                    engine = :lt, progress = true)   # fp8 engine; the fp16 engine has no `engine` kwarg
res.correct, res.total, res.rank_hist               # mapping score

batches = RopeRealBatch{Float16}[]                 # encoding-only stream (Channel):
for nt in rope_encode_real_stream(re, "reads.fasta"; k = re.k,
                                  batch_size = 2^13, normalize = 0, fp16 = true)
    push!(batches, nt)                             # rope fp16 embeddings + heads
end
eng = TopKEngine(B; k = 20, w = 2^16, rows_cap = 2^13, segs = 8, engine = :lt)
for bat in topk_flow(batches, eng)                  # GEMM + top-k only (no provenance scoring)
    # bat.vals (nb × k scores), bat.locs (nb × k global db column ids)
end
```

Lower still: `encode_frag_real_batch!` (the production encode kernel),
`fasta_reads` / `FastaBatchesV2` (parallel FASTA ingestion), `mutate` /
`sample_reads` (CPU), `F8` quantization helpers in `gemm/fp8_convert.jl`,
kernels in `gemm/*.jl`. The compact flow's building blocks live in
`search/engine_compact.jl` (fp8) and `search/engine_compact16.jl` (fp16):
`index_database_host`/`index_database_host_f16` (host-RAM databases),
`CompactEngine`/`CompactEngine16` (triple-buffered rings),
`topk_flow_compact`/`topk_flow_compact16`,
`rope_topk_stream_compact`/`rope_topk_stream_compact16` (auto residency via
`resident = nothing`) and `resident_fits`/`resident_fits16` (the policies,
with the VRAM-need formulas on their docstrings).
`RotorMap/lab/` holds the low-level kernel playgrounds; `RotorMap/legacy/`
the frozen pre-reorganization package.

### 3.6 Data formats

* **FASTA input** — plain records; junk/IUPAC/N decode to `G` keeping its
  position; CRLF fine; records shorter than `k` are skipped by the index and
  the sampler.
* **Index `.bin`** — Julia-serialized `NamedTuple`, format tag
  `indexreal.fp8.v1` (fp8 mapping layout: `embeds8 (rdim, 2·n_frag)`,
  transposed e4m3 columns) or `indexreal.v1` (classic fp16 rows
  `embeds (2·n_frag, rdim)`); both carry `norms (m, 2·n_frag)`, `heads`,
  `starts`, `strand`, `k, kstep, s, m, c, normalize, n_frag, source`.
  Pre-rename saves with the `indexflowreal.*` tags keep loading.
* **Provenance headers** — sample format
  `>read_<i> start=<1-based> len=<k> src=<record name>` (`src` = the record
header's first whitespace token, sans `>`) , or the
  colon eval format `>ACCESSION:<0-based start>:<true|false>` (reverse-
  complement flag; length trimmed by the reader). Both are parsed by
  `parse_read_head` (RotorMap/reads/provenance.jl).


# Citation

```
@misc{yakymenko2026rotormapquantumfingerprintsdna,
      title={RotorMap and Quantum Fingerprints of DNA Sequences via Rotary Position Embeddings}, 
      author={Danylo Yakymenko and Maksym Chernyshev and Illia Savchenko and Sergii Strelchuk},
      year={2026},
      eprint={2603.22245},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.22245}, 
}
```
