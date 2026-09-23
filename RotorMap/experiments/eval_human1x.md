# eval_human1x.jl — human chrX eval batch through the e2e_compact flow

HOW-TO + RESULTS for the instrumented eval of maksym's human chrX batch
(`human_1X_5.fasta`, colon provenance headers) against the two GRCh38
primary25 fp8 indexes.  The driver reuses the whole `e2e_compact.jl` flow
verbatim (auto VRAM residency: both indexes fit the RTX 5090, so the resident
engine runs) and adds the metric collection on top.

## Inputs

| file | what |
|---|---|
| `/share/q4bio/maksym/rotormap/generated_data/eval/human_1X_5.fasta` | 121,468 reads x 23,021 bases from chrX, headers `>NC_000023.13:<0-based start>:<true/false revcomp>` (provenance format 2 of `reads/provenance.jl`; scoring keys records by ACCESSION) |
| `/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.indexreal_fp8_s5m1c5.bin` | 6.373 GB (5.94 GiB), 3,087,812 cols x rdim 2048, k = 20,000, kstep = 2,000, encoder (5,1,5) |
| `/share/q4bio/dandan/rotormap/data/GCF_000001405.40_GRCh38.p14_primary25.indexreal_fp8_s4m1c4.bin` | 1.630 GB (1.52 GiB), 3,087,812 cols x rdim 512, k = 20,000, kstep = 2,000, encoder (4,1,4) |

Both indexes fit the 32 GiB VRAM -> `rope_topk_stream_compact` auto-delegates
to the resident engine (one B upload per pass, no per-batch PCIe sweep); the
log states the decision (`resident = true`).

Scoring geometry: the reader trims every read to its FIRST k = 20,000 bases,
so the scored fragment is `ref[start : start+20k]` -- valid for revcomp reads
only when readlen < 2k (here 23,021 < 40,000).  See the provenance-geometry
note in `legacy/test/e2ecompact.jl`'s RESULTS header before scoring batches
with longer reads.

## Metric definitions (all from `map_and_count`, `search/engine_fp8.jl`)

- **top20**  -- reads whose true window intersects ANY of the top-20 hits
  (`correct / total`), %.
- **top1**   -- reads whose FIRST hit intersects (`rank_hist[1] / total`), %.
- **avg20**  -- the average BEST (max) overlap length over each mapped read's
  20 hits (`inter_sum / inter_cnt`), reported in bases and as % of the
  20,000-base fragment.  Grid geometry predicts a k - kstep/4 = 19,500
  (97.5%) plateau at low error; external-generator batches score below it
  (harsher error model + revcomp trimmed-fragment geometry -- same signature
  as the maize eval rows in `e2ecompact`'s RESULTS header).
- **time total e2e**  -- `t_load + t_build + warm map_s` (host index load +
  fp8-database build + the warm `map_and_count` wall time).  rep1 of the
  first index pays the process's one-time pipeline JIT (16.4 s / 9.8 s
  measured) and is reported separately.
- **time gpu compute**  -- NOT measured in-process; the nvidia-smi sampler
  the run wrapper starts alongside produces a 200 ms utilization log, and
  the GPU-busy seconds per pass = the util% integral over the pass's
  `PHASE rep*_start/_end` window (parser below).
- **index size GB**  -- the .bin file size (GB decimal and GiB binary).
- **VRAM used (peak)**  -- a 20 ms `CUDA.free_memory()` poller; driver-level
  used = total - min free (the process owns the GPU).  The STEADY resident
  set is rep1's fresh peak (B + C buffers + staging); the max over a 2-pass
  process is higher because the pool keeps rep1's freed B cached while rep2
  uploads its own -- quote the steady number per index, mention the max.

## Running (on kau; develop locally, execute remotely)

The kau `.bashrc` exports the Nix driver libs (`LD_LIBRARY_PATH=...nvidia-x11-595.../lib`)
-- non-interactive ssh picks it up; CUDA must not start without it.

```bash
# 1. deploy the driver (committed from the local repo)
cd /workspace/rotormap && git add -A && git commit -m "..." && git push origin main

# 2. warm the page caches (load_s otherwise includes cold /share I/O)
ssh kau 'cat <index1.bin> <index2.bin> <reads.fasta> > /dev/null'

# 3. run ONE index per julia process (clean VRAM peak per index), with the
#    smi sampler alongside; logs land in the repo root (gitignored *.log)
ssh kau '
cd /home/dandan/aprojects/rotormap/RotorMap
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 200 > ../eval_human1x.smi.log &
SMI=$!
julia --project=. -t 8 experiments/eval_human1x.jl <index.bin> \
  > ../eval_human1x.log 2>&1
kill $SMI 2>/dev/null
'
# no ARGS = both GRCh38 indexes in ONE process (JIT paid only once, but the
# second index's VRAM peak then carries the first one's pool leftovers)
```

## Output

- `PHASE ...` lines on stderr carry wall-clock stamps for load / map /
  per-rep windows (correlate with the smi log's timestamp column).
- A human summary per index (`== EVAL ...`, per-rep `top20/top1/avg20` lines,
  peak VRAM, total e2e) and one machine-readable row per index:

```
EVALRESULT idx=... file_gb=... gib=... k=... kstep=... cols=... rdim=... reads=... \
load_s=... build_s=... resident=... map1_s=... map2_s=... top20_pct=... top1_pct=... \
avg20_bases=... avg20_pct=... vram_peak_gib=... vram_total_gib=... e2e_warm_s=...
```

`vram_peak_gib` is the max over the whole per-index process (see the
pool-carryover caveat above).

### GPU-busy parser (python3, run against the log pair)

```python
import re, datetime as dt
def pts(s):
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S.%f"):
        try: return dt.datetime.strptime(s.strip(), f)
        except ValueError: pass
phases = []
for line in open('eval_human1x.log', errors='ignore'):
    m = re.match(r'PHASE (\S+)(?: idx= (\S+))?.*?(\d{4}[-/]\d{2}[-/]\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*$', line)
    if m: phases.append((m.group(1), m.group(2), pts(m.group(3))))
samples = []
for line in open('eval_human1x.smi.log'):
    p = [x.strip() for x in line.split(',')]
    if len(p) == 3:
        try: samples.append((pts(p[0]), float(p[1]), float(p[2])))
        except ValueError: pass
wins, cur = {}, None
for name, idx, ts in phases:
    if name.endswith('_start') and name.startswith('rep'): cur = (idx, name[:4], ts)
    elif name.endswith('_end') and name.startswith('rep') and cur and cur[0] == idx:
        wins[(idx, cur[1])] = (cur[2], ts); cur = None
for (idx, rep), (t0, t1) in sorted(wins.items()):
    busy = 0.0; mem = 0.0; prev = None
    for ts, u, m in samples:
        if t0 <= ts <= t1:
            busy += u / 100 * ((ts - prev).total_seconds() if prev else 0.1)
            mem = max(mem, m)
        prev = ts if t0 <= ts <= t1 else prev
    w = (t1 - t0).total_seconds()
    print(f"{idx} {rep}: wall {w:.1f} s, gpu busy {busy:.2f} s ({100*busy/w:.0f}%), "
          f"max mem {mem/1024:.2f} GiB")
```

## Results (2026-09-20, kau RTX 5090 32 GiB, julia 1.12.2 -t 8, page cache warm; defaults: batch 8192, w = 2^16, engine :lt, ktop 20; 121,468 reads)

| metric | s5m1c5 | s4m1c4 |
|---|---|---|
| top1 | 98.67 % (119,857) | 98.25 % (119,341) |
| top20 | 99.97 % (121,432; 36 unmapped) | 99.85 % (121,287; 181 unmapped) |
| avg20 | 18,680 bases = 93.40 % | 18,435 bases = 92.18 % |
| time total e2e (load + build + warm map) | 6.1 s (1.5 + 0.0 + 4.6) | 3.1 s (0.6 + 0.0 + 2.5) |
| time gpu compute (warm pass) | ~4.0 s (86 % util) | ~2.1 s (83 % util) |
| index size | 6.373 GB / 5.94 GiB | 1.630 GB / 1.52 GiB |
| VRAM used, steady resident peak | 9.2 GiB (17.1 GiB max over a 2-pass process) | 4.4 GiB (7.8 GiB max) |

rep1 (one-time pipeline JIT paid): 16.4 s (s5m1c5) / 9.8 s (s4m1c4) wall at
the same metrics -- bitwise identical across reps and across the separate
one-index-per-process re-run.  Readings: the 5-mer encoder edges the 4-mer
(+0.42 pp top1, +0.12 pp top20, +1.2 pp avg20) at 4x the index size and ~2x
the warm map time; both run the resident engine well under VRAM.

## Gotchas

- `@printf` format strings must be ONE string literal: a `"a" * "b"`
  concatenation is not a `String` literal, so Julia 1.12's Printf macro
  misreads the call as the `(io, fmt)` form and throws at include time.
- Run one index per process when quoting VRAM peaks (pool carryover);
  running both in one process is fine for the mapping metrics (bitwise).
- `resident_fits` judges free VRAM at call time -- anything else holding
  VRAM (another process, a stale pool) flips the decision to the ring;
  check the log's `resident = ...` line.
- With `reps >= 2` on the resident path, a per-rep `rope_topk_stream_compact`
  call re-uploads B -- and re-checks residency against its own still-live
  copy, which cannot fit for any index over ~half of VRAM (rep2 threw with
  `22.5 GiB need vs 8.3 GiB free` on the maize k4000 s5m1c5 index).  The
  driver now builds the resident engine ONCE (upload under the VRAM poller)
  and streams all reps through it; rep1 pays upload + JIT, rep2 is a true
  warm pass.  Ring-path runs are unchanged.
- The driver's 20 ms poller measures driver-level used memory (context +
  pool), not the kernel's working set; the nvidia-smi `memory.used` column
  matches it exactly.
