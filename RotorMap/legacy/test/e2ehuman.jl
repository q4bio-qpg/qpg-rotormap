# ==============================================================================
# e2ehuman.jl -- END-TO-END HUMAN-GENOME MAPPING on the fp8 flow: the
# indexflowreal human-reference index (a REAL rope-encoded database, fwd +
# revcomp) loaded as the fp8 GEMM database, the indexsample reads (mutated
# genome samples) streamed through the ropeflowreal -> flowtopkfp8 pipeline,
# and every read's top-k hits checked against its PROVENANCE (ground truth).
#
# PROBLEM
#   flowtopkfp8.jl matches its streamed fragment embeddings against a RANDOM
#   database B.  This script swaps that B for the real thing and scores the
#   pipeline end to end:
#     index .bin (indexflowreal save: human GRCh38.p14, k = 20,000,
#                 kstep = 2,000 -> embeds (2*n_frag, 2*m*4^c) fp16, fwd+rc,
#                 with heads/starts/strand location meta)
#        -> B = e4m3^(rdim x 2*n_frag) resident on the GPU (columns = the
#           reference windows; normalize = 0 makes every row unit-energy, so
#           the columns are already normalized as the flow assumes)
#     reads fasta (indexsample gen: n = 131,072 reads x k = 20,000, err = 0.05,
#                 headers >read_<i> start=<s> len=<k> src=<record header>)
#        -> rope fp16 stream (ropeflowreal, the flow's own encoder config)
#        -> fp8 GEMM + per-row top-k (flowtopkfp8's TopKEngine, k = 20)
#        -> GROUND TRUTH: read r is CORRECTLY MAPPED iff its true window
#           [start, start+len-1] in record src INTERSECTS at least one of its
#           k returned database windows [db_starts[c], db_starts[c]+dbk-1]
#           (same record, |start difference| < window length).  The score is
#           the NUMBER of correctly mapped records (plus the rate, the
#           first-hit rank histogram and the average intersection LENGTH of
#           the BEST (largest-overlap) hit among the k returned ones with
#           the true window -- kept when > 0, reported over the correctly
#           mapped reads as a % of the fragment length, 20,000).
#
# DESIGN
#   1. Whole-matrix load.  ALL index rows become database columns -- the
#     forward AND the revcomp halves (3,094,162 columns, 6.33 GiB e4m3).  An
#     rc hit is the same genomic location (the .bin stores the same head/
#     start for both strands), so the intersection test needs no strand
#     special-casing.  Column j of B = window j of the index.
#   2. Index -> GPU.  DEFAULT (the indexflowreal fp8 save, format
#     "indexflowreal.fp8.v1", <fasta>.indexflowreal_fp8.bin): the stored
#     matrix embeds8 IS B -- (rdim, 2*n_frag) e4m3 columns quantized at save
#     time with the exact F8.(Float32.(.)) conversion -- so building B is a
#     PURE H2D (6.33 GiB, no transpose, no arithmetic; peak host RAM halved
#     too).  Legacy fp16-row .bin files (format "indexflowreal.v1") still
#     load: they are transposed on the GPU in 2^16-row blocks with the same
#     fused broadcast (index_to_f8) -- bitwise the same B, at ~4x the build
#     time (the link here is PCIe Gen1 x16 ~4 GB/s, so H2D bytes dominate).
#   3. normalize = 0 ASSUMED (asserted): the saved embeddings have unit
#     energy, so B's columns need no renormalization and the streamed reads
#     use the same normalize mode (rope_topk_stream's normalize = 0 default).
#   4. Provenance parsing.  TWO read-header formats are understood
#     (`parse_read_head`):
#       indexsample: `>read_<i> start=<s> len=<k> src=<record header sans '>'`
#         (start 1-based, matched against the record header verbatim)
#       colon eval:  `>record:start:revcomp`
#         e.g. >NC_000001.11:80433546:false -- the eval reads generated from
#         GCF_000001405.40_GRCh38.p14_primary25.fna (start 0-BASED, record
#         matched by ACCESSION = the '>' header's first whitespace token, no
#         length in the header: the fasta reader keeps every record's first
#         k bases -- records with different sizes are trimmed to exactly k =
#         20,000, records shorter than k would be skipped).
#     The database's per-column location is (heads[j] verbatim with '>',
#     starts[j] 1-based, window length = the .bin's k).  Per-column record ids
#     are precomputed once, keyed BOTH by verbatim header and by accession
#     (the human file has 68 unique records among 3.1M columns).
#   5. Shared runner.  `map_and_count` is the whole scoring loop; the tiny
#     self-test and the production run differ only in the inputs.
#
# SELF-TEST (mode `test`; production files not needed)
#   A tiny 1M-base two-record reference is generated ONCE (cached in
#   fastareads_v3's data dir, indexflowreal-style sprinkle: lowercase + IUPAC
#   junk, wrapped lines -- the builders' wrapped() emits width+1 chars per
#   line, i.e. ~1.4% duplicated boundary bases; harmless, every consumer
#   reads the same bytes), turned into an indexflowreal-FORMAT .bin by a
#   miniature of indexflowreal's fp8 save (same encode_frag_real_batch!
#   kernel, transposed e4m3 columns, same field set), 1000 reads of
#   k = 20,000 sampled at err = 0
#   (indexsample's rule: uniform over all valid windows, with replacement),
#   and pushed through the SAME load -> fp8 -> stream -> score path.  At
#   err = 0 every read window contains an index-window start within kstep/2,
#   i.e. shares >= 95% of its bases with a database window -- a margin no
#   quantization noise can close -- so the test asserts n/n correctly mapped
#   and a rank histogram dominated by rank 1.
#
# RESULTS (kau, RTX 5090, Ryzen 9 9950X, julia -t 16; index = the production
# indexflowreal save of GRCh38.p14: 1,547,081 windows x2 strands = 3,094,162
# columns x rdim 2048, 67 unique records; reads = the indexsample n=131,072
# k=20,000 err=0.05 s=42 file; k(top) = 20, w = 2^16, batch = 2^13 -> 16
# batches, engine :lt):
#
#   fp8 index .bin load (5.98 GiB, page-cache warm)       1.3 s   (was 2.0)
#   fp8 database build (index_database: PURE H2D)         0.55 s  (was 8.9 s
#     via index_to_f8's H2D + GPU transpose of the 12.7 GiB fp16 rows -- the
#     PCIe Gen1 x16 link made the extra 6.4 GiB + transpose cost ~8 s per
#     run; the fp8 save moved that work to save time, once)
#
#   MAPPING FLOW STAGE SPEEDS (bench mode, --reps=3, min reported; the first
#   pipeline pass is a ~4.2 s JIT/compile warmup and is excluded):
#     rope stage (fasta read + 2-bit pack + rope encode)   0.55 s   4.77 GB/s
#       quantize+upload (fp16 H2D + e4m3 convert)          1.50 ms/batch
#       fp8 GEMM (w = 2^16, M = 2^13)                      3.01 ms/chunk  731.5 TFLOPS
#       seg rowtop-k (fp16 chunks, segs = 8)               1.99 ms/chunk  503 GiB/s
#     gemm+topk stage (compute-only, rope replayed)        3.90 s   425.7 TFLOPS
#       (48 chunks/batch, gemm || topk double-buffered: 768 chunks/batch-set,
#       GEMM floor 2.31 s, exposed top-k ~1.5 s)
#     END-TO-END (fasta -> rope -> fp8 gemm -> topk -> score)  4.29 s
#       124,486/131,072 = 94.98% correctly mapped
#     stage budget: rope 0.55 + compute 3.90 vs e2e 4.29 s -> the rope stage
#       hides completely behind the compute stage (task-level pipelining);
#       the GEMM and top-k overlap only partially (their serialized sum
#       2.31 + 1.53 ~ 3.8 s ~ the measured compute-only time).
#
#   Bitwise-equality witness: with the fp8-format index the mapping numbers above are
#   EXACTLY the run that quantized the fp16 rows on the GPU at load time
#   (same count AND rank histogram) -- save-time F8.(Float32.(.)) is the
#   same conversion index_to_f8 applied.  With the err = 0.15 reads:
#   77,943 / 131,072 = 59.47% (rank-1: 54,978) -- one boundary read vs the
#   old path's 77,942 (cuBLASLt near-tie noise across processes; the
#   histograms differ by +-1 in a few ranks).  Full rank histogram (err =
#   0.05): 122429, 1220, 447, 194, 48, 35, 19, 15, 16, 9, 9, 7, 4, 5, 5, 3,
#   7, 6, 3, 5 | unmapped 6,586 (5.02%).
#   Error-rate sweep (same pipeline, n = 131,072 each, indexsample s = 42):
#     err = 0.05: 124,486 / 131,072 = 94.98%   (rank-1: 122,429)
#     err = 0.15:  77,943 / 131,072 = 59.47%   (rank-1:  54,978; unmapped
#       53,129 -- ~3,000 mutations per read push most reads below the top-20
#       threshold and the rank histogram flattens into a long tail: 3985,
#       2971, 2039, 1714, 1403, 1274, 1092, 1037, 937, 862, 795, 712, 732,
#       623, 649, 578, 546, 544, 471 for ranks 2..20)
#   fp16 flow comparison (e2ehuman16.jl, no quantization at all): 124,474
#   (94.97%) / 77,920 (59.45%) -- the e4m3 quantization costs ~12-23 reads
#   of 131,072 at these error rates, while the fp8 GEMM runs ~1.4x faster.
#   Top-1 only (--k=1, bench --reps=3 / run): the top-k kernel needs no
#   reservoir inserts and no list merge -> 0.93 ms/chunk (1072 GiB/s, 2.1x
#   the k=20 rate); compute-only 3.22 s (515.3 TFLOPS), e2e 3.52 s (~18%
#   faster than k = 20).  Accuracy: err = 0.05 -> 122,428 / 131,072 = 93.41%
#   (the k = 20 rank-1 count 122,429 +- 1 near-tie read: at this error rate
#   top-1 is nearly as good as any-of-20); err = 0.15 -> 54,975 / 131,072 =
#   41.94% (vs 59.47% at k = 20: under heavy mutation the long rank tail is
#   worth ~17.5 points of recall).
#   Context: the ~5% misses at err = 0.05 are consistent with ~1,000
#   mutations per read degrading the rope match plus e4m3's near-tie churn
#   (flowtopkfp8's within-noise location swaps) in repetitive regions; the
#   tiny self-test pins the location bookkeeping down exactly (err = 0 maps
#   1000/1000, every one at rank 1).
#
#   EVEN-DRAW reads + KSTEP SWEEP (executed): the indexsample default draw is
#   now the deterministic :even (record quotas ∝ window counts via largest
#   remainder, equal-strata midpoint starts, seed-free); the three s42 reads
#   batches (n = 131,072, k = 20,000) were REWRITTEN in place (66 of 68
#   records sampled vs 61-62 under the old :random draw) and score the same
#   as the random batches on the kstep-2000 index (sampling noise only).
#   Second index: the same reference/encoder at kstep = 1250
#   (indexflowreal save, 2,475,311 windows x2 = 4,950,622 columns, 9.57 GiB
#   fp8, encode 55.5 s):
#     err \ kstep (cols)     2000 (3,094,162)     1250 (4,950,622)
#     0.05  top-20 / top-1   95.04% / 93.52%      95.09% / 93.89%
#     0.10  top-20 / top-1   95.01% / 93.19%      95.08% / 93.76%
#     0.15  top-20 / top-1   59.64% / 42.05%      76.15% / 58.02%
#   The finer grid leaves the low-err plateau unchanged but buys ~+16.5
#   points of top-20 recall at err = 0.15: ~16 differently-offset index
#   windows overlap each 20 kb read (vs ~10 at kstep = 2000), so the odds
#   that at least one of them survives ~3,000 mutations rise sharply.  Full
#   kstep-1250 rank histograms (top-20): 0.05 -> 123056, 980, 353, 140, 19,
#   8, 9, 8, 2, 3, 11, 3, 7, 3, 5, 5, 6, 5, 3, 4 | 0.10 -> 122894, 1049,
#   384, 157, 30, 15, 11, 11, 9, 4, 0, 7, 4, 8, 9, 11, 1, 4, 4, 6 |
#   0.15 -> 76042, 4272, 2925, 2140, 1997, 1529, 1358, 1156, 1044, 961,
#   877, 792, 721, 665, 624, 604, 548, 533, 540, 487.
#
#   n = 2^20 CONFIRMATION (executed with julia -t 8: the flow is
#   thread-schedule-free, so 8 vs 16 threads changes only wall-clock):
#   fresh even-draw batches n = 1,048,576, k = 20,000, s = 42 (20.1 GiB per
#   file, 67 of 68 records sampled; gen 28-34 s + save ~8 s; verify bitwise
#   PASS) mapped over the same two indexes -- every rate within +-0.08 pp of
#   the n = 2^17 runs above (the estimates were already tight; 8x the reads
#   just shrink the error bars ~2.8x).  Top-20 / top-1:
#     err \ kstep            2000                   1250
#     0.05                   95.045% / 93.511%      95.069% / 93.892%
#     0.10                   95.015% / 93.204%      95.062% / 93.795%
#     0.15                   59.560% / 41.883%      76.168% / 57.973%
#
#   err = 0.125 (~2,500 mutations/read, n = 2^20, julia -t 8, page cache
#   warm; timing = map_and_count e2e, rep1 pays the process's one-time
#   pipeline JIT, rep2 warm): the dense index holds the plateau to ~2,500
#   mutations per read while the k/10 grid is already sliding off:
#     err \ kstep            2000                   1250
#     top-20                 90.508%                94.887%   (+4.4 pts)
#     top-1                  81.973%                92.704%   (+10.7 pts)
#     ktop-20 rank hist      859573, 23016, ...     972080, 10966, ...
#   TIMING (n = 2^20 reads = 20.1 GiB fasta, -t 8): index load 1.34 s (5.98
#   GiB) / 4.65 s (9.57 GiB); e2e mapping warm: kstep2000 34.05 s (ktop 20)
#   / 28.75 s (ktop 1), kstep1250 52.52 s / 44.36 s -- the 1.6x column count
#   scales the GEMM+top-k ~1.6x, ktop = 1 saves the reservoir-merge cost
#   (~16% faster), gen 36.9 s + save 8.8 s, verify (bitwise regen) PASS.
#
#   n = 2^20 REGENERATED (N-FREE) + INTERSECTION-LENGTH METRIC (executed
#   julia -t 16, warm page cache): the three s42 2^20 batches (e0.05/0.10/
#   0.15, even draw, k = 20,000, 20.1 GiB each) were RE-sampled with the
#   N-free pool after the pre-nfree files were deleted -- N-free windows
#   remove the fake-G N-run corruption, so every rate below sits a few pp
#   above the pre-nfree n = 2^20 rows above (the low-err top-20 plateau
#   becomes a PERFECT 100.00%).  map_and_count now also keeps the
#   intersection LENGTH of the first intersecting hit (kept when > 0) and
#   the run reports its average over the mapped reads as a % of the
#   20,000-base fragment.  Full matrix (n = 2^20 each row; the e0.125 batch
#   sampled + run the same way), per kstep: top-1 / top-20 correctly-mapped
#   % then avg first-hit intersection % (of 20,000) for that ktop (avg1 /
#   avg20):
#     err \ kstep    2000: top1   top20   avg1   avg20    1250: top1   top20   avg1   avg20
#     0.05                 98.45  100.00   97.50   97.48         98.85  100.00   98.43   98.42
#     0.10                 98.14   99.96   97.49   97.46         98.74  100.00   98.42   98.40
#     0.125                86.27   95.16   97.73   97.54         97.60   99.82   98.42   98.39
#     0.15                 43.99   62.56   98.35   98.06         60.93   80.08   98.61   98.48
#   (ks1250 top-20 at err = 0.05 mapped 1,048,576/1,048,576 -- the first
#   perfect whole-genome run; ks2000 top-20 dropped 25 reads.)  At
#   err = 0.125 (~2,500 mutations/read) the kstep-1250 grid still holds
#   97.60 top-1 / 99.82 top-20 while kstep-2000 already slides to 86.27 /
#   95.16 (the pre-nfree runs above showed the same ~+5 pp N-free lift at
#   this error rate: 81.97 -> 86.27 ks2000 top-1, 94.89 -> 99.82 ks1250
#   top-20).  The
#   intersection metric tracks the grid geometry: the first hit is the
#   nearest index window, offset ~ U(-kstep/2, +kstep/2), so the expected
#   intersection at low err is k - kstep/4 -> 19,500 bases = 97.50% at
#   kstep 2000 and 19,687.5 = 98.44% at kstep 1250 (measured 97.46-97.50%
#   and 98.39-98.43%).  From err = 0.125 upward it CLIMBS (to as much as
#   98.61 at err = 0.15): heavy mutation biases which hits survive into
#   the top-k toward better-aligned (smaller-offset) windows.  e2e mapping
#   warm: 33.4-39.6 s (ks2000), 48.9-58.1 s (ks1250).
#
#   ENCODER SWEEP s = 5, m = 1, c = 5 (executed julia -t 16, same reads
#   batches): indexflowreal save now takes the encoder config via env
#   (INDEXFLOW_S/M/C); e2ehuman reads s/m/c back from the .bin, so the flow
#   follows automatically (run_e2e_human's RopeEncoder is d.s/d.m/d.c;
#   run_e2e_bench likewise builds it as (d.k, d.s, d.m, d.c)).
#   New indexes: <fasta>.indexflowreal_fp8_s5m1c5.bin (kstep 2000,
#   3,094,162 cols, 5.95 GiB, encode 36.7 s) and
#   <fasta>.indexflowreal_fp8_s5m1c5_kstep1250.bin (4,950,622 cols,
#   9.52 GiB, encode 55.6 s) -- rdim = 2*1*4^5 = 2048 UNCHANGED (same B
#   shape, same GEMM/top-k geometry as the s8m4c4 indexes).  The single-
#   rotation m = 1 config collapses the high-err collapse entirely:
#     err \ kstep    2000: top1   top20   avg1   avg20    1250: top1   top20   avg1   avg20
#     0.05                 98.56   99.99   97.47   97.45         98.87   99.99   98.40   98.39
#     0.10                 98.35   99.90   97.46   97.44         98.68   99.93   98.38   98.37
#     0.15                 97.92   99.63   97.44   97.41         98.27   99.68   98.36   98.34
#     0.20                 97.45   99.19   97.41   97.39         97.76   99.24   98.32   98.30
#     0.20                 97.45   99.19   97.41   97.39         97.76   99.24   98.32   98.30
#     0.25                 96.98   98.71   97.37   97.35         97.32   98.74   98.28   98.26
#     0.30                 95.44   98.17   97.34   97.31         96.39   98.28   98.21   98.20
#   err = 0.2 (~4,000 mutations/read) keeps sliding gently down the same
#   slope: -0.47 pp top-1 / -0.44 pp top-20 vs err = 0.15 (ks2000), and the
#   avg intersection STAYS flat at k - kstep/4.  err = 0.25 / 0.3 (5,000 /
#   6,000 mutations/read): the slide steepens gently (top-1 ks2000 -0.47,
#   then -1.54 pp per 0.05 step) but there is NO collapse even at 30%
#   mutation -- top-20 saturates around 98.2-98.7 on BOTH ksteps (the
#   denser grid adds nothing there any more) while top-1 keeps the ks1250
#   edge (+0.34 at 0.25, +0.95 at 0.3); the avg intersection keeps
#   tracking k - kstep/4 minus a hair, i.e. rank-1 hits stay
#   nearest-neighbor-aligned at every error rate tested.
#   vs the s8m4c4 rows above, top-1 at err = 0.15 goes 43.99 -> 97.92
#   (ks2000) and 60.93 -> 98.27 (ks1250); top-20 goes 62.56 -> 99.63 /
#   80.08 -> 99.68.  At low err the two configs tie (~98.5-99.0 top-1).
#   The avg intersection is now FLAT across err (97.41-97.47 ks2000,
#   98.34-98.40 ks1250 -- still k - kstep/4): rank-1 dominance no longer
#   degrades with err (ks2000 top-20 at 0.15: 1,026,805 of 1,044,716 first
#   hits at rank 1), so the mutation-biased-hit-selection effect is gone.
#   The kstep grid barely matters at high err any more (+0.35 pp top-1 at
#   0.15 for ks1250) -- the encoder config dominates the error robustness.
#
#   ENCODER SWEEP AT err = 0.3 (executed julia -t 16, kstep = 2000, n = 2^20
#   e0.3 reads batch; all indexes built via the INDEXFLOW_S/M/C envs and kept
#   in the data dir as ..._fp8_s<m>s<c> names below, kstep-2000 fp8 saves):
#     encoder  bins  rdim  index     encode  top-1    top-20   avg1    avg20   e2e 1/20
#     s4m4c4   1024  2048  5.98 GiB   37 s   96.25%   99.25%   97.48%  97.38%  33.7/39.4 s
#     s5m1c5   1024  2048  5.95 GiB   37 s   95.44%   98.17%   97.34%  97.31%  33.6/39.5 s
#     s3m4c3    256   512  1.56 GiB   14 s   93.01%      --    97.53%     --    19.7 s
#     s2m4c2     64   128  0.45 GiB   12 s   72.81%      --    97.94%     --    17.8 s
#     s2m8c2    128   256  0.86 GiB   19 s   68.18%      --    98.03%     --    19.7 s
#     s8m4c4   1024  2048  5.98 GiB    --    43.99%*  62.56%*     --       --      --
#   (* already at err = 0.15 -- the old config collapses long before 0.3.)
#   Readings: (1) the s-mer length is the primary lever -- 4-mers edge out
#   5-mers (+0.81 top-1, +1.08 top-20), and both crush 3-mers (+3.2) and
#   2-mers (+23.4); the old 8-mer failure was long-s-mer fragility, not the
#   rotations.  (2) Rotation multiplicity m >= 4 adds nothing at low s and
#   slightly HURTS: s2m8c2 loses 4.6 pp to s2m4c2 -- 16 2-mers leave the
#   extra rotation rates nothing new to resolve while their higher phase
#   frequencies decorrelate faster under indel frame noise.  (3) The rdim =
#   512 config runs ~1.7x faster e2e (4x thinner GEMM) for -3.2 pp top-1;
#   below that the rope + 3.1M-column top-k dominate (s2m4c2 is only ~2 s
#   faster than s3m4c3 despite a 4x smaller GEMM again).  (4) The avg
#   intersection stays pinned at k - kstep/4 (~97.5%) for the healthy
#   configs; RISING above it (97.9-98.0% for the s2m*c2 runs) marks the
#   survivor-selection regime where only reads with near-maximal overlap
#   still map.
#
#   EVAL READS, COLON PROVENANCE (executed julia -t 8): the external eval
#   batch /share/q4bio/maksym/rotormap/generated_data/eval/human_1X_15.fasta
#   (119,489 reads generated from GCF_000001405.40_GRCh38.p14_primary25.fna,
#   1X over the 24 primary chromosomes, readlen 20,247..29,718, ~50% revcomp,
#   headers >record:start:revcomp, 0-based start, 2 duplicate-locus #2
#   suffixes) mapped over
#   GCF_000001405.40_GRCh38.p14_primary25.indexflowreal_fp8_s5m1c5.bin
#   (k = 20,000, kstep = 2,000, encoder (5,1,5), 3,087,812 columns x rdim
#   2048, 24 records, 5.89 GiB): the reader trims every record to its first
#   k = 20,000 bases (all reads >= k, none skipped; fragments of the revcomp
#   reads still overlap [start, start+k) since readlen < 2k),
#     MAPPED 119,289/119,489 = 99.83% correctly mapped in 10.4 s
#     rank histogram: 117512, 1068, 386, 158, 48, 30, 14, 13, 10, 6, 12, 8,
#       3, 5, 4, 4, 0, 3, 4, 1 | unmapped 200 (0.17%)
#     first-hit intersection: avg 17,671 bases = 88.36% of the fragment
#   (rank-1 alone = 98.35%; the avg intersection sits well below the
#   k - kstep/4 ~ 97.5% of the project's own err <= 0.3 sampler batches,
#   i.e. this eval generator's error model is substantially harsher, yet the
#   top-20 flow still places every read but ~1 in 600 on its true locus).
#
# Run modes (first non-flag ARGV[1]):
#   test  tiny end-to-end (1M reference -> index -> err=0 reads -> all mapped)
#   bench stage timings: rope-only, quantize/gemm/topk microbenches,
#         compute-only, end-to-end (+ mapping score)
#   run   production: human index + n131072_e0.05 reads -> mapping score
#   all   test + bench + run (default)
#   flags: --k=20 (top-k size), --w=65536 (B-column chunk), --batch=8192,
#          --segs=8, --engine=lt (mma needs w | N: the human N = 3,094,162
#          is not a power of two, so :lt is the production engine), --reps=1
#   env:   E2HUMAN_FASTA (reference fasta; defaults of the two paths below)
#          E2HUMAN_INDEX (index .bin; default <fasta>.indexflowreal_fp8.bin,
#          the fp8 mapping layout; legacy fp16-row files load via the same
#          code path)
#          E2HUMAN_READS (reads fasta; default
#          <fasta>.indexsample_n131072_k20000_e0.05_s42.fasta)
#
# USAGE
#   julia --project=. -t 16 test/e2ehuman.jl test
#   julia --project=. -t 16 test/e2ehuman.jl run --engine=lt
#   julia --project=. -t 16 test/e2ehuman.jl            # test + run
# ==============================================================================

include(joinpath(@__DIR__, "flowtopkfp8.jl")) # the whole fp8 flow: TopKEngine,
# topk_flow, rope_topk_stream (+ ropeflowreal/fastareads_v3 transitively:
# RopeEncoder, encode_frag_real_batch!, _ref_forward_pack, _next_rec3, _LUT3,
# Utils, Mmap, Random, CUDA)

using Random
using Printf
using Base.Threads

# ------------------------------------------------------------------------------
# Configuration (env overrides)
# ------------------------------------------------------------------------------
const E2H_FASTA = get(ENV, "E2HUMAN_FASTA",
                      "/share/q4bio/dandan/rotormap/data/GCA_000001405.29_GRCh38.p14_genomic.fasta")
# the index matrix: by DEFAULT the fp8 save (indexflowreal's INDEXFLOW_FP8
# production layout: transposed e4m3 database columns -- B verbatim, a pure
# H2D at load time).  The classic fp16 row layout still loads: the format tag
# inside the .bin dispatches (see index_database).
const E2H_INDEX_BIN = get(ENV, "E2HUMAN_INDEX",
                          string(splitext(E2H_FASTA)[1], ".indexflowreal_fp8.bin"))
# the mutated reads (indexsample `gen` output)
const E2H_READS = get(ENV, "E2HUMAN_READS",
                      string(splitext(E2H_FASTA)[1],
                             ".indexsample_n131072_k20000_e0.05_s42.fasta"))

const E2H_FORMAT = "indexflowreal.v1"        # the fp16-row .bin format tag
const E2H_FP8_FORMAT = "indexflowreal.fp8.v1" # the transposed-e4m3 .bin format tag

# tiny self-test fixture
const E2H_TEST_BASES = 1_000_000 # tiny reference: two records, total bases
const E2H_TEST_N = 1000          # tiny test reads
const E2H_TEST_SEED = 42

# ------------------------------------------------------------------------------
# Ground truth: TWO provenance header formats
#
#   1. indexsample: >read_<i> start=<1-based> len=<k> src=<record header sans '>'
#      (`src` is the record's verbatim header minus the leading '>'; everything
#      after " src=" is free text and never parsed)
#
#   2. colon eval (reads generated from GCF_000001405.40_GRCh38.p14_primary25
#      by maksym's simulator):
#      >record:start:revcomp       e.g. >NC_000001.11:80433546:false
#      (a re-drawn duplicate locus may carry a trailing `#<n>` suffix, e.g.
#      `>NC_000011.10:49059499:true#2` -- ignored: same provenance)
#      `record` is the record ACCESSION (the first whitespace-delimited token
#      of the '>' header, sans '>'), `start` is the 0-BASED offset of the
#      read's first base in the record (verified base-by-base against the
#      reference fasta), `revcomp` (true/false) marks the read as the reverse
#      complement of ref[start:start+readlen].  No length is carried: the
#      fasta reader keeps each record's FIRST k bases (records shorter than k
#      are skipped, longer ones are trimmed to exactly k), so the mapped
#      fragment covers ref[start:start+k] -- for a revcomp read its source
#      window sits at the END of ref[start:start+readlen], which always
#      overlaps ref[start:start+k] whenever readlen < 2k (true for every eval
#      batch: readlen <= 29,718 < 2*20,000).  The parser normalizes `start`
#      to 1-based and returns len = nothing; the scoring loop then uses the
#      dbk-long window at that start -- the db's rc columns are the SAME
#      locations, so no strand special-casing is needed.
# ------------------------------------------------------------------------------
function parse_read_head(h::AbstractString)
    startswith(h, ">") || return nothing
    if startswith(h, ">read_") # format 1: indexsample
        sp = findfirst(" src=", h)
        sp === nothing && return nothing
        fields = split(h[1:first(sp)-1])
        length(fields) >= 2 || return nothing
        kv = Dict{String,String}()
        for f in fields[2:end]
            eq = findfirst('=', f)
            eq === nothing && return nothing
            kv[String(f[1:eq-1])] = String(f[eq+1:end])
        end
        haskey(kv, "start") && haskey(kv, "len") || return nothing
        start = tryparse(Int, kv["start"])
        len = tryparse(Int, kv["len"])
        (start === nothing || len === nothing) && return nothing
        return (id = String(fields[1]), start = start, len = len,
                src = String(h[last(sp)+1:end]))
    end
    # format 2: colon eval provenance >record:start:revcomp[#<dup>]; `start`
    # is 0-based -> normalized to 1-based, length unknown (the reader trims
    # every kept record to exactly the db window length k); the optional
    # trailing #<n> duplicate-locus marker is stripped
    f = split(h, ':')
    length(f) == 3 || return nothing
    start = tryparse(Int, f[2])
    start === nothing && return nothing
    rev = split(f[3], '#')[1]
    rev in ("true", "false") || return nothing
    return (id = String(h), start = start + 1, len = nothing,
            src = String(f[1][2:end]))
end

# ------------------------------------------------------------------------------
# The index .bin -> resident fp8 database
# ------------------------------------------------------------------------------

"""
    load_index(path = E2H_INDEX_BIN)

Load an indexflowreal-format index .bin (Julia-serialized NamedTuple): either
the fp8 mapping layout (`format = "indexflowreal.fp8.v1"`: embeds8
(rdim, 2*n_frag) Float8_E4M3FN -- the database columns, quantized at save
time) or the classic fp16 row layout (`"indexflowreal.v1"`: embeds
(2*n_frag, rdim)).  This flow assumes (asserts) `normalize = 0`: unit-energy
rows => normalized database columns.
"""
function load_index(path::String = E2H_INDEX_BIN)
    isfile(path) || error("index .bin not found: $path (generate with " *
                          "indexflowreal.jl `save`)")
    d = Utils.load(path)
    fmt = get(d, :format, nothing)
    fmt in (E2H_FORMAT, E2H_FP8_FORMAT) ||
        error("not an $E2H_FORMAT/$E2H_FP8_FORMAT index: $path")
    d.normalize == 0 ||
        error("e2ehuman assumes a normalize = 0 index (got $(d.normalize)): " *
              "unit-energy rows are what makes B's columns normalized")
    rdim = 2 * d.m * 4^d.c
    N = 2 * d.n_frag
    if fmt == E2H_FP8_FORMAT
        @assert eltype(d.embeds8) == F8 "embeds8 eltype mismatch"
        @assert size(d.embeds8) == (rdim, N) "embeds8 shape mismatch"
    else
        @assert eltype(d.embeds) == (d.fp16 ? Float16 : Float32) "embeds eltype mismatch"
        @assert size(d.embeds) == (N, rdim) "embeds shape mismatch"
    end
    @assert length(d.heads) == length(d.starts) == length(d.strand) == N "meta length mismatch"
    @info "index loaded" file = basename(path) format = fmt k = d.k kstep = d.kstep n_frag =
        d.n_frag cols = N rdim encoder = (d.s, d.m, d.c) normalize = d.normalize
    return d
end

"""
    index_to_f8(embeds; block = 2^16) -> B::CuMatrix{F8}

Upload the index matrix (rows = reference-window encodings, fp16/fp32) as the
fp8 GEMM database B (columns = reference windows): per row-block H2D + GPU
transpose + e4m3 quantize, `F8.(Float32.(.))` -- the pipeline's exact
quantization model.  Peak device memory: B + one block; peak host memory: the
index matrix itself (12.7 GiB fp16 for the human genome).  Runs on the
default stream and returns synchronized.
"""
function index_to_f8(embeds::AbstractMatrix{T}; block::Int = 2^16) where {T<:Union{Float16,Float32}}
    N, rdim = size(embeds)
    B = CuMatrix{F8}(undef, rdim, N)
    for j0 in 1:block:N
        j1 = min(j0 + block - 1, N)
        db = CuArray{T}(@view embeds[j0:j1, :]) # (nb, rdim) H2D, contiguous rows
        @views B[:, j0:j1] .= F8.(Float32.(permutedims(db))) # fused transpose+quantize
    end
    CUDA.synchronize()
    return B
end

"""
    index_database(d) -> B::CuMatrix{F8}

The resident fp8 database for a loaded index.  For the fp8 mapping layout
("indexflowreal.fp8.v1") the stored matrix IS B -- (rdim, 2*n_frag) e4m3
columns, quantized at save time -- so this is a pure H2D with no transpose
and no arithmetic (the fast path: half the bytes of the fp16 rows, which
matters double on a PCIe Gen1 x16 link).  For the classic fp16 row layout
("indexflowreal.v1") it is built on the GPU blockwise (`index_to_f8`: H2D +
transpose + e4m3) -- bitwise the same bytes, at ~4x the build cost.
"""
function index_database(d)
    if get(d, :format, nothing) == E2H_FP8_FORMAT
        return CuArray(d.embeds8)
    end
    return index_to_f8(d.embeds)
end

# ------------------------------------------------------------------------------
# The mapping run: stream the reads, score the top-k against the provenance
# ------------------------------------------------------------------------------

"""
    map_and_count(re, reads_file, B, db_heads, db_starts, dbk; ktop, w,
                  batch_size, segs, engine, normalize, progress)
        -> (total, correct, rank_hist, seconds)

Run the full fp8 flow (`rope_topk_stream`: fasta -> rope fp16 -> e4m3 GEMM +
per-row top-k) over `reads_file` and score every row against its provenance
header (either format of `parse_read_head`): CORRECTLY MAPPED iff the true
window [start, start+len-1] in record src intersects at least one of the ktop
returned database windows (db_heads[c], db_starts[c], length dbk).  The
reader has already trimmed every kept record to exactly dbk bases (records
shorter than dbk are skipped), so the scored fragment window is dbk-long in
both formats -- for the colon format (no header length) it is the dbk-long
window at the (1-based-normalized) header start.  For every correctly mapped
read the BEST (maximum) intersection LENGTH over ALL its ktop returned hits
is also kept (when > 0) -- not just the first intersecting hit's.  Returns
the processed-read count, the number of correctly mapped records, the
first-hit rank histogram (rank_hist[j] = reads whose first intersecting hit
was rank j), the best-intersection-length total count (inter_sum /
inter_cnt; the average as a share of the fragment length is what the run
mode reports) and the processed-batch count nb.  `stream` selects the
batch producer (default: the resident-database `rope_topk_stream`;
e2ecompact passes its chunk-streaming variant with the same contract).
"""
function map_and_count(re::RopeEncoder, reads_file::String, B::AbstractMatrix{F8},
                       db_heads::Vector{String}, db_starts::Vector{Int}, dbk::Int;
                       ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, normalize::Int = 0,
                       progress::Bool = false,
                       stream::Function = (re, f, B; kw...) ->
                                           rope_topk_stream(re, f, B; kw...))
    re.k == dbk ||
        throw(ArgumentError("encoder k = $(re.k) != db window length $dbk"))
    N = size(B, 2)
    # per-column record id: the .bin's heads repeat a handful of unique
    # record headers over millions of columns -- intern them once.  The colon
    # eval provenance names records by ACCESSION (first whitespace token of
    # the '>' header, sans '>'), so the same ids are also keyed by accession.
    rec_id = Dict{String,Int32}()
    rec_id_acc = Dict{String,Int32}()
    rec_of_col = Vector{Int32}(undef, N)
    for j in 1:N
        h = db_heads[j]
        id = get(rec_id, h, Int32(0))
        if id == 0
            id = Int32(length(rec_id) + 1)
            rec_id[h] = id
            rec_id_acc[String(split(h)[1][2:end])] = id # accession sans '>'
        end
        rec_of_col[j] = id
    end
    total = correct = nb_bat = 0
    inter_sum = inter_cnt = 0
    rank_hist = zeros(Int, ktop)
    err = Ref{Any}(nothing)
    tks = Ref{Vector{Task}}(Task[])
    t = @elapsed for bat in stream(re, reads_file, B; k = ktop, w,
                                   batch_size, rows_cap = batch_size,
                                   normalize, segs, engine, progress,
                                   err_out = err, tasks_out = tks)
        nb_bat += 1
        nb = size(bat.vals, 1)
        for r in 1:nb
            h = parse_read_head(bat.heads[r])
            h === nothing &&
                error("unreadable provenance header: $(bat.heads[r])")
            # indexsample headers carry len == dbk; colon headers carry no
            # length -- the reader trimmed every kept record to dbk bases
            h.len === nothing || h.len == dbk ||
                error("read $(h.id): len = $(h.len) != db window length $dbk")
            rid = h.len === nothing ?
                  get(rec_id_acc, h.src, Int32(0)) : # colon: accession key
                  get(rec_id, string('>', h.src), Int32(0))
            hit = 0
            ilen = 0
            @inbounds for j in 1:ktop
                c = Int(bat.locs[r, j])
                # read window [h.start, h.start+dbk-1] vs db window
                # [db_starts[c], db_starts[c]+dbk-1] overlap?
                if rec_of_col[c] == rid &&
                   h.start <= db_starts[c] + dbk - 1 &&
                   db_starts[c] <= h.start + dbk - 1
                    hit == 0 && (hit = j) # first intersecting hit -> rank hist
                    # how MANY bases the two windows share (> 0 whenever the
                    # overlap test above fired); keep the BEST (max) overlap
                    # over ALL ktop hits, not just the first intersecting one
                    il = min(h.start + dbk - 1, db_starts[c] + dbk - 1) -
                         max(h.start, db_starts[c]) + 1
                    il > ilen && (ilen = il)
                end
            end
            total += 1
            if hit > 0
                correct += 1
                rank_hist[hit] += 1
                ilen > 0 && (inter_sum += ilen; inter_cnt += 1)
            end
        end
        @printf("  batch %2d: +%d reads, %d/%d correctly mapped so far\n",
                nb_bat, nb, correct, total)
    end
    foreach(wait, tks[]) # deterministic: all three pipeline stages unwound
    err[] === nothing || error("flow failed: $(err[])")
    return (total = total, correct = correct, rank_hist = rank_hist, seconds = t,
            inter_sum = inter_sum, inter_cnt = inter_cnt, nb = nb_bat)
end

# ------------------------------------------------------------------------------
# Tiny self-test fixture: a 1M-base two-record reference (built once), an
# indexflowreal-format index built from it, and err = 0 sampled reads
# ------------------------------------------------------------------------------

const _E2H_TEST_REF = joinpath(_v3_data_dir(), "e2ehuman_ref1M.fasta")

# two records (~70/30 split) so the record-discrimination half of the location
# check is exercised; indexflowreal's builder style (lowercase + IUPAC junk,
# wrapped lines -- its wrapped() emits width+1 chars per line, so the file
# holds ~1.4% more than `total` bases).  Cached: regenerated only when
# missing/too small.
function _ensure_e2h_test_ref(; total::Int = E2H_TEST_BASES)
    isfile(_E2H_TEST_REF) && filesize(_E2H_TEST_REF) > total + 10_000 &&
        return _E2H_TEST_REF
    lenA = 7 * total ÷ 10
    lenB = total - lenA
    rs = MersenneTwister(2024)
    bases = collect("ACGT")
    iupac = collect("NRYSWKMBDHV")
    randseq(len) = String([bases[rand(rs, 1:4)] for _ in 1:len])
    function sprinkle(s)
        v = collect(s)
        for i in 1:97:length(v)
            v[i] = lowercase(v[i])
        end
        for i in 41:211:length(v)
            v[i] = rand(rs, iupac)
        end
        return String(v)
    end
    wrapped(s, width) = join((s[i:min(i + width, end)] for i in 1:width:length(s)), "\n")
    mkpath(dirname(_E2H_TEST_REF))
    @info "Generating the tiny $(total)-base test reference -> $_E2H_TEST_REF"
    open(_E2H_TEST_REF, "w") do io
        write(io, ">chrA e2ehuman tiny test record A\n",
              wrapped(sprinkle(randseq(lenA)), 70), "\n")
        write(io, ">chrB e2ehuman tiny test record B\n",
              wrapped(sprinkle(randseq(lenB)), 70), "\n")
    end
    return _E2H_TEST_REF
end

# record table of a fasta in the indexflowreal sense: (verbatim '>' heads,
# inclusive sequence byte spans, sequence-char counts -- every byte except
# \n/\r; junk/IUPAC -> G keeping its position)
function _e2h_ref_table(raw::Vector{UInt8})
    heads = String[]
    lo = Int[]
    hi = Int[]
    nchar = Int[]
    NL = UInt8('\n')
    CR = UInt8('\r')
    n = length(raw)
    s = _next_rec3(raw, 1, n) # junk before the first record is discarded
    while s <= n
        h = s
        while h <= n && raw[h] != NL
            h += 1
        end
        hend = h - 1
        (hend >= s && raw[hend] == CR) && (hend -= 1)
        e = _next_rec3(raw, s + 1, n)
        cnt = 0
        @inbounds for p in (h+1):(e-1)
            b = raw[p]
            (b == NL || b == CR) || (cnt += 1)
        end
        push!(heads, String(raw[s:hend]))
        push!(lo, h + 1)
        push!(hi, e - 1)
        push!(nchar, cnt)
        s = e
    end
    return (heads, lo, hi, nchar)
end

# scalar scan of the record span [lo, hi] into 2-bit codes (the _LUT3
# convention; deliberately not the production SWAR path)
function _e2h_span_codes(raw::Vector{UInt8}, lo::Int, hi::Int)
    codes = Vector{UInt8}()
    @inbounds for p in lo:hi
        c = raw[p]
        (c == UInt8('\n') || c == UInt8('\r')) && continue
        push!(codes, _LUT3[Int(c) + 1])
    end
    return codes
end

"""
    _e2h_sample_reads(ref; n, k, err = 0.0, seed, out) -> out::String

Sample `n` reads of length `k` from the tiny reference (indexsample's
:random draw: n iid uniform windows, with replacement), optionally mutate each
with Utils.mutate (err = fraction; err = 0 keeps the exact window), and save
as plain fasta with indexsample's provenance headers.  Deterministic in
(ref, n, k, err, seed).  The output name mirrors indexsample's parameter
record so the ground truth stays self-describing.
"""
function _e2h_sample_reads(ref::String; n::Int, k::Int, err::Real = 0.0,
                           seed::Int = E2H_TEST_SEED,
                           out::String = string(splitext(ref)[1],
                                                ".indexsample_n$(n)_k$(k)_e$(err)_s$(seed).fasta"))
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    w = [max(l - k + 1, 0) for l in rlen] # sampling weights: valid windows
    total = sum(w)
    total > 0 || error("reference too short for k = $k")
    cum = accumulate(+, w)
    rng = Xoshiro(seed)
    recs = Vector{Int}(undef, n)
    starts = Vector{Int}(undef, n)
    for i in 1:n # every choice up front: thread-schedule-free determinism
        r = rand(rng, Int64(1):total)
        j = searchsortedfirst(cum, r)
        recs[i] = j
        starts[i] = Int(r - (j > 1 ? cum[j-1] : Int64(0)))
    end
    rngs = [Xoshiro(rand(rng, Int64)) for _ in 1:n] # one child rng per read
    reads = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        codes = _e2h_span_codes(raw, rlo[recs[i]], rhi[recs[i]])
        win = codes[starts[i]:starts[i]+k-1]
        err > 0 && (win = Utils.mutate(win, err; rng = rngs[i]))
        reads[i] = win
    end
    heads = [">read_$(i) start=$(starts[i]) len=$(k) src=$(rheads[recs[i]][2:end])"
             for i in 1:n]
    Utils.save_fasta(reads, out; heads)
    @info "sampled $n reads (err = $err) -> $out"
    return out
end

"""
    _e2h_build_index(ref; k, kstep, normalize = 0, binfile) -> binfile::String

A miniature of indexflowreal's fp8 `save`: every k-window (step kstep, last
window fully inside the record) of every record of the tiny reference,
forward AND reverse complement, encoded with the PRODUCTION kernel
(`encode_frag_real_batch!`), transposed + quantized to e4m3 columns and
serialized under the production naming `<ref>.indexflowreal_fp8.bin`
(format "indexflowreal.fp8.v1") -- so the test exercises the same
load path as the human run.
"""
function _e2h_build_index(ref::String; k::Int = 20_000, kstep::Int = max(k ÷ 10, 1),
                          normalize::Int = 0,
                          binfile::String = string(splitext(ref)[1], ".indexflowreal_fp8.bin"))
    isfile(binfile) && return binfile # shared fixture: a previous run built it
    re = RopeEncoder(k = k, s = 8, m = 4, c = 4) # the workflow's config
    rdim = 2 * re.m * 4^re.c
    W = cld(k, 16)
    raw = open(ref, "r") do io
        Mmap.mmap(io)
    end
    (rheads, rlo, rhi, rlen) = _e2h_ref_table(raw)
    gheads = String[] # per-window location meta (file order)
    gstarts = Int[]
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        for wi in 0:((rlen[j] - k) ÷ kstep)
            push!(gheads, rheads[j])
            push!(gstarts, wi * kstep + 1)
        end
    end
    nwin = length(gheads)
    # a 1M reference holds ~nwin windows: materialize ALL fragment words
    # (a few MB) and encode both strands in one kernel call each
    fw = Vector{UInt32}(undef, W * nwin)
    rcw = Vector{UInt32}(undef, W * nwin)
    g = 0
    for j in eachindex(rheads)
        rlen[j] >= k || continue
        codes = _e2h_span_codes(raw, rlo[j], rhi[j]) # full record, once
        @assert length(codes) == rlen[j]
        for wi in 0:((rlen[j] - k) ÷ kstep)
            g += 1
            s = wi * kstep
            copyto!(fw, (g - 1) * W + 1, _ref_forward_pack(codes[s+1:s+k], k), 1, W)
            rcc = [UInt8(0x03 - codes[s+k+1-t]) for t in 1:k] # revcomp codes
            copyto!(rcw, (g - 1) * W + 1, _ref_forward_pack(rcc, k), 1, W)
        end
    end
    @assert g == nwin
    dest_f = CUDA.zeros(Float16, nwin, rdim)
    dest_r = CUDA.zeros(Float16, nwin, rdim)
    dn_f = CUDA.zeros(Float32, re.m, nwin)
    dn_r = CUDA.zeros(Float32, re.m, nwin)
    encode_frag_real_batch!(dest_f, dn_f, re, cu(fw); normalize)
    encode_frag_real_batch!(dest_r, dn_r, re, cu(rcw); normalize)
    ef = Matrix{Float16}(undef, nwin, rdim)
    er = Matrix{Float16}(undef, nwin, rdim)
    nf = Matrix{Float32}(undef, re.m, nwin)
    nr = Matrix{Float32}(undef, re.m, nwin)
    copyto!(ef, dest_f)
    copyto!(er, dest_r)
    copyto!(nf, dn_f)
    copyto!(nr, dn_r)
    # transposed e4m3 columns -- the production save's layout and the exact
    # F8.(Float32.(.)) quantization (rows 1:nwin fwd, nwin+1:2nwin rc)
    embeds8 = Matrix{F8}(undef, rdim, 2 * nwin)
    @views embeds8[:, 1:nwin] .= F8.(Float32.(permutedims(ef)))
    @views embeds8[:, nwin+1:2*nwin] .= F8.(Float32.(permutedims(er)))
    norms = Matrix{Float32}(undef, re.m, 2 * nwin)
    copyto!(view(norms, :, 1:nwin), nf)
    copyto!(view(norms, :, nwin+1:2*nwin), nr)
    nt = (format = E2H_FP8_FORMAT, source = abspath(ref), k, kstep, s = re.s,
          m = re.m, c = re.c, normalize, fp8 = true, n_frag = nwin, embeds8,
          norms, heads = vcat(gheads, gheads), starts = vcat(gstarts, gstarts),
          strand = vcat(fill(UInt8(0), nwin), fill(UInt8(1), nwin)))
    Utils.save(nt, binfile)
    @info "tiny index built" binfile k kstep nwin cols = 2 * nwin rdim
    return binfile
end

# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

"""
Tiny end-to-end: build the 1M reference (once), index it, sample 1000 err = 0
reads, run the FULL load -> fp8 -> stream -> score path, and assert every
read is correctly mapped (its window shares >= 95% of its bases with an index
window -- any miss means broken location bookkeeping, not noise).
"""
function run_e2e_test(; kfrag::Int = 20_000, n::Int = E2H_TEST_N, ktop::Int = 20,
                      w::Int = 2^16, batch_size::Int = 2^13, segs::Int = 8,
                      engine::Symbol = :lt, seed::Int = E2H_TEST_SEED)
    @show CUDA.name(device())
    @show nthreads()
    ref = _ensure_e2h_test_ref()
    binfile = _e2h_build_index(ref; k = kfrag)
    readsf = _e2h_sample_reads(ref; n, k = kfrag, err = 0.0, seed)
    d = load_index(binfile)
    B = index_database(d)
    re = RopeEncoder(k = d.k, s = d.s, m = d.m, c = d.c)
    N = size(B, 2)
    w_test = min(w, prevpow(2, N)) # the tiny N (≈10^3) admits no production w
    res = map_and_count(re, readsf, B, d.heads, d.starts, d.k; ktop,
                        w = w_test, batch_size, segs, engine)
    @assert res.total == n "processed $(res.total) != $n reads"
    @assert res.correct == n "err = 0 sanity FAILED: only $(res.correct)/$n " *
                             "correctly mapped"
    @info "TINY E2E PASSED: $n/$n correctly mapped" first_hit_ranks = res.rank_hist
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

"""
    run_e2e_bench(; ktop, w, batch_size, segs, engine, reps)

Stage timings of the production mapping flow: the fp8 index -> resident B
upload, the rope stage alone (fasta read + 2-bit pack + rope encode, no-op
consumer; GB/s of fasta), per-chunk microbenches (quantize+upload, fp8 GEMM,
segmented top-k), the GEMM+top-k stage alone (compute-only: the materialized
embedding batches replayed through the engine), and the full end-to-end
pipeline (including the ground-truth score).  The rope stage is a separate
TASK pipeline, so in the pipelined flow it hides behind the compute stage
(e2e ~= compute-only); the GEMM and top-k overlap each other through the
double-buffered chunk streams, so their split is visible only in the
microbenches.
"""
function run_e2e_bench(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt, reps::Int = 1)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2H_READS) ||
        error("reads fasta not found: $E2H_READS (generate with indexsample.jl)")
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    t_B = @elapsed B = index_database(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    (ds, dm, dc) = (d.s, d.m, d.c) # bench follows the index's encoder config
    @info "fp8 database" dims = size(B) gib = round(length(B) / 2^30; digits = 2) seconds = round(t_B; digits = 2) load_seconds = round(t_load; digits = 2)
    d = nothing
    GC.gc(); CUDA.reclaim()

    re = RopeEncoder(k = dk, s = ds, m = dm, c = dc)
    N = size(B, 2)
    rdim = size(B, 1)
    chunks = cld(N, w)

    # warm the page cache: on the network /share a cold 2.6 GB read runs at
    # ~0.6 GB/s and would pollute the rope-stage timing with disk bandwidth
    @info "warming the reads fasta page cache..."
    read(E2H_READS)

    # ---- rope stage alone (no-op consumer) + materialize for the replay ----
    # (the first pass pays the pipeline's JIT/first-call cost (~4 s: GPU rope
    # kernel, channels, parser paths), so it is the warmup AND the
    # materialization pass; `reps` timed passes follow, min reported)
    batches = RopeRealBatch{Float16}[]
    t_jit = @elapsed for nt in rope_encode_real_stream(re, E2H_READS; k = re.k,
                                                       batch_size,
                                                       normalize = 0, fp16 = true)
        push!(batches, nt)
    end
    nreads = sum(nt -> size(nt.embeds, 1), batches)
    bytes = filesize(E2H_READS)
    t_rope = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for nt in rope_encode_real_stream(re, E2H_READS; k = re.k, batch_size,
                                              normalize = 0, fp16 = true)
                n += length(nt.heads)
            end
            @assert n == nreads
        end
        t_rope = min(t_rope, t)
    end
    @printf("rope stage  (fasta read + 2-bit pack + rope encode): %.2f s  (%.2f GB/s of fasta, %d reads -> %d batches; JIT pass %.2f s)\n",
            t_rope, bytes / 1e9 / t_rope, nreads, length(batches), t_jit)

    # ---- per-chunk microbenches at the real geometry -----------------------
    eng = TopKEngine(B; k = ktop, w, rows_cap = batch_size, segs, engine)
    emb1 = batches[1].embeds
    @assert size(emb1) == (batch_size, rdim)
    Bp1 = byteptr(pointer(B))
    CUDA.stream!(eng.sg) do
        upload_fp16_as_f8!(eng, emb1)
        _gemm_chunk!(eng, eng.buf1, Bp1, w) # warm (JIT + the lt heuristic)
    end
    CUDA.device_synchronize()
    tq = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            upload_fp16_as_f8!(eng, emb1)
        end
    end
    tg = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            _gemm_chunk!(eng, eng.buf1, Bp1, w)
        end
    end
    D_val = CuMatrix{Float32}(undef, batch_size, ktop)
    D_loc = CuMatrix{Int32}(undef, batch_size, ktop)
    launch_seg_rowtopk_merge!(D_val, D_loc, eng.buf1, 0, ktop; segs) # warm (JIT)
    CUDA.device_synchronize()
    tt = _timed_gpu(20) do
        CUDA.stream!(eng.sg) do
            launch_seg_rowtopk_merge!(D_val, D_loc, eng.buf1, 0, ktop; segs)
        end
    end
    @printf("  quantize+upload: %.2f ms/batch | fp8 gemm: %.2f ms/chunk (%.1f TFLOPS) | seg top-k: %.2f ms/chunk (%.0f GiB/s)\n",
            tq * 1e3, tg * 1e3, 2.0 * batch_size * rdim * w / tg / 1e12,
            tt * 1e3, batch_size * w * sizeof(eltype(eng.buf1)) / tt / 2^30)

    # ---- compute-only: GEMM+top-k stage, rope excluded (replayed batches) --
    t_cmp = Inf
    for _ in 1:reps
        t = @elapsed begin
            n = 0
            for bat in topk_flow(batches, eng; out_cap = 2)
                n += size(bat.vals, 1)
            end
            @assert n == nreads
        end
        t_cmp = min(t_cmp, t)
    end
    flops = 2.0 * nreads * rdim * N
    @printf("gemm+topk stage (quantize+H2D -> %d chunks/batch, gemm || topk): %.2f s  (%.1f TFLOPS sustained over 2^%.1f FLOP)\n",
            chunks, t_cmp, flops / t_cmp / 1e12, log2(flops))

    # ---- end-to-end: the full pipelined flow (and the mapping score) -------
    t_e2e = Inf
    correct = total = 0
    for _ in 1:reps
        res = map_and_count(re, E2H_READS, B, heads, starts, dk; ktop, w,
                            batch_size, segs, engine)
        t_e2e = min(t_e2e, res.seconds)
        correct = res.correct
        total = res.total
    end
    @printf("END-TO-END (fasta -> rope -> fp8 gemm -> topk -> score): %.2f s  (%d/%d = %.2f%% correctly mapped)\n",
            t_e2e, correct, total, 100 * correct / total)
    hides = t_e2e < t_rope + t_cmp ? "hides completely behind the compute stage" : "is (partially) exposed"
    @printf("stage budget: rope-only %.2f s + compute-only %.2f s vs e2e %.2f s -> the rope stage %s\n",
            t_rope, t_cmp, t_e2e, hides)
    eng = B = batches = nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

"""
Production run: load the human index (whole matrix, fwd + rc), build the fp8
database on the GPU, stream the indexsample reads through the fp8 flow and
report THE number of correctly mapped records (plus rate, first-hit rank
histogram and best top-k intersection).
"""
function run_e2e_human(; ktop::Int = 20, w::Int = 2^16, batch_size::Int = 2^13,
                       segs::Int = 8, engine::Symbol = :lt)
    @show CUDA.name(device())
    @show nthreads()
    isfile(E2H_READS) ||
        error("reads fasta not found: $E2H_READS (generate with indexsample.jl)")
    @info "loading the human index" E2H_INDEX_BIN
    t_load = @elapsed d = load_index(E2H_INDEX_BIN)
    @info "building the fp8 database on GPU"
    t_B = @elapsed B = index_database(d)
    heads = d.heads
    starts = d.starts
    dk = d.k
    re = RopeEncoder(k = dk, s = d.s, m = d.m, c = d.c)
    @info "fp8 database built" dims = size(B) gib = round(length(B) / 2^30; digits = 2) seconds = round(t_B; digits = 1) load_seconds = round(t_load; digits = 1) unique_records = length(Set(heads))
    d = nothing # the 12.7 GiB host index matrix can go
    GC.gc(); CUDA.reclaim()
    res = map_and_count(re, E2H_READS, B, heads, starts, dk; ktop, w,
                        batch_size, segs, engine, progress = true)
    @printf("MAPPED %d/%d reads correctly (%.2f%%) in %.1f s\n",
            res.correct, res.total, 100 * res.correct / res.total, res.seconds)
    @printf("  first-hit rank histogram (ranks 1..%d): %s | unmapped: %d\n",
            ktop, res.rank_hist, res.total - res.correct)
    res.inter_cnt > 0 && @printf(
        "  best top-%d intersection: avg %d bases over %d mapped reads = %.2f%% of the %d-base fragment\n",
        ktop, round(Int, res.inter_sum / res.inter_cnt), res.inter_cnt,
        100 * res.inter_sum / res.inter_cnt / dk, dk)
    B = nothing
    GC.gc(); CUDA.reclaim()
    return res
end

# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    function getflag(name::String, default::String)
        for a in ARGS
            startswith(a, "--$name=") && return String(split(a, '=')[2])
        end
        return default
    end
    ktop = parse(Int, getflag("k", "20"))
    w = parse(Int, getflag("w", string(2^16)))
    batch = parse(Int, getflag("batch", string(2^13)))
    segs = parse(Int, getflag("segs", "8"))
    engine = Symbol(getflag("engine", "lt"))
    reps = parse(Int, getflag("reps", "1"))
    modeargs = filter(a -> !startswith(a, "--"), ARGS)
    mode = isempty(modeargs) ? "all" : modeargs[1]
    mode == "test" && run_e2e_test(; ktop, w, batch_size = batch, segs, engine)
    mode == "bench" && run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps)
    mode == "run" && run_e2e_human(; ktop, w, batch_size = batch, segs, engine)
    mode == "all" && (run_e2e_test(; ktop, w, batch_size = batch, segs, engine);
                      run_e2e_bench(; ktop, w, batch_size = batch, segs, engine, reps);
                      run_e2e_human(; ktop, w, batch_size = batch, segs, engine))
    mode in ("test", "bench", "run", "all") ||
        error("unknown mode $mode (use test|bench|run|all)")
end
