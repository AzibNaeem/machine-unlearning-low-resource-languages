# Step 4: VRAM cost of embedding-layer noise

Code: `text/vram_probe.py`. Raw: `outputs/step4/C/vram.json`, `vram.log`. RTX 3080 10 GB, torch 2.6 cu124, seed-0 baseline checkpoint, fp32, AdamW as in train_baseline.py. Numbers are `max_memory_allocated` (GB) over 3 timed steps after 1 warm-up step; reserved is in the json. Other processes held about 1.4 GB of the card during the run (nvidia-smi before start), so OOM limits below are conservative by that amount; allocated peaks are unaffected.

## Headline
**No, the noise does not force a plan change.** Feeding half a batch of N(0,1) embeddings through the encoder adds at most 0.014 GB of peak VRAM over plain training at the same B and L (the noise tensor itself is 0.1 to 0.6 MB per sample; activations are the same as for real data). Impair at B=32, L=164 peaks at 7.28 GB, inside 10 GB.

## 1. Impair-like step (half real via `embed()`, half `torch.randn`, through `forward_from_embeddings`, CE, backward, AdamW)
| B | L | plain peak | mixed real+noise peak | extra | noise tensor | step time (s) plain / mixed |
|---|---|---|---|---|---|---|
| 16 | 164 | 5.714 | 5.728 | 0.014 | 4.8 MB (8 samples) | 0.141 / 0.141 |
| 32 | 164 | 7.274 | 7.281 | 0.007 | 9.6 MB | 0.234 / 0.234 |
| 16 | 64 | 5.227 | 5.228 | 0.001 | 1.5 MB | 0.081 / 0.080 |
| 32 | 64 | 5.358 | 5.363 | 0.005 | 3.0 MB | 0.114 / 0.114 |
| 16 | 32 | 5.206 | 5.209 | 0.003 | 0.75 MB | 0.065 / 0.065 |
| 32 | 32 | 5.227 | 5.228 | 0.001 | 1.5 MB | 0.082 / 0.081 |

Plain B=16, L=164 reproduces the 5.71 GB baseline figure. About 5.2 GB is a fixed floor (weights, grads, AdamW states; the 250k-token embedding matrix dominates), so shorter L saves little until B is large. The noise mask is all ones (position 0 unmasked).

## 2. USI(F)R noise learning (model frozen, eval, noise leaf tensor, Adam lr 0.1)
Peak VRAM GB / step time s:
| noise batch | L=32 | L=64 | L=164 |
|---|---|---|---|
| 16 | 1.26 / 0.017 | 1.44 / 0.028 | 2.05 / 0.075 |
| 64 | 1.83 / 0.054 | 2.60 / 0.099 | 5.05 / 0.273 |
| 256 | 4.15 / 0.192 | 7.25 / 0.370 | OOM |

Largest noise batch that fit (search, multiple of 8): L=32 -> 1216, L=64 -> 608, L=164 -> 224 (with ~1.4 GB used by other processes).

## 3. Noise-set storage (fp32, L x 768 x 4 bytes per sample)
| L | per sample | N=600 | N=1200 | N=5000 | N=18000 | max in 10 GB |
|---|---|---|---|---|---|---|
| 32 | 0.094 MB | 0.055 GB | 0.11 | 0.46 | 1.65 | 109k |
| 64 | 0.188 MB | 0.11 | 0.22 | 0.92 | 3.30 | 54.6k |
| 164 | 0.481 MB | 0.28 | 0.56 | 2.35 | 8.45 | 21.3k |

Impair training already uses 5.7 to 7.3 GB, so a 2x-retain set of ~1200 samples fits on GPU at any L (at most 0.56 GB). Sets of 5000+ at L=164 do not fit beside training; keep them on CPU. Moving a pinned (64, 164, 768) batch to the GPU took 1.2 ms against 140 to 270 ms per step, so CPU storage is free.

## 4. bf16 autocast (reported only; fp32 stays default)
| Setting | fp32 peak | bf16 peak |
|---|---|---|
| impair B16 L164 (plain / mixed) | 5.714 / 5.728 | 5.375 / 5.376 |
| impair B32 L164 (plain / mixed) | 7.274 / 7.281 | 6.377 / 6.378 |
| noise learning n64 L164 | 5.048 | 3.719 |
| noise learning n256 L64 | 7.251 | 5.085 |

Saves 0.3 to 0.9 GB for impair and about 30 percent for noise learning (also about 2x faster there); not needed to fit.

## 5. Sentence length (SIB-200 English train, 701 sentences, XLM-R tokens incl. `<s>`,`</s>`)
mean 31.99, median 30, p95 49, p99 60, max 74, min 12. Baseline L=164 is mostly padding.

## Recommendation
- **L = 64** for real data (covers max 74 after truncating only a handful; use 80 to truncate nothing) and for noise. It cuts the noise-learning cost 2x versus 164 and impair step time by about 40 percent. L=164 also works if exact baseline parity is preferred. Noise must keep position 0 unmasked (mask of ones; the head reads only `<s>`); real rows keep their own mask.
- **Batches:** impair B=16 to 32 (7.3 GB at L=164, 5.4 GB at L=64); noise learning batch 64 at L=64 (2.6 GB, 0.1 s/step) or up to 256 (7.3 GB).
- **Noise set:** 2x retain (~1200 samples, L=64: 0.22 GB) can live on GPU; use CPU plus per-batch transfer for anything larger.

## Not measured / caveats
- Real end-to-end impair (many steps, LR schedule) not run; only 3 timed steps per setting with lr 1e-6, which does not affect memory.
- Noise half of batch only; the peak is shape-determined, so an all-noise batch would give the same peak.
- Peaks at L=32 use truncated real sentences (the median length is 30, so roughly 40 percent of sentences exceed 32 tokens).
- Largest-batch limits were found while ~1.4 GB was held by other processes.
