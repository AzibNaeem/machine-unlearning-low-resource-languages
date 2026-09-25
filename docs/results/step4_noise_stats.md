# Step 4 (Worker B): embedding-layer statistics and noise scale

Output of `model.roberta.embeddings` (word + position + token-type, LayerNorm, dropout off), XLM-R base fine-tuned baseline (`outputs/final_run/text/baseline_eng_Latn_seed0.pt`), SIB-200 eng_Latn train (701 sentences, 22,421 real tokens, 92,683 pad positions at max_len 164). Code: `text/embed_stats.py`; raw numbers: `outputs/step4/B/stats.json` (+ `stats_finetuned.pt`).

## Real activations
| Quantity (real tokens) | Value |
|---|---|
| per-token L2 norm | mean 6.31, std 0.28, p1 5.83, p50 6.34, p99 6.84 (very tight) |
| N(0,1) per-token norm | sqrt(768) = 27.7, so real is **0.228x** (N(0,1) is 4.4x too large, ~19x in energy) |
| overall scalar mean / std | -0.003 / 0.228 (rms 0.228) |
| per-dim mean | mean -0.003, range [-0.44, +0.34], std across dims 0.068 |
| per-dim std | mean 0.212, median 0.211, range [0.032, 1.266] |
| outlier dims | dim 259 has std 1.27 (6x median; 4.4% of total energy alone), dim 588 mean -0.44; otherwise no heavy tail (top 10 dims = 7.2% of energy) |
| anisotropy | mean-vector norm 1.89; per-token cosine to mean vector 0.30 |
| LayerNorm gamma / beta | gamma mean 0.295 (0.025 to 1.0), beta mean 0.0, std 0.12 (max 0.93) |
| sentence length (real tokens) | mean 32.0, std 9.7, min 12, max 74 |

LayerNorm explains the scale: after normalisation each token has unit variance, then gamma (mean 0.29) shrinks it, so norm ~ sqrt(sum gamma^2) ~ 6.3. That is why the norm is so uniform across tokens. Per-token norm is set by the LN parameters, not by the token.

Pad positions: a single constant vector (std 0), norm 11.2 (1.8x a real token), cosine -0.23 to the real mean vector. With padding to 164, 80.5% of positions are pad, real fraction 19.5%, so any statistic over padded positions is dominated by the pad vector (mean norm 10.3 instead of 6.3). Stats above are over real tokens only.

Pretrained (not fine-tuned) encoder: embedding stats are essentially identical (norm 6.312, scalar std 0.228, same outlier dim 259). Fine-tuning barely changes this layer, so the stats transfer across seeds/checkpoints.

## Noise into encoder + head (512 samples, L=32, all-ones mask, fine-tuned model)
API verified: own forward (`create_bidirectional_mask` + `roberta.encoder` + `classifier`) reproduces model logits exactly (max diff 0.0) and test acc 91.18. Note `model.roberta(inputs_embeds=...)` is NOT a bypass: it still adds position + token-type embeddings and applies LayerNorm.

| Noise | class histogram (ent,geo,health,pol,sci/tech,sport,travel) | mean max-prob | entropy (max 1.946) |
|---|---|---|---|
| N(0,1) | 0, 8, 35, 2, 455, 4, 8 | 0.234 | 1.88 |
| N(0,1) x 0.228 (match_norm) | 0, 0, 164, 4, 344, 0, 0 | 0.245 | 1.88 |
| scalar std (0.228) | 0, 0, 154, 1, 356, 1, 0 | 0.247 | 1.87 |
| per-dim mean/std | 0, 9, 37, 7, 458, 0, 1 | 0.290 | 1.84 |
| LN-like (norm, x gamma + beta) | 0, 3, 25, 2, 481, 0, 1 | 0.223 | 1.90 |
| match_norm x0.25 / x0.5 | all 512 sci/tech | 0.20 | 1.92-1.93 |
| match_norm x2 | 0, 1, 39, 2, 459, 6, 5 | 0.240 | 1.88 |

## Findings
- Every noise type gives a near-uniform posterior (max-prob 0.22-0.29 vs 0.14 uniform) and collapses mostly onto sci/tech (the majority train class, 25%), then health. The model is not confident on any noise; forget-class labelling of noise therefore has a large loss to reduce, so impair should matter, but no noise looks like a forget-class input.
- Scale matters less than expected for the output distribution (post-LN blocks renormalise), but it matters for the input to the first attention/FFN and for gradients. Differences between N(0,1), norm-matched, per-dim are small in entropy; per-dim is the only mode with clearly higher confidence (0.29) and is closest to real in structure.
- Uncertainties: one checkpoint, one language; predictions on noise depend on L (32 chosen to match mean real length) and on the all-ones mask; the pretrained model's head is random so noise behaviour was not evaluated for it; nothing here shows what impair does to retain accuracy.

## Recommendation
- Default generator: `make_noise(shape, stats, mode='per_dim')` (mean + std per dimension, real-token stats, scale 1.0); `match_norm` (N(0,1) x 0.228, no stats needed beyond one scalar) is the simple fallback. Both put noise at the real scale (norm ~6.3), so the learned-noise optimiser (USI(F)R) starts on the data manifold's scale and a step of size ~0.1 is not negligible against 0.23 per-dim std.
- Keep plain N(0,1) as the paper-faithful ablation, not the default: it is 4.4x off in norm; it is not "wrong" for the classifier output (similar histogram) but dominates the first residual stream.
- Inject at L ~ 32 with an all-ones mask (or the real mask lengths), not 164 padded positions: 80% of a padded real input is one constant vector.
- Use one fixed stats file (`stats_finetuned.pt`), computed on the retain/train split of the model being unlearned.

## Open questions
- Does learned noise stay near the real norm, or drift (add an L2 penalty relative to 6.3, not to 0)?
- Whether noise with a forget label at real scale gives clean forgetting (step 5 tests this; not run here).
- Dim 259 outlier: keep per-dim std (recommended) or clip.
