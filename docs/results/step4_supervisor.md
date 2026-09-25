# Step 4 supervisor report (2026-09-25)

## Verdict
Step 4 (plumbing) is MET; VRAM measurement MET. Nothing here says anything about forgetting. Evidence: one checkpoint (XLM-R, English SIB-200, seed 0, val 87.88), one language, one seed, 99 validation sentences.

- Plumbing: `embed` + `forward_from_embeddings` reproduce the normal logits with max abs diff 0.0 on 99/99 validation sentences (re-run by me, 5/5 tests pass); gradients reach noise with frozen weights; pad-position values do not affect logits. MET.
- VRAM: re-run by me, impair B16 L164: plain 5.714 GB, mixed real+noise 5.728 GB (+0.014); noise learning n64 L64 2.602 GB. Matches C. Plan does not change. MET.
- Noise realism: real embedding-output token norm 6.31 (re-run: 6.312), scalar std 0.228, mean real length 31.99: match B. N(0,1) has norm 27.7. MET as measurement.

## Independent verification of the API claims (transformers 5.17 source)
- `XLMRobertaEmbeddings.forward(inputs_embeds=...)` still adds token-type and position embeddings and applies LayerNorm and dropout. `roberta(inputs_embeds=...)` is therefore NOT a bypass; confirmed. Bypass = call `roberta.encoder` directly with a mask from `create_bidirectional_mask`.
- `XLMRobertaClassificationHead.forward` reads `features[:, 0, :]` only; confirmed.
- My extra check (real val sentences, per_dim noise, accuracy 87.9 clean): noise only at position 0 -> 60.6%; noise at all other real positions with real `<s>` kept -> 15.2%; noise everywhere -> 20.2%. So positions other than 0 matter fully through attention, and position 0 matters partly. Position 0 is not the only thing that counts.

## Per-worker status
| Worker | Deliverable | Status | Note |
|---|---|---|---|
| A | embed_inject, tests, doc | verified | claims reproduced exactly |
| B | embed_stats, stats.json, doc | verified with one fix | duplicate forward removed; numbers reproduced |
| C | vram_probe, vram.json, doc | verified | numbers reproduced; cosmetic unused args left (`batch_of(data,..)`, `storage_table(free)`, `noise_behaviour(ident_scale)`) |

## Edits made (only in text/)
1. `text/embed_inject.py`: `forward_from_embeddings(model, embeds, attention_mask=None)` (None = attend everywhere; verified identical to an all-ones mask) and docstring says the head reads position 0.
2. `text/embed_stats.py`: deleted its own `forward_from_embeds` and the `create_bidirectional_mask` import; now `from embed_inject import forward_from_embeddings as forward_from_embeds`. The two were numerically identical (max diff 0.0), so `stats.json` is unchanged and still valid. Step 5 imports `embed_inject.forward_from_embeddings` only.
No other code changed; no recommendation altered silently. Worker docs were not edited; where they differ from the decisions below, this file takes precedence.

## Decided design spec for step 5
- **Length L = 64 tensor**, not 32. 32 truncates 42% of real training sentences (tokenised mean 32, p95 49, max 74); 64 truncates 0.6%. Real inputs are padded to L=64 with the attention mask; VRAM cost is negligible (B16 L64 well under the L164 figure).
- **Noise mask = length-matched.** Each noise sample gets a length drawn with replacement from the empirical real training lengths (capped at 64) and a mask of that length; positions beyond it are masked. Reason: fully unmasked length-64 noise next to mostly-short real inputs gives the network a trivial length shortcut, so impair could teach "long unmasked input -> forget class" without touching real forget sentences (shallow forgetting). Real and noise share one padded L within a batch; real padding is masked, so no mismatch problem. Ablation: full-length noise (mask all ones).
- **Noise mode: per_dim at real scale** (mean and std per dimension from real embedding outputs, computed on the training set; B's `make_noise(mode="per_dim")`). Paper-faithful ablation: `N(0,1)` (unit scale, 4.4x too large), plus `match_norm` as a scalar-only intermediate. Keep all three selectable by one argument and report the N(0,1) row next to the default, so the deviation from DNF's definition is measured, not asserted. Tune noise scale (x0.5, 1, 2) on validation.
- **Position 0: fixed real `<s>` embedding** (`embed(model, ids)[:, 0]`, identical for all samples), noise at positions 1..len-1. Reason: the head reads only position 0 and its input is always `<s>` in real data; noise stands for content. With this, unlabelled noise gives spread, moderately confident predictions (my check: conf 0.46-0.51, classes spread) instead of collapsing onto sci/tech (conf 0.29, 97% sci/tech with noise at position 0). Ablation: noise at position 0 too (paper-literal "noise everywhere"). Non-position-0 noise is essential, not decorative: it drives the output through attention.
- **Noise-set and batch sizes:** impair batch 16 (mixed real+noise, VRAM +0.014 GB so no constraint), noise count a tunable, grid {1x, 2x} retain_sub with retain_sub about 100 per retained class capped at the train set; DNF starts at 2x, USI(F)R learning starts at n<=256 (2.6 GB at L64, largest batch far above need). Noise labelled with forget class. Dropout: real embeddings go through embedding dropout in train mode, noise does not (minor, disclosed).
- **Optimiser tunable: AdamW (default, lr about 1e-5 to 2e-5 first pass) and SGD-momentum 0.9**; lr, epochs and noise count chosen on validation with the rule stated before running. No vision hyperparameter carries over. fp32 by default; bf16 only if needed (it works and saves memory).
- **Controls, mandatory:** (1) noise_n=0 control (below), (2) FineTune on retain, (3) retrain-from-scratch from pretrained XLM-R on retain classes only (state its budget), (4) noise labelled with a random non-forget class (does the forget label matter), (5) ablations above. Retain data: unlearn on `retain_sub`, evaluate on the test split.
- **Metrics:** Acc_f, Acc_r (over retained classes), W_D, R_t (epochs of forget-class fine-tuning to reach 50% of base Acc_f), time, plus the probe from step 3. All 7 classes as forget sets, class as the independent unit, mean and SE over classes, several base seeds (3 exist).
- **noise_n=0 control in text:** meaningful and more so than in vision, because retain-only fine-tuning of a fine-tuned transformer with about 100 samples per class may drift. Run it as the identical pipeline (same optimiser, lr, epochs, batches, repair) with zero noise items in impair; the impair epoch is then plain retain fine-tuning. A positive Acc_f counts only if it beats this control (CLAUDE.md go/no-go rule).
- Base checkpoints: use `outputs/final_run/text/baseline_eng_Latn_seed{0,1,2}.pt` (post loader-fix), not the older `outputs/baseline_*.pt`.

## Unresolved doubts
- Everything is measured on one checkpoint; statistics (mean/std per dim) are from 701 training sentences and ignore cross-dimension correlation (real tokens have cosine about 0.30 to the mean vector; independent per-dim noise loses that).
- Length-matching and `<s>`-at-position-0 are my design choices, argued from the check above, not from any forgetting result; both are ablated in step 5.
- The relation between noise on the embedding output and DNF's input-noise definition is a design analogy; a null result at step 5 could stem from that choice, not only from the method.
- The VRAM numbers assume the RTX 3080 idle and B16; no long-run soak.
