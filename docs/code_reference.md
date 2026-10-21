# Code Reference — "Where are you doing this?"

A lookup sheet: for every claim made in the presentation, exactly which file and function implements it. Use this if an evaluator asks to see the code behind a specific slide.

## Quick index

| Claim on slide | File | Function |
|---|---|---|
| Noise injected after the embedding layer (slide 5) | `text/embed_inject.py` | `embed()`, `forward_from_embeddings()` |
| Bypass confirmed identical to normal model | `text/embed_inject.py` | `forward_from_embeddings()` |
| English/Urdu/Swahili baselines (slide 11) | `text/train_baseline.py`, `text/data.py` | `main()`, `get_loaders()` |
| USI(F)R and DNF reimplemented, vision (slide 13) | `vision/dnf.py`, `vision/usifr.py` | `dnf_unlearn()`, `unsir_unlearn()`, `usifr_unlearn()` |
| CIFAR-10 validation table | `vision/run_dnf.py`, `vision/run_usifr.py`, `vision/common.py` | `class_split()`, `evaluate()` |
| FineTune / NegGrad / Retrain baselines | `vision/baselines.py` | `finetune()`, `neggrad()`, `retrain()` |
| The go/no-go text experiment (slide 14) | `text/unlearn.py` | `main()`, `dnf_text()` |
| The zero-noise control | `text/unlearn.py` | `main()` — the `mult=0` case |
| Noise built from real embedding statistics | `text/embed_stats.py` | `compute_stats()`, `make_noise()` |
| The "+1.9 ± 2.1" style numbers | `text/analyze_step5.py` | the paired-difference loop |
| Forget Accuracy / Retain Accuracy | `text/unlearn.py`, `vision/common.py` | `score()`, `evaluate()` |
| Weight Distance (W_D) | `text/unlearn.py`, `vision/common.py` | `weight_distance()` |
| Time to Unlearn | `text/unlearn.py`, `vision/usifr.py` | `dnf_text()`, `_unlearn()` (both time the run) |
| Data loading and tokenising (slide 12) | `text/data.py` | `load_sib200()`, `get_loaders()` |
| UNSIR reference check (slide 11, step 2) | `Machine Unlearning.ipynb` | original notebook, run unmodified |

---

## Slide 5 — Noise injection after the embedding layer

**Claim:** we bypass the embedding lookup and feed a noise tensor straight into the transformer encoder, and verified it gives identical output to the normal model.

- `text/embed_inject.py`, function `embed(model, input_ids)` — runs the model's real embedding layer on real tokens and returns the continuous 768-number vectors.
- `text/embed_inject.py`, function `forward_from_embeddings(model, embeds, attention_mask)` — the actual bypass. It builds the attention mask, then calls `model.roberta.encoder` directly instead of going through the embedding lookup, then the classifier head.
- The "identical output" check lives in the Step-4 supervisor audit (`docs/results/step4_supervisor.md`): bypass logits vs normal logits, maximum difference 0.0 across the test set.

## Slide 11 — Baseline classifiers in three languages

**Claim:** English 90.7%, Urdu 84.6%, Swahili 77.1%, three seeds each.

- `text/data.py`, function `get_loaders(lang, batch_size)` — loads the SIB-200 TSV for the given language, tokenises with the XLM-R tokeniser, returns train/validation/test loaders.
- `text/train_baseline.py`, function `main()` — the training loop: AdamW optimiser, picks the best epoch by validation accuracy, saves the checkpoint, prints the final test accuracy. Run three times per language with `--seed 0/1/2`.

## Slide 13 — CIFAR-10 validation of USI(F)R and DNF

**Claim:** reimplemented from the paper's algorithm, Forget Accuracy reaches 0, Retain Accuracy partially matches published numbers.

- `vision/dnf.py`, function `dnf_unlearn()` — draws plain random noise, labels it with the forget class, impairs one epoch, repairs one epoch. No noise-learning step, matching DNF's definition.
- `vision/usifr.py`, function `learn_noise()` — the noise-learning phase (Algorithm 1, Phase 1): optimises a noise batch against the frozen model.
- `vision/usifr.py`, function `unsir_unlearn()` — fixes 1 repair epoch (this is the original UNSIR method).
- `vision/usifr.py`, function `usifr_unlearn()` — exposes `repair_epochs` and `alpha`, matching "USI(F)R = UNSIR but repair runs 1–3 epochs" from our plan.
- `vision/common.py`, function `class_split()` — builds the forget/retain split and the held-out validation sets used to tune hyperparameters honestly.
- `vision/run_dnf.py` and `vision/run_usifr.py` — the scripts actually executed to produce the numbers on this slide, looping over all 10 classes and 3 seeds.

## Slide 13 (supporting) — FineTune, NegGrad, Retrain

- `vision/baselines.py`, function `finetune()` — trains on retain data only.
- `vision/baselines.py`, function `neggrad()` — trains with the loss sign flipped on forget-class data (gradient ascent).
- `vision/baselines.py`, function `retrain()` — trains a brand-new model from scratch on retain data only; this is the upper-bound comparison.

## Slide 14 — The main result (the go/no-go experiment)

**Claim:** across English, Urdu and Swahili, the noise method does not forget more than a no-noise control; this is the central finding of the project.

- `text/unlearn.py`, function `main()` — the experiment runner. For every language, forget class, random seed, and noise setting, it builds the retain data, builds the noise, runs the unlearning, and records every metric to `outputs/step5/<tag>.jsonl`. This single function produced every number on this slide.
- `text/unlearn.py`, function `dnf_text()` — runs one unlearning attempt: impair epoch(s), then repair epoch(s), and times the whole thing.
- `text/unlearn.py`, function `impair_epoch()` — mixes real sentences (through the real embedding layer) and noise tensors (through the bypass) in the same training batch.
- `text/unlearn.py`, function `repair_epoch()` — ordinary fine-tuning on the retain data only, using the model's normal forward pass.
- **The control:** inside `main()`, the noise multiplier list includes `0`. When the multiplier is 0, no noise tensors are built and the run is the identical pipeline with zero noise — this is the control row that every noise result is compared against.
- `text/embed_stats.py`, function `compute_stats()` — measures the real embedding mean and standard deviation per dimension, from the model's own retain-class training data (not invented numbers).
- `text/embed_stats.py`, function `make_noise()` — generates the noise tensor using those statistics (mode `per_dim` is the default we used).

## The "+1.9 ± 2.1" numbers (standard error)

- `text/analyze_step5.py` — reads the JSONL results file, groups runs by setting, and for every noise setting computes `(control Forget Accuracy − noise Forget Accuracy)` for every class-and-seed pair, then reports the mean and the standard error of that list. This is exactly the calculation behind every "±" figure quoted in the presentation and in the Q&A notes.

## The five metrics — where each one is computed

- **Forget Accuracy / Retain Accuracy:** `text/unlearn.py` function `score()` (text); `vision/common.py` function `evaluate()` (vision). Both split predictions into "forget class" vs "everything else" and report accuracy on each.
- **Weight Distance (W_D):** `text/unlearn.py` function `weight_distance()` (text); `vision/common.py` function `weight_distance()` (vision). Both compute the L2 distance between the unlearned model's parameters and the original model's parameters.
- **Time to Unlearn:** timed inside `dnf_text()` (text) and `_unlearn()` in `vision/usifr.py` (vision), using `time.time()` around the impair-and-repair steps.
- **Relearn Time:** implemented for vision only, in `vision/rt_probe.py`. Not yet ported to text — stated openly as a "Next" item on slide 7.

## Slide 12 — Data loading and preprocessing

- `text/data.py`, function `load_sib200(lang)` — reads the raw TSV file for a language (`train.tsv`, `dev.tsv`, `test.tsv`).
- `text/data.py`, function `get_loaders(lang, batch_size)` — tokenises with the XLM-R tokeniser (max length 164), builds the 7-way label mapping, returns PyTorch DataLoaders for train, validation and test.

## Slide 11, step 2 — the UNSIR reference check

- `Machine Unlearning.ipynb` — the original UNSIR authors' notebook, run unmodified on our machine as a sanity check that our environment and reference pipeline behave correctly (70.95% retain / 0.0% forget, classes 0 and 2), before trusting our own reimplementations above.

---

## If asked something not on this list
Say plainly: "That part is still in progress — it is on our 'Next' list on slide 7," and name the item (relearn time and probe on text, native-corpus runs, the remaining baselines on text). That is a safe, honest answer and matches what the slides already say.
