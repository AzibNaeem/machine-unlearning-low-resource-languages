# Text baseline: XLM-R on SIB-200 (eng_Latn)

**Setup:** XLM-RoBERTa-base (local ModelScope copy), SIB-200 English (7 classes; 204 test samples), 10 epochs, lr 2e-5, batch 16, seeds 0/1/2. Best epoch chosen by validation accuracy; test accuracy reported at that epoch. Fresh run, `outputs/final_run/text/`.

## Per-seed results

| Seed | Best epoch | Val acc (%) | Test acc (%) |
|------|-----------:|------------:|-------------:|
| 0 | 3 | 87.88 | 91.18 |
| 1 | 5 | 86.87 | 89.22 |
| 2 | 9 | 87.88 | 91.67 |
| **Mean ± std (ddof=1)** | | **87.54 ± 0.58** | **90.69 ± 1.30** |

## Comparison

| Source | Test acc (%) |
|--------|-------------:|
| This run (mean of 3) | 90.69 |
| Audited fixed-loader run (PROGRESS.md: 91.18 / 89.22 / 91.67) | 90.69 |
| Authors' reported target | 92.5 |

The fresh run reproduces the audited numbers exactly, per seed (val and test), so the pipeline is deterministic given the seed. The mean is 1.8 points below the 92.5 target.

## Cost (per run)

Peak VRAM 5.71 GB; wall time 68 s / 68 s / 70 s (RTX 3080 10 GB, GPU otherwise idle).

## Verdict

The baseline is healthy and reproducible; it is about 1.8 points under the authors' 92.5. That gap is within noise: the test set has only 204 samples (binomial standard error 2.0 points at ~90% accuracy, and one sample is 0.49 points), and seed-to-seed std is 1.3, so the gap is not evidence of a real deficit. Caveats: HuggingFace returned HTTP 429 for this machine, so data and weights are local copies: SIB-200 TSVs from the original repo (checked equal to the HF rows in the earlier audit) and the ModelScope XLM-R weights (not hash-checked against the HF release). The validation set is 99 sentences, so epoch selection is noisy; single language (English) only.

Run (repo root, for S in 0 1 2): `.venv\Scripts\python text\train_baseline.py --lang eng_Latn --seed S --epochs 10 --lr 2e-5 --batch_size 16 --out_dir outputs/final_run/text`
