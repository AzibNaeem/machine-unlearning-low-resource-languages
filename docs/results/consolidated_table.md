# Consolidated step-3 table (single base, equal configs), 2026-09-25

Every row was run fresh on the CURRENT base `outputs/cifar10_resnet18.pt`: test acc 76.80, mean 9-class retain acc 76.80 (per-split; base forget acc averages 76.80 too). 10 forget classes x seeds 0,1,2 = 30 runs per row, test split, 1000 retain samples/class (9000 retain). Raw JSON, logs, scripts and `summary.json` in `outputs/cleanup/table/`; numbers recomputed from raw JSON by `analyze.py`.

sha256 of the base, start: `894470b4941a3dfdb6c119796c321e84a7cf1e2e7835f287bf7a5e33288f5749`; end: identical (also identical to `cifar10_resnet18_backup_1218_test76.80.pt`). Base untouched.

Noise count: `run_usifr.py` learns ONE batch of 250 noise samples and repeats it `nb` times, so 18000 = nb 72 (250 unique samples duplicated 72x); DNF draws 18000 distinct N(0,1) samples. Not equal in diversity, only in count. Hyperparameters other than optimiser and noise count are unchanged (lr 0.02/0.01, batch 250, alpha 10, E_r 2 for USI(F)R, UNSIR E_r 1). Adam or SGD momentum 0.9 as stated. FineTune = lr 0.01, 5 epochs; NegGrad = lr 1e-3, 3 epochs (both picked by run_baselines.py's own val grid on class 0, seed 0; not re-tuned; optimiser Adam as in baselines.py).

## Table (mean +- sample std over 30 runs; SE = class-level over 10 class means)

| Row | Acc_f | Acc_r | Acc_r class SE | W_D | time s | d Acc_r vs own-opt no-noise control (SE, classes better) |
|---|---|---|---|---|---|---|
| DNF-SGD noise 0 (control) | 78.86 +- 8.24 | 76.42 +- 0.91 | 0.30 | 0.30 | 0.7 | - |
| **DNF-SGD noise 18000** | **0.25 +- 0.52** | 76.46 +- 1.40 | 0.46 | 5.2 | 1.5 | +0.04 (0.21), 3/10 |
| UNSIR-SGD nb72 (18000) | 15.41 +- 11.35 | 77.08 +- 1.51 | 0.49 | 4.4 | 1.7 | +0.66 (0.23), 8/10 |
| USI(F)R-SGD nb72 (18000) | 3.74 +- 4.61 | 77.48 +- 1.54 | 0.50 | 4.0 | 2.0 | +1.06 (0.24), 10/10 |
| USI(F)R-SGD nb0 (E_r=2 control) | 78.54 +- 8.37 | 76.52 +- 0.95 | 0.31 | 0.3 | 1.3 | (vs it: USI(F)R-SGD nb72 +0.96 (0.22), 10/10) |
| DNF-Adam noise 0 (control) | 0.00 +- 0.00 | 69.55 +- 1.70 | 0.47 | 88.9 | 0.7 | - |
| DNF-Adam noise 18000 | 0.00 +- 0.00 | 66.95 +- 1.93 | 0.59 | 105.2 | 1.4 | -2.60 (0.27), 0/10 |
| UNSIR-Adam nb72 (18000) | 0.00 +- 0.00 | 64.89 +- 1.70 | 0.47 | 103.9 | 1.8 | -4.66 (0.20), 0/10 |
| USI(F)R-Adam nb72 (18000) | 0.00 +- 0.00 | 69.07 +- 1.70 | 0.49 | 107.0 | 2.1 | -0.48 (0.13), 2/10 |
| UNSIR-Adam nb20 (5000, earlier config) | 0.00 +- 0.00 | 68.10 +- 2.11 | 0.60 | 96.9 | 1.2 | -1.45 (0.29), 0/10 |
| USI(F)R-Adam nb5 (1250, earlier config) | 0.00 +- 0.00 | 69.48 +- 2.08 | 0.60 | 96.7 | 1.4 | -0.06 (0.29), 5/10 |
| USI(F)R-Adam nb0 (E_r=2 control) | 0.00 +- 0.00 | 69.00 +- 1.77 | 0.46 | 93.8 | 1.4 | (nb72 +0.07 (0.18), 6/10; nb5 +0.49 (0.32), 8/10) |
| FineTune (lr 0.01, 5 ep) | 0.09 +- 0.16 | 70.03 +- 1.76 | 0.51 | 54.7 | 1.7 | - |
| NegGrad (lr 1e-3, 3 ep) | 0.51 +- 0.73 | 61.82 +- 3.84 | 1.20 | 8.5 | 0.5 | - |

Base: retain 76.80. No Retrain row here: Retrain 40 ep is in `relearn_probe.md` (classes 0-4, 1 seed: 0.0 / 74.8, i.e. 2.3 below base retain 77.1 on those splits) and `retrain.md` (3 classes); it is a reference, not an upper bound. Times are indicative (same GPU, but not concurrent load-controlled, and USI(F)R time includes noise learning). Paired Acc_f differences vs SGD control: DNF-SGD -78.6 (SE 2.7, 10/10 classes), UNSIR-SGD -63.4 (3.4, 10/10), USI(F)R-SGD -75.1 (2.8, 10/10).

## Honest reading

- Own no-noise control: under Adam nothing beats it. DNF-Adam and UNSIR-Adam are clearly worse (-2.6, -4.7; 0/10 classes), USI(F)R-Adam is -0.5 vs the DNF control and +0.07 (6/10, within noise) vs its own E_r=2 control. The Adam control alone (Acc_f 0.00, Acc_r 69.55, W_D 89) forgets everything: Adam lr 0.02 retain-only fine-tuning is drift, so Adam rows say nothing about noise. A larger noise count (18000 vs 1250/5000) did not help either method.
- Genuine noise effect: only under SGD. The SGD no-noise control does not forget (Acc_f 78.9, retain 76.4, W_D 0.3), so DNF-SGD's Acc_f 0.25 with retain 76.46 is attributable to the noise (10/10 classes). USI(F)R-SGD (3.74) and UNSIR-SGD (15.4) also show a noise effect, unlike the earlier val-run expectation that SGD with learned noise does not forget: on this base it partly does, but UNSIR-SGD is incomplete (large class spread) and USI(F)R-SGD is incomplete. Order of forgetting: DNF > USI(F)R > UNSIR. The learned-noise machinery gave no advantage over plain Gaussian noise at equal count: DNF forgets more completely (Acc_f 0.25 vs 3.7 / 15.4), and the retain edge of UNSIR/USI(F)R (+0.6 to +1.1) comes with residual forget accuracy. Caveat: learned noise is only 250 unique samples repeated, so diversity is not matched. The no-noise control removes the noise samples entirely, so it shows noise samples cause the SGD forgetting, not whether noise content or the forget-class label does it (a random-label noise control is still missing).
- Retain under SGD is ~ base level (76.5-77.5 vs 76.80); the +0.7 to +1.1 over the control for UNSIR/USI(F)R-SGD is small, consistent across classes (8-10/10) but with residual forget accuracy left; do not read it as a method ranking. DNF-SGD retain equals the control (+0.04, 3/10).
- Paper targets (single base seed, class as independent unit): DNF retain 74.45 is met or exceeded by DNF-SGD (76.46), but the config (SGD, 18000) was chosen post hoc on the old base and its forgetting is shallow (earlier probe about 29%). USI(F)R 74.97 retain is exceeded only under SGD (77.48 with Acc_f 3.7, not fully forgotten) and missed under Adam (69.07). UNSIR 71.06 retain is exceeded under SGD only with Acc_f 15.4 (class not forgotten, so the joint target is not met) and missed under Adam (64.9-68.1). Retain targets compare against a base that differs from the paper's, so do not treat these as reproduction; the paper ordering USI(F)R > DNF > UNSIR is not reproduced on Acc_f + Acc_r together (DNF-SGD forgets best with equal retain).
- FineTune 70.03 / NegGrad 61.82 are reference points; FineTune lands where the Adam controls do.

## What changed vs the earlier mixed-base numbers

Earlier figures (old 75.38 base, then 76.40) are superseded. On 76.80: DNF-SGD 18000 now 0.25 / 76.46 (was 0.11 / 75.57, then 0.44 / 76.37): stable within noise, still noise-driven. Adam UNSIR/USI(F)R retain are now 64.9-69.5 (earlier 68.2-70.4; no-noise Adam control 69.55 vs 68.85 / 69.96 earlier). UNSIR at nb72 is worse than nb20, and equal noise count did not change the "no effect over control" verdict for Adam. New: SGD with learned noise (UNSIR/USI(F)R) does forget partially, an unexpected result to be reported as is. FineTune 70.03 (was 69.6 / 71.39), NegGrad 61.82 (was 61.3 / 62.75).

## Caveats

Single base seed (100) and one base checkpoint (76.80, GPU-nondeterministic training); class is the independent unit (10 classes), so SEs are over 10 class means, and 30 runs are not 30 independent samples. Optimiser and noise count were not tuned here; the SGD/18000 config was chosen earlier on a different base. Noise diversity differs (DNF 18000 distinct vs UNSIR/USI(F)R 250 repeated). Acc for FineTune/NegGrad converted from fractions to percent. The noise-free controls for UNSIR/USI(F)R use DNF noise0 (1 repair epoch) plus USI(F)R nb0 (E_r=2, learned noise discarded) as a matched control.
