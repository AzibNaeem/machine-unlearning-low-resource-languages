> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# DNF (Step 3, CIFAR-10 / ResNet-18): final sequential re-run

Script `vision/run_dnf.py`, test split, 10 forget classes x seeds 0,1,2 = 30 runs per row, one job at a time on an idle RTX 3080. Raw: `outputs/final_run/dnf/`. Mean +- sample std (ddof=1) over the 30 runs; SE = std of the 10 class means / sqrt(10). Acc_r is over the 9 retain classes; the base model's own accuracy on those is 76.40 (mean over the 10 splits; base test 76.40).

| Config | Acc_f | Acc_r (class SE) | W_D | Time (s) |
|---|---|---|---|---|
| SGD, noise 18000 (chosen post hoc) | 0.44 +- 1.13 | 76.37 +- 1.44 (0.47) | 5.07 +- 0.30 | 1.34 |
| Adam, noise 9000 | 0.00 +- 0.00 | 68.29 +- 2.30 (0.70) | 81.3 +- 0.8 | 1.09 |
| Adam, noise 1000 | 0.00 +- 0.02 | 69.52 +- 1.45 (0.41) | 74.5 +- 0.6 | 0.76 |
| Control: SGD, noise 0 | 77.83 +- 7.23 | 76.13 +- 0.80 (0.26) | 0.35 +- 0.12 | 0.68 |
| Control: Adam, noise 0 | 0.00 +- 0.00 | 69.96 +- 1.75 (0.48) | 72.3 +- 0.5 | 0.73 |

Paper target: Acc_f 0.0, Acc_r 74.45.

## Verdict: partially reproduced (mechanism yes, paper config no)

- SGD/18000: Acc_f 0.44 (worst class mean 2.9, class 8), retain equal to base (76.37 vs 76.40), above the paper's 74.45 (different base, not comparable). The forgetting is noise-driven: SGD without noise leaves Acc_f 77.8 (class means 65 to 87). Paired by class, noise adds +0.24 +- 0.22 SE to Acc_r (n.s.).
- Adam rows: Acc_f 0 but 6 to 8 retain points lost and W_D 14 to 16x larger. The no-noise Adam control forgets fully too (0.00 / 69.96 / 72.3), so this is fine-tune drift, not a noise effect. Noise does not help there: Adam-9000 is 1.67 +- 0.33 SE below the control (worse in 9/10 classes), Adam-1000 0.44 +- 0.20 below.
- DNF is not faster than UNSIR here (1.34 s vs 1.22 s); the paper's speed-up is not reproduced.

## Caveats

- Optimiser and noise count are unspecified in the paper. SGD/18000 was picked post hoc on train-derived validation splits (4 of 10 classes; the base memorised them, so Acc_r ranking = least weight movement) after the Adam variants missed the retain target, and it was tuned on the OLD base (75.38), not re-tuned on the current one. Not a reproduction with paper hyperparameters.
- SGD forgetting is shallow: a feature probe (earlier supervisor, old base) recovered about 29% forget accuracy vs a 5% retrain floor. Cited, not re-measured. R_t not measured.
- Class is the independent unit (effective n 10); single base seed (100).
- Base checkpoint changed: `outputs/cifar10_resnet18.pt` was overwritten at 2026-09-25 05:31 (test 75.38 -> 76.40) and again at 12:18 after this run (now 76.80). All rows above used the 76.40 version (`base_Acc_r` in the JSON). Earlier numbers (SGD 0.11 / 75.57 / 5.28; Adam-9000 0.00 / 67.07 / 95.3; Adam-1000 0.00 / 68.99 / 87.6; Adam noise-0 0.00 / 68.85 / 84.8) were on the 75.38 base: same pattern, Acc_r 1 to 1.3 higher, Adam W_D 10 to 14 lower.
