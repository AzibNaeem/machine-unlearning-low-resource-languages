> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# UNSIR (single-class, CIFAR-10 ResNet-18): final run

**Config.** Base `outputs/cifar10_resnet18.pt` as it was during this run (test 76.40, single base seed 100, overfit; see base note). Noise learned with loss -CE + 0.1*||N||^2, Adam lr 0.1, 5x8 steps; 20 noise batches (250 each); 1 impair + 1 repair epoch, Adam (0.02 / 0.01), 1000 retain samples per class. 10 forget classes x seeds 0,1,2 = 30 runs, test split. Source: `outputs/final_run/usifr/usifr_main.json`.

| Method | Acc_f | Acc_r | W_D | time (s) |
|---|---|---|---|---|
| UNSIR (this run) | 0.00 +- 0.00 | 69.14 +- 1.63 (class SE 0.47) | 77.9 +- 0.6 | 1.22 +- 0.12 |
| Base retain (9-class) | - | 76.40 | - | - |
| Paper target | 0.0 | 71.06 | - | - |
| Earlier run, old base 75.38 | 0.00 | 68.19 +- 1.39 | 92.5 | 4.4 (shared GPU) |

(mean +- sample std ddof=1 over 30 runs; SE over the 10 class means.)

**Verdict: partially reproduced.** Acc_f = 0 reproduced; retain 69.14 vs 71.06 target (-1.9) not reproduced. It differs from the earlier 68.19 by +0.95, but the base checkpoint also differs (75.38 vs 76.40), so the shift cannot be assigned to run noise.

**Caveats.**
- Not notebook-faithful (notebook: all noise labelled class 0, batch 256, first-1000 retain, 2 forget classes, 8 retain classes). The notebook run (classes 0,2) is a different experiment (`notebook_unsir.md`), so the gap to 71.06 is not a like-for-like failure.
- Adam lr 0.02 on this overfit base is fine-tune drift: the no-noise Adam control (DNF script, same splits) gives Acc_f 0.00, Acc_r 69.96, W_D 72.3. UNSIR is 0.82 +- 0.23 SE below it (worse in 9/10 classes): learned noise does not help.
- Config carried over from the old-base tuning, not re-tuned. Independent unit is the class (10); one base seed. The base checkpoint was overwritten again after this run (76.80 now), so the run cannot be regenerated bit for bit.
