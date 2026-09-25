> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# NegGrad baseline (CIFAR-10, ResNet18, single-class forgetting)

**Config:** gradient ascent (-CE) on the full forget-class training data, Adam lr 1e-3, 3 epochs. Re-tuned on val (class 0, seed 0; rule: val Acc_f <= 0.02, then highest val Acc_r; grid lr {1e-5,1e-4,1e-3} x epochs {1,3}); same choice as before.
**Runs:** 10 classes x seeds 0,1,2 = 30 runs, test set, mean +- sample std (ddof=1).

| Acc_f (%) | Acc_r (%) | W_D | time (s) |
|---|---|---|---|
| 1.50 +- 1.64 | 62.75 +- 3.68 | 7.11 +- 0.12 | 0.55 +- 0.02 |

SE over the 10 class means: Acc_f 0.53, Acc_r 1.18, W_D 0.04.

**Meaning:** forgetting works (1.5%) but retain accuracy falls 13.7 points below base retain (76.40), with the highest variance of all methods. It moves the weights least of the forgetting methods (W_D 7.1) and is fastest, so the damage is concentrated rather than diffuse.

**Reference:** earlier 0.6 / 61.3 / 8.3 / 2.5 s were on the old base (75.38; times on a shared GPU). Acc_r within about 1.5 points, Acc_f slightly higher: qualitatively replicated, not exactly.

**Caveats:** class is the independent unit (effective n 10). Config tuned on one class and one seed; only the 3-epoch setting passed the val Acc_f rule, at the cost of retain. Single base seed; base checkpoint overwritten (76.40 during this run, 76.80 now). Acc_r is over the 9 retain classes.

Files: `outputs/final_run/baselines/ft_ng.json`, `summary.json`.
