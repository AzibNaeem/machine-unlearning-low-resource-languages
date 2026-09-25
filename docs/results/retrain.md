> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# Retrain-from-scratch reference (CIFAR-10, ResNet18)

**Config:** new ResNet18 from random init, 40 epochs on all retain-class training data (about 43.2k images), same recipe as the base model; does not use base weights.
**Runs:** only 3 forget classes (0, 3, 6) x seed 0 = 3 runs (cost), test set. Std (ddof=1) over 3 runs is indicative only.

| Acc_f (%) | Acc_r (%) | W_D | time (s) |
|---|---|---|---|
| 0.00 +- 0.00 | 75.98 +- 2.19 | 106.5 +- 1.5 | 71.4 +- 0.3 |

Per class Acc_r: class 0 74.06, class 3 78.37, class 6 75.52.

**Meaning:** exact-unlearning reference. Acc_f is 0 by construction. Acc_r 75.98 vs the base's retain accuracy on the same three splits, 76.59 (-0.6, one seed, within noise). It does not beat DNF-SGD on these classes (76.66), so it is a reference point, not a proven upper bound (3 runs, one training seed each). W_D is large because it is a different network. Cost 71 s vs about 1 to 2 s for approximate methods.

**Reference:** the earlier 20-epoch run (0.0 / 73.9, 10 classes x 3 seeds, shared GPU, 118 s) was under-trained; 40 epochs gains about 2 points, but the class subsets differ, so it is not strictly comparable.

**Caveats:** 3 classes, 1 seed; class is the independent unit. Base checkpoint overwritten (76.40 during this run, 76.80 now); retrain itself does not depend on it, only the accuracy reference does. Acc_r is over the 9 retain classes.

Files: `outputs/final_run/baselines/retrain40.json`, `summary.json`.
