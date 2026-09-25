> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# FineTune baseline (CIFAR-10, ResNet18, single-class forgetting)

**Config:** Adam lr 0.01, 5 epochs on the retain subset (1000 imgs per retain class = 9000). Chosen by the rule in `vision/run_baselines.py` on the train-derived val split (class 0, seed 0, never test): among configs with val Acc_f <= 0.02, take the highest val Acc_r. Grid lr {1e-3,1e-2} x epochs {1,5}.
**Runs:** 10 forget classes x seeds 0,1,2 = 30 runs, test set, mean +- sample std (ddof=1).

| Acc_f (%) | Acc_r (%) | W_D | time (s) |
|---|---|---|---|
| 0.87 +- 0.99 | 71.39 +- 1.90 | 44.7 +- 1.2 | 1.70 +- 0.02 |

SE over the 10 class means: Acc_f 0.22, Acc_r 0.56, W_D 0.27.

**Meaning:** retain-only fine-tuning drives the forget class to about 0 (drift), at a cost of 5 retain points vs base retain 76.40 and a large weight shift (W_D 44.7 vs about 5 for DNF-SGD). It is 1.42 +- 0.33 SE above the no-noise Adam control of the DNF script (69.96).

**Why 5 epochs, not the earlier 1 epoch:** code and rule are unchanged; the validation outcome moved. On this base, lr 0.01 / 1 epoch gave val Acc_f 0.025 (just above the 0.02 cut, so excluded) and 5 epochs gave 0.000 (val Acc_r 0.804 vs 0.813). In an aborted attempt the same 1-epoch config scored 0.010 and would have been chosen. So the choice is a threshold effect on 5 of 200 validation images: legitimate (validation only) but not robust. Supervisor spot check on the current base (seed 0, 10 classes, test): 1 epoch 2.03 / 70.58 / W_D 33.7; 5 epochs 1.17 / 71.18 / 44.1. The choice moves Acc_r by about 0.6 and does not change the picture. Earlier numbers (1.4 / 69.6 / 39.1 / 1.7 s, 1 epoch, old base 75.38) are therefore not an exact replication; about 1 point of the Acc_r difference is the base change.

**Caveats:** class is the independent unit (effective n 10). Config tuned on one class and one seed. Single base seed (100). Base checkpoint overwritten (76.40 during this run, 76.80 now). Acc_r is over the 9 retain classes.

Files: `outputs/final_run/baselines/ft_ng.json`, `summary.json`.
