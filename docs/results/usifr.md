> Current-base table (76.80, all methods, controls, paired deltas, probe/R_t): see [consolidated_table.md](consolidated_table.md) and [relearn_probe.md](relearn_probe.md). Numbers below are the earlier run on the 76.40 base.

# USI(F)R (single-class, CIFAR-10 ResNet-18): final run

**Config.** Same base (76.40 version), split and protocol as `unsir.md`. Algorithm 1 as literally read: noise loss CE(f(N), y_f) - alpha*mean||softmax(f(N)) - onehot||; Adam, lr 0.02/0.01. Chosen config alpha 10, E_r 2, 5 noise batches (1250 noise samples, fewer than the 9000 retain samples) was selected on the old base's validation split and not re-tuned. 10 classes x 3 seeds, test split. Files: `outputs/final_run/usifr/usifr_*.json`.

| Config | Acc_f | Acc_r | W_D | time (s) | Acc_r vs UNSIR (paired by class, SE) |
|---|---|---|---|---|---|
| UNSIR (nb20) | 0.00 | 69.14 +- 1.63 | 77.9 | 1.22 | - |
| USI(F)R a10 E2 nb5 (chosen) | 0.00 | 70.39 +- 1.82 | 78.4 | 1.42 | +1.25 +- 0.26, 10/10 |
| USI(F)R a1 E1 nb5 | 0.00 | 69.94 +- 1.90 | 75.4 | 1.07 | +0.79 +- 0.19, 9/10 |
| USI(F)R a1 E3 nb5 | 0.00 | 69.97 +- 1.85 | 82.0 | 1.76 | +0.82 +- 0.21, 9/10 |
| USI(F)R a10 E2 nb36 (noise = retain, Alg. 1) | 0.00 | 70.09 +- 1.93 | 84.1 | 1.75 | +0.95 +- 0.30, 8/10 |
| No-noise Adam control (DNF script) | 0.00 | 69.96 +- 1.75 | 72.3 | 0.73 | - |
| Paper target | 0.0 | 74.97 | - | - | (UNSIR 71.06) |
| Earlier run, old base | 0.00 | 69.34 +- 1.93 | 92.2 | 5.2 | (UNSIR 68.19) |

(30 runs each; mean +- sample std ddof=1 over runs.)

**Verdict: not reproduced on retain, Acc_f = 0 met.** 70.39 vs 74.97 target (-4.6). The ordering USI(F)R > UNSIR holds by 1.25 points; the paper's 3.9-point gap is not reproduced. The paper ordering USI(F)R > DNF > UNSIR is not reproduced either: DNF-SGD (76.37, post hoc tuned) is above both.

**Caveats.**
- The gain over UNSIR is not attributable to the algorithm. Against the no-noise Adam control the chosen USI(F)R is only +0.43 +- 0.17 SE (8/10 classes) and UNSIR is 0.82 below; all rows sit within about 1 point of the drift baseline (69.1 to 70.4) while losing 6 to 7 points vs base retain 76.40. alpha (1 vs 10) and E_r (1 to 3) have no clear effect.
- Chosen config uses noise < retain (deviation from Algorithm 1); the conformant nb36 row gives 70.09, no better.
- Optimiser, oversampling ratio, loss form and base model are unspecified in the paper; 74.97 stays unresolved without the authors' code.
- Class is the independent unit (10); single base seed; base checkpoint was overwritten after the run (76.80 now).
