# Supervisor report: fresh sequential runs (2026-09-25)

All headline numbers below were recomputed from the raw JSON in `outputs/final_run/` (mean, sample std ddof=1, class-level SE); machine-readable copy in `docs/results/summary_tables.json`.

## Overall verdict

Step 3 stays PARTIALLY reproduced. Acc_f ~ 0 is reproduced for every method; the retain target is met only by DNF-SGD (a post hoc, old-base-tuned config); UNSIR (69.14 vs 71.06) and USI(F)R (70.39 vs 74.97) miss it; the paper ordering USI(F)R > DNF > UNSIR is not reproduced. All Adam rows are fine-tune drift, not a noise effect (a no-noise Adam control forgets fully). The text baseline is healthy (90.69 vs 92.5, within test noise). Worker reports were numerically correct; the docs had wording errors, now fixed.

## Base-checkpoint disclosure (must go in the paper)

`outputs/cifar10_resnet18.pt` has had three versions: 75.38 test (09-24 21:27, original `base_train.log`), 76.40 (overwritten 09-25 05:31, used by ALL final_run vision results; confirmed by `base_Acc_r` 76.40 in every DNF row and my own re-evaluation), and 76.80 (overwritten 12:18 by me, see below). Training with seed 100 is not bit-reproducible on GPU. The mtime 05:31 does not match `COMMON_READY` (21:27), so the earlier overwrite was a `train_base_model()` call that never finished the `__main__` block, most likely a worker in the failed parallel attempt; not provable. My own accident: while checking commands I ran `vision/common.py --help`, which ignores arguments and retrains and overwrites the base. The 76.40 file is therefore lost. Backup of the current file: `outputs/cifar10_resnet18_backup_1218_test76.80.pt`. I added a guard in `vision/common.py` (refuses to overwrite unless `--retrain`). Consequences: (1) no result mixes bases within one table, but old-base (75.38) and new-base (76.40) numbers must not be pooled or called replications; (2) DNF (SGD/18000), UNSIR and USI(F)R configs were tuned on the 75.38 base and not re-tuned; only FineTune and NegGrad were re-tuned; (3) rerunning any vision script now gives 76.80-base numbers, expect Acc_r shifts of about 1 point.

## Status per experiment

| Experiment | Result (this run) | Status | One honest line |
|---|---|---|---|
| Text XLM-R SIB-200 en | test 90.69 +- 1.30 (91.18/89.22/91.67); val 87.54 | Reproduced (vs audit exact per seed; 1.8 below 92.5, inside SE 2.0) | Deterministic, matches audit exactly; only English, 204 test sentences. |
| UNSIR (single class) | 0.00 / 69.14 / W_D 77.9 | Partial | Forgets; retain 1.9 below paper; 0.82 below the no-noise control. |
| USI(F)R | 0.00 / 70.39 / W_D 78.4 | Not reproduced (retain -4.6) | +1.25 over UNSIR but only +0.43 over the no-noise control; noise < retain deviation. |
| DNF SGD 18000 | 0.44 / 76.37 / W_D 5.1 | Partial | Meets 74.45 only with a post hoc SGD choice; noise-driven (no-noise SGD Acc_f 77.8) but shallow. |
| DNF Adam 9000 / 1000 | 0.00 / 68.3, 69.5 | Not a method effect | Drift; control gets 69.96, Adam-9000 is 1.67 +- 0.33 worse. |
| FineTune | 0.87 / 71.39 / W_D 44.7 | Reference reproduced | Same picture as earlier (1.4 / 69.6); 5-epoch choice is a val-threshold effect. |
| NegGrad | 1.50 / 62.75 / W_D 7.1 | Reference reproduced | Forgets but costs 13.7 retain points. |
| Retrain 40 ep | 0.00 / 75.98 (3 classes, 1 seed) | Reference only | Still 0.6 below base retain and not above DNF-SGD: not a proven upper bound. |
| Notebook UNSIR (0,2) | 0.00 / 72.68; base 75.15 / 76.80 | Pipeline reproduced | Single seed, label-0 quirk, no no-noise control, so class-2 forgetting cause unknown. |

## Edits I made
- `dnf.md`: replaced "indistinguishable from control" for Adam by paired numbers (Adam-9000 worse by 1.67 +- 0.33); added "not re-tuned on new base", DNF not faster than UNSIR (1.34 vs 1.22 s), class SE column, stated old-vs-new base comparison.
- `unsir.md`: base test 75.38 -> 76.40 (was wrong); removed "run-to-run nondeterminism" as explanation of the +0.95 (base changed); replaced the stale old-base control (68.85 / 85) by the current one (69.96 / 72.3); paired vs control.
- `usifr.md`: same control fix; DNF-SGD number 75.57 -> 76.37; USI(F)R vs control +0.43 +- 0.17 (not "equal"); config tuned on old base.
- `finetune.md`: "std over class means" values relabelled as SE (0.22/0.56/0.27); explained the 1 -> 5 epoch change (same rule, val Acc_f 0.025 vs 0.000 at a 0.02 cut; aborted attempt had 0.010) plus my spot check (1 ep 2.03 / 70.58; 5 ep 1.17 / 71.18).
- `neggrad.md`: SE relabel, 13.6 -> 13.7 points, tuning note. `retrain.md`: reference is base retain on classes 0,3,6 (76.59, diff -0.6, not -0.4 vs 76.40); "upper bound" wording removed.
- `notebook_unsir.md`: base row filled from my evaluation of the saved checkpoint (forget 75.15, retain 76.80, overall 76.47); removed unsupported "repair alone leaves no class-0/2 predictions" claim; flagged hardcoded data path.
- `text_baseline_xlmr.md`: "HF blocked" -> HTTP 429 with local copies; binomial SE; command. `vision/common.py`: overwrite guard.

## Unresolved doubts
- Who or what triggered the 05:31 overwrite (hypothesis only). The 76.40 base is unrecoverable.
- Single base seed and one training run: base variance unmeasured (75.38 / 76.40 / 76.80 across identical-seed retrains is itself a 1.4-point spread, similar to the differences between methods' Acc_r).
- SGD-forgetting depth (probe 29%) and R_t are old-base / not measured here. The USI(F)R vs UNSIR ordering and the paper's 74.97 remain unexplained without the authors' code.
- Retrain has 3 runs; notebook has 1 run; FineTune/NegGrad tuned on one class and seed.

## Correct run commands (repo root; PowerShell: `$env:PYTHONPATH='.'`; bash: `PYTHONPATH=.`)
Do NOT run `vision/common.py` (now guarded; needs `--retrain` to replace the base).
- Text: `.venv\Scripts\python text\train_baseline.py --lang eng_Latn --seed S --epochs 10 --lr 2e-5 --batch_size 16 --out_dir outputs/final_run/text` (S = 0,1,2; works without PYTHONPATH).
- DNF (works with or without PYTHONPATH; I ran SGD class 0 both ways): `.venv\Scripts\python vision\run_dnf.py --opt sgd --noise_n 18000 --out outputs/final_run/dnf/dnf_sgd18000.json`; other rows: `--opt adam --noise_n 9000|1000|0`, `--opt sgd --noise_n 0`.
- UNSIR + USI(F)R (works without PYTHONPATH): `.venv\Scripts\python vision\run_usifr.py --out outputs/final_run/usifr/usifr_main.json`; variants `--methods usifr --alpha 1 --repair_epochs 1|3`, `--methods usifr --noise_batches 36`.
- FineTune/NegGrad: `PYTHONPATH=. .venv\Scripts\python vision\run_baselines.py --methods finetune neggrad --out outputs/final_run/baselines/ft_ng.json` (fails without PYTHONPATH: ModuleNotFoundError, verified); Retrain: `... --methods retrain --classes 0 3 6 --seeds 0 --retrain_epochs 40 --out outputs/final_run/baselines/retrain40.json`.
- Notebook: `.venv\Scripts\python outputs\final_run\notebook\notebook_pipeline.py` needs the hardcoded `DATA_DIR` (scratchpad ImageFolder PNGs, not in repo) edited first.
