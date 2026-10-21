# Progress Log

Running record of what has been done, so any team member can pick up where we left off.
Newest entries go at the bottom. Project goals and rules live in [CLAUDE.md](CLAUDE.md).

## 2026-09-24: Baseline notebook running and fixed

### Starting point
The repo is a clone of the original UNSIR code (`vikram2000b/Fast-Machine-Unlearning`): one notebook, `Machine Unlearning.ipynb`. It trains a ResNet18 on CIFAR-10, learns noise for the classes to forget (0 and 2), fine-tunes with that noise (impair), then fine-tunes on retained data (repair).

### Getting it to run
- No PyTorch was installed. The README pins `torch==1.5.0`, which cannot be installed on current Python.
- Built a Python 3.12 venv with `torch 2.6.0+cu124` and `torchvision 0.21.0`. The GPU (RTX 3080) is detected.
- Ran the notebook code as a script with training cut to 2 epochs (smoke test only). It completed end to end.

### Smoke test result (2-epoch base model, not the paper's setup)
| Stage | Forget acc | Retain acc |
|---|---|---|
| Base model | n/a | 54.8% |
| After impair | 0.2% | 44.9% |
| After repair | 0.0% | 57.1% |

The unlearning pipeline works. These are not the numbers to compare against the paper; see the validation targets in CLAUDE.md.

### Fixes made to the notebook
| Problem | Fix |
|---|---|
| `distance()` used `p.data0`, which does not exist (crashes if called) | Now uses `p.data`. This function computes weight distance (W_D). |
| Hardcoded `cuda:0` and `.cuda()` calls; fails without a GPU | `device` falls back to CPU; all `.cuda()` became `.to(device)`; `torch.load` uses `map_location` |
| `num_workers=3` breaks DataLoaders on Windows | `num_workers = 0 if os.name == 'nt' else 3` (defined in cell 1) |
| `pin_memory=True` warns without CUDA | Only enabled when CUDA is available |
| `torch.tensor(labels)` copy warning | Labels are already tensors, so removed the wrapper |
| Deprecated `verbose=True` in `ReduceLROnPlateau` and tar extraction warning | Removed `verbose`; extraction uses `filter='data'` with a fallback for old Python |
| README pinned ancient versions | README now says modern torch (>= 2.0, tested 2.6) works, and that Windows/CPU are supported |

The fixes were made by three parallel workers (one per concern), then merged and verified by a supervisor. After merging, a full CPU run produced no errors and no warnings.

### How to run it
```
pip install torch torchvision
```
Open `Machine Unlearning.ipynb` and run the cells in order. The notebook sets `epochs = 40` for the base model. CIFAR-10 (~160 MB) is downloaded on first run into `./data`.

### Not done / open items
- `environment.yml` still pins the old package versions.
- The fixed notebook was verified on CPU only. A GPU run of the fixed version has not been done.
- The venv used for testing is in a temporary folder and will disappear. Each member should create their own.
- `transformers` and `datasets` are not installed yet; needed for the text work.
- Repo has no `.gitignore` yet (should exclude `data/`, `*.pt`, `*.tgz`), and nothing is committed or pushed. The upstream repo has no license, so fork it rather than uploading a fresh copy and credit the original author.

### Next step
Build step 1 from CLAUDE.md: fine-tune XLM-RoBERTa base on English SIB-200 (7 classes), target about 92% accuracy, before doing anything else.

[Corrected 2026-09-25: stale. Step 1 was done the same day (next section), step 3 is PARTIAL and step 4 plumbing is done; see "2026-09-25: Step 3 status and corrections" at the bottom.]

## 2026-09-24: Step 1: baseline (XLM-R on English SIB-200)

### Built
- `.venv` (Python 3.12) with `requirements-text.txt` (torch 2.6.0+cu124, transformers 5.17.0, datasets 5.0.1, scikit-learn, numpy). `.gitignore` added.
- `text/data.py`: loads SIB-200 for a language code, tokenizes (max_len 164, pad to max_length), returns train/validation/test DataLoaders and the label mapping.
- `text/train_baseline.py`: plain PyTorch loop, AdamW (wd 0.01), 10% linear warmup then linear decay, grad clip 1.0, fp32 (no autocast). Picks the epoch with best validation accuracy, reloads that checkpoint, reports test accuracy. Saves `outputs/baseline_{lang}_seed{seed}.pt`. Falls back to CPU.

### How to run
```
.venv\Scripts\python text\train_baseline.py --lang eng_Latn --seed 0 --epochs 10 --lr 2e-5 --batch_size 16
```

### Data (verified from the files)
Columns are `index_id, category, text`; categories are strings, mapped alphabetically to ids: entertainment 0, geography 1, health 2, politics 3, science/technology 4, sports 5, travel 6.
Split sizes train/validation/test = 701/99/204. Train distribution: ent 65, geo 58, health 77, politics 102, sci/tech 176, sports 85, travel 138 (imbalanced; sci/tech is 25%). Longest sentence is 76 tokens, so max_len 164 truncates nothing (padding is wasteful but matches the spec).

### Results (lr 2e-5, 10 epochs, batch 16, test accuracy at the epoch picked by validation)
| Seed | Best epoch | Val acc | Test acc |
|---|---|---|---|
| 0 | 3 | 89.90 | 87.75 |
| 1 | 5 | 87.88 | 91.18 |
| 2 | 4 | 88.89 | 88.73 |
| mean | | 88.89 | 89.22 +- 1.77 (sample std) |

Peak VRAM 5.71 GB (max_memory_allocated), about 70 s per 10-epoch run on the RTX 3080.

lr sweep (3 seeds each, 10 epochs, selected by mean validation accuracy): 1e-5 gave val 85.9 / test 90.2; 3e-5 gave val 87.2 / test 89.3; 2e-5 gave val 88.9 / test 89.2. 2e-5 was kept because it had the best validation; the test differences between the three are within noise.

### Open issues
- Mean test 89.2% is about 3 points below the authors' 92.5%; not fully reproduced. Validation set is only 99 sentences, so epoch selection is noisy (seed std about 1.8 points). Only lr was swept, not epochs or batch size.
- huggingface.co returned HTTP 429 for this machine's IP for over an hour, so `Davlan/sib200` and `xlm-roberta-base` could not be fetched from HF. Workaround: SIB-200 TSVs from the original repo `dadelani/sib-200` (data/annotated/eng_Latn, counts match the HF split sizes) and the `xlm-roberta-base` weights from the ModelScope mirror (`AI-ModelScope/xlm-roberta-base`), both in the gitignored `data/`. `data.py` uses the local copies if present and falls back to HF otherwise. The mirror was not hash-checked against the HF weights.
- Checkpoints of the sweep are in `outputs/sweep_lr*`; logs in `outputs/*.log`.

### Supervisor audit (step 1)
- Bug found and fixed in `text/data.py`: TSVs were read with `quoting=csv.QUOTE_NONE`, so 53 sentences containing quotes (41 train, 6 val, 6 test) kept CSV escaping (extra outer quotes, doubled `""`). Now default csv quoting; train/val/test then match the HF `Davlan/sib200` eng_Latn rows exactly (checked via datasets-server.huggingface.co: texts, labels, index_id, order). Worker numbers above were produced with the buggy loader (seed 0 reproduced exactly: 87.75).
- Corrected runs, documented default (lr 2e-5, 10 ep, bs 16), seeds 0/1/2: val 87.88/86.87/87.88, test 91.18/89.22/91.67, mean test 90.69. Best epochs 3/5/9. Outputs in `outputs/audit/`.
- Validation-only diagnostics (3 seeds, mean val / mean test): 2e-5 20ep bs16 87.9 / 89.9; 2e-5 10ep bs32 87.2 / 88.6; 3e-5 15ep bs16 89.2 / 91.2; 3e-5 15ep bs32 87.2 / 89.4; default 87.5 / 90.7. Val differences are 1-2 examples of 99, so no config is clearly better; documented default kept. Gap to 92.5 is about 1.8 points, inside test noise (1 SE about 2 points at n=204).
- HF model files still unreachable (429); `data/xlmr` weights/tokenizer not hash-checked against HF. sha256 of model.safetensors: 6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb.
- Baseline checkpoints `outputs/baseline_*.pt` were trained on the buggy loader; regenerate (or use `outputs/audit/fixed/`) before step 5.

## 2026-09-24: Step 3: DNF (Worker 1)

- `vision/dnf.py`: `dnf_unlearn(model, retain_sub, forget_class, noise_n, ...)`. No noise learning: N~Normal(0,1) shaped like the input, forget label; impair 1 epoch on retain_sub + noise, repair 1 epoch on retain_sub; batch 250, impair lr 0.02, repair lr 0.01. Returns a copy of the model with `.unlearn_time` set.
- `vision/run_dnf.py`: CLI (`--forget_class`, `--seed`, `--noise_n`, `--opt`, `--tune`); results in `outputs/dnf_results.json` (chosen config), `dnf_results_adam9000.json`, `dnf_results_adam1000.json`; tuning logs in `outputs/dnf_tune*.json`.
- Tuning (optimizer x noise count, classes 0,3,6,9, seeds 0,1) used only train-derived held-out val splits, never test. Rule: Acc_f ~0, then highest Acc_r. Chosen: SGD (momentum 0.9), 18000 noise samples (2x retain of 9000).
- Test, 10 classes x 3 seeds (base model test acc 75.38): SGD/18000 Acc_f 0.11+-0.26, Acc_r 75.57+-1.68, W_D 5.28, ~6.6 s. Adam/9000: Acc_f 0.00, Acc_r 67.07+-2.08, W_D 95.3. Adam/1000: 0.00 / 68.99+-1.53 / 87.6.
- Caveat: val splits are base-model training data (memorised), so val Acc_r is inflated; it was used only for ranking. Paper's 74.45 retain target is met by SGD config, not by Adam (notebook style).

[Corrected 2026-09-25: "met" is overstated. The SGD/18000 config is unspecified in the paper and was chosen post hoc, after Adam missed the target, and tuned on the old 75.38 base (never re-tuned). Its forgetting is noise-driven but shallow, and the 75.57 figure is not comparable across bases. The final-run DNF-SGD row is 0.44 / 76.37 / W_D 5.1 on the 76.40 base. The Adam rows are fine-tune drift (no-noise Adam control forgets fully). See the 2026-09-25 corrections section.]

## 2026-09-24: Step 3: UNSIR + USI(F)R (Worker 2)

- [Corrected 2026-09-25: "notebook-faithful" below is wrong. The notebook labels all noise as class 0 and uses a different protocol (batch 256, first-1000 retain, 2 forget classes, 8 retain classes). This is paper-style single-class UNSIR.]
- `vision/usifr.py`: `unsir_unlearn` (notebook-faithful: loss -CE + 0.1*||N||^2, Adam lr 0.1, 5x8 steps, one 250-sample noise batch repeated, 1 impair + 1 repair epoch) and `usifr_unlearn(alpha, repair_epochs)` (Algorithm 1). Noise learning runs on frozen weights (BN in train mode, as in the notebook) and is inside the timed region. `vision/run_usifr.py` is the CLI; `--eval val` uses the train-derived held-out split for tuning.
- USI(F)R noise loss (form is ambiguous in the paper): CE(f(N), y_forget) - alpha * mean_i ||softmax(f(N_i)) - onehot(y_forget)||_2, literal reading of Algorithm 1. Noise labelled with the forget class in impair. Adam for impair/repair (lr 0.02/0.01), batch 250, 1000 retain per class.
- Tuning (val split, seed 0, all 10 classes; never test): alpha {0.1,1,10} x E_r {1,2,3} x noise batches {20,5} all gave Acc_f 0.0; Acc_r 68.0-73.0, differences within noise (std ~2 across classes). alpha has no visible effect. 5 batches beat 20 in 8 of 9 pairs. Chosen by top val Acc_r: alpha 10, E_r 2, 5 noise batches (val 72.97). SGD momentum 0.9 does not forget with learned noise (val Acc_f 11-54%, W_D ~3). Logs in `outputs/tune/`.
- Test, 10 classes x 3 seeds (`outputs/usifr_results.json`): UNSIR (faithful, 20 batches) Acc_f 0.00, Acc_r 68.19+-1.39, W_D 92.5, ~4.4 s. USI(F)R (alpha 10, E_r 2) Acc_f 0.00, Acc_r 69.34+-1.93, W_D 92.2, ~5.2 s. Sensitivity (`usifr_results_er1_a1.json`, `_er3_a1.json`): E_r 1 68.93, E_r 3 68.90.
- Misses targets: UNSIR 68.19 vs 71.06 (step-2 notebook run 70.95 was classes 0,2), USI(F)R 69.34 vs 74.97. USI(F)R gain over UNSIR is ~1 point, inside noise. Adam at lr 0.02 moves weights far (W_D ~92 vs ~5 for DNF-SGD), which is superficial-forgetting-unfriendly and is the likely reason retain stays ~69. Paper's 74.97 is not reproduced.

[Corrected 2026-09-25: (1) the step-2 70.95 was classes 0 and 2 jointly, not comparable to the single-class 68.19, so the gap to 71.06 is not a like-for-like failure. (2) "USI(F)R gain ~1 point, inside noise" is superseded by the paired-by-class analysis: about +1.1 to +1.25 (final run +1.25 +- 0.26 SE, 10/10 classes), but not attributable to the algorithm (only +0.43 +- 0.17 over the no-noise Adam control). (3) The alpha inertness means the noise loss is inert, and the chosen config uses 1250 noise samples vs 9000 retain, a deviation from Algorithm 1 (the noise = retain nb36 row gives 70.09, no better). (4) Adam UNSIR/USI(F)R are fine-tune drift: the no-noise Adam control gives Acc_f 0.00, Acc_r 69.96, W_D 72.3. The 69.34 and 68.19 figures are old-base numbers.]

## 2026-09-24: Step 3: shared code + baselines (Worker 3)

- `vision/common.py`: `load_cifar10` (in-memory normalised tensors, uint8 cache `data/cifar10_tensors.pt`, no augmentation), `load_base_model`, `train_base_model`, `evaluate` (dict loss/acc), `weight_distance` (raw L2, W_D), `class_split`, `set_seed`, `device`. Additions: `make_loader`, `fit`, `new_model`; `class_split` also returns `forget_train`, `retain_train` (train-split samples excluding the val holdout).
- Base model: ResNet18, 40 epochs, Adam 0.01, clip 0.1, wd 1e-4, seed 100, no augmentation -> test acc 75.38% (train loss 0.0009, heavily overfit). `outputs/cifar10_resnet18.pt`, log `outputs/base_train.log`.
- [Corrected 2026-09-25: this checkpoint was later overwritten twice: 76.40 (used by all final-run vision results, now lost) and 76.80 (current, after an accidental retrain). See the corrections section.]
- Caveat: base model trains on all 50k train images, so `val_forget_dl`/`val_retain_dl` (200/class, carved from train) are disjoint from the unlearning data but NOT unseen by the base model; their accuracies are inflated (val_r ~0.8-1.0 vs test ~0.7). Fine for ranking configs, not for absolute numbers.
- Baselines (`vision/baselines.py`, `run_baselines.py`), single class, 10 classes x 3 seeds, 1000 retain/class. Configs chosen on val split (class 0, seed 0): FineTune Adam lr 0.01, 1 epoch on retain_sub; NegGrad Adam lr 1e-3, 3 epochs over full forget-class train data (-CE). Retrain: from scratch, 20 epochs (not 40, for cost) on all retain-class train data (43.2k). Results (`outputs/baseline_results.json`, mean +- sample std over 30 runs):

| Method | Acc_f | Acc_r | W_D | time (s) |
|---|---|---|---|---|
| FineTune | 1.4 +- 2.1 | 69.6 +- 2.0 | 39.1 | 1.7 |
| NegGrad | 0.6 +- 0.9 | 61.3 +- 4.3 | 8.3 | 2.5 |
| Retrain (20 ep) | 0.0 +- 0.0 | 73.9 +- 1.6 | 104.1 | 118 |

- Note: Acc_r is over the 9 retain classes; the base model's own retain acc on those is higher than 75.38 for some classes. Retrain time was measured while the GPU was shared with other workers.

[Corrected 2026-09-25: the Retrain row (20 epochs, 3 seeds per the table) is under-trained (base used 40 epochs) and must not be called an upper bound. The later 40-epoch run covers only 3 classes, 1 seed: 0.00 / 75.98, still ~0.6 below base retain on those classes and not above DNF-SGD. "30 runs" here means 10 classes x 3 seeds, not 30 independent samples. FineTune/NegGrad numbers above were superseded by the final run (FineTune 0.87 / 71.39, NegGrad 1.50 / 62.75, base 76.40).]

## 2026-09-25: Full re-run attempt (incomplete, GPU overloaded)

Tried to re-run every experiment fresh with five parallel workers (notebook, text baseline, DNF, UNSIR/USI(F)R, FineTune/NegGrad/Retrain). This overloaded the 10 GB RTX 3080 and the host RAM: jobs stalled or crashed and produced no usable results. Lesson: run GPU jobs one or two at a time.

Partial outputs in `outputs/final_run/` (do not treat as results):
- `usifr/`: UNSIR 30 runs finished on the overloaded GPU (Acc_f 0.00, Acc_r 69.07 +- 1.86, W_D 77.9, time inflated); USI(F)R did not run.
- `notebook/`: pipeline script written, run stalled; `docs/results/notebook_unsir.md` is a placeholder marked pending.
- `text/`, `dnf/`, `baselines/`: logs from stalled or crashed runs only.

Earlier verified results (Step 1 and Step 3 sections above, and `outputs/review/final_decision.md`) are unchanged and still stand.

[Corrected 2026-09-25: "still stand" applies only with the corrections noted under the Step 3 entries; the later sequential re-run (next section) supersedes the stalled attempt.]

### To do after restarting the PC
Run sequentially, one or two jobs at a time, with the GPU free: text baseline (3 seeds), then USI(F)R and DNF, then FineTune/NegGrad/Retrain (40 epochs), then the notebook. Then compile the PDF and per-method files in `docs/results/`.

## 2026-09-25: Full sequential re-run completed, supervised, PDF built

Re-ran everything one job at a time on the idle GPU (five workers in sequence, then a supervisor audit, then a PDF builder). Raw outputs: `outputs/final_run/{text,usifr,dnf,baselines,notebook}/`. The stalled parallel attempt is kept in `outputs/final_run_stalled_attempt/` (not results).

Deliverables:
- `docs/FYP_Results_Report.pdf` (10 pages): summary, master table + charts, one section per method, caveats, reproduce commands, next steps.
- `docs/results/*.md`: one short file per method with the key metrics; `docs/results/supervisor_report.md` (audit, disclosures, commands); `docs/results/summary_tables.json` (all headline numbers, from raw files).

Status: Step 1 reproduced within noise (test 90.69 +- 1.30 vs 92.5). Step 3 PARTIAL: Acc_f ~0 for every method; retain target met only by DNF-SGD (76.37 vs 74.45, chosen post hoc); UNSIR 69.14 vs 71.06 and USI(F)R 70.39 vs 74.97 missed; paper ordering not reproduced; Adam rows are fine-tune drift (no-noise Adam control forgets fully). Notebook pipeline works (classes 0 and 2: Acc_f 0.00, retain 72.68 after repair).

Base checkpoint warning: `outputs/cifar10_resnet18.pt` has been three different files (test 75.38 original, 76.40 used for all final vision runs and now lost, 76.80 current after an accidental retrain). Never run `vision/common.py` directly; it now needs `--retrain` to overwrite. A backup of the current file is `outputs/cifar10_resnet18_backup_1218_test76.80.pt`. Do not pool numbers across bases.

Commands: `run_baselines.py` needs `PYTHONPATH=.` (see supervisor_report.md). The notebook script hardcodes a temporary data path.

Next: step 4 (embedding-layer noise for XLM-R) can start; cleanup items from `outputs/review/final_decision.md` (relearn time R_t, feature probe, 40-epoch Retrain on more classes) gate step 5.

## 2026-09-25: Step 4: embedding-layer noise plumbing (three workers + supervisor)

Files: `text/embed_inject.py` (`embed`, `forward_from_embeddings`), `text/embed_stats.py` (`compute_stats`, `make_noise`), `text/vram_probe.py`, `text/tests/test_embed_inject.py`; outputs `outputs/step4/{A,B,C}/`; docs `docs/results/step4_{bypass,noise_stats,vram,supervisor}.md`.

Verdict: plumbing MET, VRAM MET; nothing about forgetting was tested. One checkpoint (seed 0, val 87.88), English only.
- `roberta(inputs_embeds=...)` is not a bypass in transformers 5.17 (still adds position + LayerNorm); the bypass calls `roberta.encoder` directly. Head reads position 0 only. Bypass logits equal normal logits exactly (max diff 0.0), val acc 87.88 both paths.
- Real embedding-output token norm 6.31 (N(0,1): 27.7), scalar std 0.228, mean real length 32 (42% of sentences exceed 32, 0.6% exceed 64).
- Extra VRAM for noise in impair 0.014 GB (5.714 -> 5.728 at B16 L164); noise learning n64 L64 2.6 GB. No plan change.
- Supervisor check: noise at position 0 only drops real val acc to 60.6%, noise at positions 1+ only to 15.2%: other positions matter through attention.
- Edit: duplicate forward in embed_stats removed (uses embed_inject's; identical).

Step 5 spec: L=64, length-matched noise masks (avoid length shortcut); per_dim noise at real scale as default, N(0,1) and match_norm as ablations; real `<s>` embedding at position 0 (ablation: noise there); impair batch 16, noise count 1x/2x retain_sub tunable; AdamW default, SGD-momentum alternative, tuned on validation with a pre-stated rule; mandatory controls: noise_n=0 (identical pipeline, zero noise), FineTune, retrain, random-label noise; metrics Acc_f, Acc_r, W_D, R_t, probe, time; all 7 classes as forget sets, class as unit, 3 base seeds. Details in `docs/results/step4_supervisor.md`.

## 2026-09-25: Step 3 status and corrections

Supersedes the original Step 3 worker wording above (kept for history; see the bracketed notes). Sources: `docs/results/supervisor_report.md`, `outputs/review/final_decision.md`, raw JSON in `outputs/final_run/`.

- **Status: PARTIAL.** Acc_f is about 0 for every method. The retain target is met only by DNF-SGD (76.37 vs 74.45, chosen post hoc, tuned on the old 75.38 base). UNSIR (69.14 vs 71.06) and USI(F)R (70.39 vs 74.97) miss their targets. The paper ordering USI(F)R > DNF > UNSIR is not reproduced.
- **UNSIR is not notebook-faithful.** The notebook labels all noise as class 0 and uses a different protocol (batch 256, first-1000 retain, 2 forget classes, 8 retain classes). Step 2's 70.95 was classes 0 and 2 jointly, so it is not comparable to the single-class number.
- **Adam rows are fine-tune drift** (DNF-Adam, UNSIR, USI(F)R, FineTune). The no-noise Adam control forgets fully: Acc_f 0.00, Acc_r 69.96, W_D 72.3. The USI(F)R noise loss is inert (alpha has no effect). Its chosen config uses 1250 noise samples vs 9000 retain, deviating from Algorithm 1 (the noise = retain nb36 row gives 70.09). "USI(F)R ~1 point inside noise vs UNSIR" is superseded by the paired-by-class analysis: about +1.1 to +1.25 (final run +1.25 +- 0.26 SE, 10/10 classes), but not attributable to the algorithm (+0.43 +- 0.17 over the control).
- **Counting and statistics.** "30 runs" = 10 classes x 3 seeds. The class is the independent unit (quote mean and SE over the 10 class means). Single base seed (100). Std is ddof=1 everywhere.
- **Retrain.** The Retrain row in the earlier table (20 epochs, 3 seeds) is under-trained. The 40-epoch run covers only 3 classes, 1 seed (0.00 / 75.98). Neither is an upper bound.
- **Base checkpoint history.** `outputs/cifar10_resnet18.pt` was 75.38 (original), then 76.40 (used by all final vision runs, now lost), then 76.80 (current, after an accidental `vision/common.py --help` retrain). Backup: `outputs/cifar10_resnet18_backup_1218_test76.80.pt`. A guard now needs `--retrain` to overwrite. Never run `vision/common.py` directly. Do not pool numbers across bases. DNF/UNSIR/USI(F)R configs were tuned on the 75.38 base and not re-tuned.
- **Read the earlier per-method bullets with these corrections.** In particular 74.45 / 71.06 / 74.97 described as "met" or "inside noise" above are overstated. Lead with the pattern (Acc_f ~0; Adam = drift; SGD noise-driven but shallow, probe ~29% vs ~5% floor), not the numbers.
- Step 5 needs the noise_n=0 control, R_t and the feature probe first (cleanup items 2 to 4 in `outputs/review/final_decision.md`).

## 2026-09-25: Step 3 cleanup complete

Files: `docs/results/consolidated_table.md` (+ `outputs/cleanup/table/`: 14 rows fresh on the current 76.80 base, 10 classes x 3 seeds, `summary.json`, `analyze.py`, `run_all.sh`), `docs/results/relearn_probe.md` (+ `vision/rt_probe.py`, `outputs/cleanup/rt_probe/results.json`), audit `docs/results/cleanup_supervisor.md` (+ `outputs/cleanup/audit/spot_probe.*`), new keys `consolidated_current_base` and `relearn_probe` in `docs/results/summary_tables.json`. Supervisor recomputed all numbers from raw JSON; spot check of DNF-SGD and Retrain40 (class 0) reproduces within noise. Base sha256 unchanged.

Key numbers (mean over 10 class means; SE over classes). SGD: no-noise control Acc_f 78.86 / Acc_r 76.42 / W_D 0.3; DNF 18000 0.25 / 76.46 / 5.2; USI(F)R nb72 3.74 / 77.48; UNSIR nb72 15.41 / 77.08. Adam: every row Acc_f 0.00, Acc_r 64.9-69.5; no-noise control 0.00 / 69.55 / W_D 88.9, no noise row beats it (paired -0.5 to -4.7, or +0.07 vs the E_r=2 control). FineTune 0.09 / 70.03; NegGrad 0.51 / 61.82. Probe and R_t (classes 0-4, seed 0): Retrain40 probe 36.6, DNF-SGD 57.3 (+20.7 over Retrain, SE 2.7, 5/5 classes), USI(F)R-SGD +25.8, UNSIR-SGD +27.4, FineTune +15.1, NegGrad +6.2, Adam methods and the Adam control +2 to +6; SGD methods relearn to 50% in 4-21 epochs (classes 2, 3 censored by the 50% threshold), Retrain and Adam methods mostly >30.

Supersedes: the "probe ~29% vs ~5% floor" wording above. The floor is protocol dependent (same Retrain model: 23% with Sup2's probe, 50% with the held-out-data probe; cause: probe trained on model-fit versus held-out data). Protocol-independent statement: DNF-SGD keeps about half of the base-over-floor separability and sits well above a retrained model on the same class.

Final statement: on one overfit CIFAR-10 base (single seed, class as unit), SGD noise samples cause forgetting versus a no-noise control (10/10 classes) without retain loss, but the forgetting is shallow; all Adam rows and FineTune are fine-tune drift (the no-noise Adam control forgets fully and no noise row beats it); learned noise is no better than Gaussian noise; the paper's ordering and targets are not reproduced (DNF-SGD config tuned post hoc on an older base). Unknown: whether noise content or the forget label drives the SGD effect, robustness to another base seed, depth of forgetting under a stronger relearn budget, cause of the missed 74.97, transfer to embedding noise in XLM-R.

Step-5 gate: done for the controlled table, R_t/probe, PROGRESS wording and step-5 prep; Retrain at 40 epochs partial (5 classes x 1 seed plus 3 x 1; label stays "reference, not upper bound"). Remaining before or during step 5: no hard blocker. Do: port R_t/probe to text (frozen features, stronger relearn budget, report forget acc after N epochs rather than only a 50% threshold, and the paired excess over Retrain), add a random-label noise control, multiple base seeds on text, quote no absolute probe floor.

## 2026-10-18: Step 5 first pass (English SIB-200, DNF with embedding-output noise)

Files: `text/unlearn.py` (DNF + noise_n=0 control, one pipeline), `text/analyze_step5.py`; raw rows `outputs/step5/*.jsonl`. Base: `outputs/final_run/text/baseline_eng_Latn_seed0.pt` only. Each of the 7 classes is forgotten in turn (class = unit, SE over 7). retain_sub 100/class, 1 impair + 1 repair epoch, batch 16, L=64, length-matched noise masks, real `<s>` at position 0, noise stats from retain train only. Selection on validation only; the test numbers below were not used for choosing anything.

Quantity: control Acc_f minus noise Acc_f on test (positive = noise forgets more than the identical pipeline with zero noise).

| Variant | lr | paired diff, test Acc_f (SE) |
|---|---|---|
| AdamW per_dim, noise 1x | 1e-6 / 3e-6 / 1e-5 / 3e-5 | -7.0 (1.4) / -2.2 (1.0) / +0.7 (4.4) / -15.7 (10.8) |
| AdamW per_dim, noise 2x | same | -8.0 (1.6) / +2.7 (1.8) / -1.6 (3.3) / -10.8 (13.8) |
| N(0,1) noise, 2x | 1e-5 / 3e-5 | -5.6 (4.5) / -25.4 (9.2) |
| noise at position 0, 2x | 1e-5 / 3e-5 | -3.9 (4.9) / -11.3 (6.5) |
| random-label noise, 2x | 1e-5 / 3e-5 | +5.6 (4.4) / -6.0 (2.7) |
| SGD-momentum per_dim, 2x | 1e-3 / 1e-2 / 3e-2 | 0.0 (3.2) / +5.3 (7.8) / +7.4 (7.5), val Acc_r -1.0 / -2.6 / -13.8 |

Findings (one base seed, English only, 204-sentence test set):
- No variant shows noise forgetting more than the no-noise control beyond its SE. Several show noise forgetting less (AdamW 1e-6, N(0,1) and position-0 at 3e-5).
- The control alone forgets: test Acc_f falls from about 90 to 27 at AdamW 3e-5 (Acc_r -2). This is retain-only fine-tune drift, the same artefact as the vision Adam rows.
- Acc_f never approaches 0 at a usable Acc_r. The only near-zero rows are SGD lr 3e-2, where Acc_r collapses to 47-61 (destructive).
- Random-label noise forgets about as much as forget-label noise, so the forget label is not what drives any effect.

Status: no evidence for contribution 1 on this configuration. This is a preliminary NO-GO, not a final one. Not yet tried: seeds 1 and 2, repair epochs >1 (USI(F)R), a learned-noise variant, other noise scales, stronger relearn-based depth tests (R_t, probe). Deadline for the go/no-go call stays 2026-10-25; if seeds 1-2 and a scale sweep agree, switch to the negative-results framing per CLAUDE.md.

## 2026-10-18: Step 5 go/no-go result (seeds, scale sweep, USI(F)R)

Raw: `outputs/step5/{seeds12,scale,usifr_s0,pooled}.jsonl`. Same protocol and quantity as the entry above (control Acc_f minus noise Acc_f, test, positive = noise forgets more). Selection was never done on test.

- **DNF, 3 base seeds pooled** (21 class-by-seed pairs, 2x noise, AdamW per_dim): lr 3e-6 +0.7 (SE 1.1), 1e-5 +1.9 (2.1), 3e-5 +1.2 (6.0). All include zero; per-seed signs flip at 3e-5 (-10.7 / +9.2 / +5.2). Control Acc_f 81.4 / 69.7 / 46.9 at those lrs.
- **Noise scale x0.5 / x2** (seed 0): lr 1e-5 -3.1 (4.3) / -1.2 (2.9); lr 3e-5 -15.6 (6.6) / -6.6 (12.0). No gain.
- **USI(F)R** (learned noise, alpha 1, 30 steps x 4 batches of 64 from per_dim init, tiled to 2x retain; 2 repair epochs; matched 2-repair-epoch control; seed 0): lr 1e-5 -13.5 (7.2), lr 3e-5 -21.7 (6.5). Learned noise forgets less than the control, and costs about 27 s versus 8.5 s. Alpha not swept; one setting.

Verdict: NO-GO on the evidence in hand for contribution 1 (English, XLM-R, DNF and USI(F)R as implemented here). Embedding-output noise gives no forgetting beyond what retain-only fine-tuning already does, and Acc_f does not approach 0 at usable Acc_r. Limits: Adam-family and SGD-momentum only, 1 impair + 1 repair epoch, ~100 retain/class, one learned-noise setting, no R_t or probe yet (a depth test cannot rescue a result where Acc_f itself does not beat control). The go/no-go deadline is 2026-10-25; per the hard rule this points to the negative-results framing.

## Current status (2026-10-18)

| Step | State |
|---|---|
| 1 Baseline XLM-R, English SIB-200 | Done. Test 90.69 +- 1.30 (3 seeds) vs 92.5 reported. |
| 2 UNSIR on CIFAR-10 | Done (70.95 was classes 0 and 2 jointly). |
| 3 USI(F)R + DNF on CIFAR-10 | PARTIAL. Acc_f ~0 everywhere; Adam rows are fine-tune drift; only DNF-SGD meets its retain target (chosen post hoc). |
| 4 Embedding-output injection | Done. Bypass (`roberta.encoder` direct) chosen over `inputs_embeds=`; logits match exactly; +0.014 GB VRAM. |
| 5 English go/no-go | Done. NO-GO: noise does not beat the zero-noise control (DNF 3 seeds, scale sweep, USI(F)R seed 0). |
| 6 Scale out | Not started; awaiting decision. |

Decision pending (deadline 2026-10-25): switch to negative-results framing (recommended) or spend a few days on rescue attempts (repair budget, alpha sweep, forget-class-derived noise).

Open items:
- Negative-result framing needs: same test on Urdu and Swahili, native corpora (MasakhaNEWS), retrain-from-scratch reference, FineTune/NegGrad baselines on text, R_t and probe port.
- Citation "Paper [3]" in CLAUDE.md is unnamed; a reviewer suggested Wichert and Sikdar 2024, not verified against the paper.
- Dr. Aamir Wali request status unknown.
- Repo still uncommitted; `docs/pdf_build/.venv-pdf` should be gitignored.

## 2026-10-21: Languages, epochs, error-maximising noise loss (English seed 0 unless stated)

Raw: `outputs/step5/{lang_urd_Arab,lang_swh_Latn,epochs3_s0,negce_r1_s0,negce_r2_s0}.jsonl`. New baselines (3 seeds, `outputs/final_run/text/`): Urdu test 84.6 +- 0.7, Swahili 77.1 +- 2.2 (English 90.7 +- 1.3). `text/unlearn.py` now takes `--lang`, `--max_len` (112 for UR/SW), `--epochs`, `--noise_loss {alg1,unsir}`.

Control Acc_f minus noise Acc_f, test, DNF 2x noise, 3 seeds x 7 classes (positive = noise forgets more):
- Urdu: +1.7 (1.2), +5.1 (3.0), +1.1 (4.3) at lr 3e-6, 1e-5, 3e-5.
- Swahili: +3.8 (1.9), +4.5 (4.0), -4.1 (3.0). No consistent direction.
- 3 impair + 3 repair epochs (seed 0): lr 1e-5 control Acc_f 17.3, noise 47.6, diff -30.2 (8.3). More epochs make the control forget more; noise forgets less. Needs seeds 1-2 before quoting.
- Error-maximising loss (-CE + 0.009 mean||N||^2, learned noise, ~23 s vs ~8 s): repair 1 epoch lr 1e-5 / 3e-5: -0.9 (4.0) / -10.6 (8.0); repair 2 epochs: -10.0 (6.7) / -17.8 (7.4). Still no gain over the matched control, so the sign of the CE term does not explain the null result.

Not yet done: seeds 1-2 for the epoch and -CE runs, alpha sweep, noise ratio 3:1, retrain and head-edit references, R_t and probe on text.
