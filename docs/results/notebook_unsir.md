# Original notebook pipeline (UNSIR, ResNet18 / CIFAR-10)

**What was run:** the converted notebook `Machine Unlearning.ipynb` as a script (`outputs/final_run/notebook/notebook_pipeline.py`, log `run.log`, metrics `results.json`). Fresh run, single job, one seed (torch seed 100, GPU non-deterministic). Notebook hyperparameters unchanged; changes are num_workers=0 (Windows), a `__main__` guard, stage timing and JSON output.

**Config:** ResNet18 from scratch, 40 epochs, Adam lr 0.01, grad-clip 0.1 (value), weight decay 1e-4, batch 256, ReduceLROnPlateau, normalisation only. Forget classes 0 (airplane) and 2 (bird) jointly. Noise: 5 epochs x 8 steps, Adam lr 0.1, loss -CE + 0.1*||N||^2, one 256-sample noise batch per class repeated 20 times. Impair: 1 epoch Adam lr 0.02 on noise + first 1000 retain images per retain class. Repair: 1 epoch Adam lr 0.01 on the same retain subset. Evaluated on the CIFAR-10 test set (10000 images).

| Stage | Acc_f (classes 0,2) | Acc_r (8 retain classes) |
|---|---|---|
| Base model (supervisor re-evaluation of saved checkpoint) | 75.15 | 76.80 (overall 76.47) |
| After impair | 3.32 | 63.55 |
| After repair | 0.00 | 72.68 |

The base row was filled in by the supervisor, not by the worker: the saved `ResNET18_CIFAR10_ALL_CLASSES.pt` was evaluated with a plain sample-weighted accuracy. The notebook's own metric averages per-batch accuracies (log: overall 76.60), which is why impair/repair rows differ from an exact count by a fraction of a point at most. After repair, retain drops 4.1 points from base, forget goes from 75.15 to 0.

**Stage times (s):** base training 705.5 (17.6 s/epoch), noise learning 1.06, impair 0.41, repair 0.36; unlearning about 1.8 s in total.

**Quirk kept as in the notebook:** in the impair cell `class_num = 0` is never updated, so the noise learned for class 2 is also labelled 0 during impair. Class 2 images are not shown either way, so the impair step never explicitly targets class 2; whether Acc_f = 0.00 for class 2 comes from repair drift or from the noise cannot be told from this single run (no no-noise control was run).

**Verdict: reproduced as a pipeline (partially vs paper).** It runs and both classes drop to 0.00 after repair with retain 72.68 (step-2 notebook run: 70.95 / 0.0; paper UNSIR 71.06). Single seed, two-class forgetting with the label-0 quirk and a different base, so not comparable with the paper's single-class table or with the single-class UNSIR in `unsir.md`. The dataset path in the script is hardcoded to a temporary scratchpad folder of ImageFolder PNGs (not in the repo) and the checkpoint is written to the working directory.
