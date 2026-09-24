import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import common as C
from vision.usifr import unsir_unlearn, usifr_unlearn


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["unsir", "usifr"])
    p.add_argument("--classes", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eval", choices=["test", "val"], default="test")
    p.add_argument("--out", default=os.path.join(C.ROOT, "outputs", "usifr_results.json"))
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--repair_epochs", type=int, default=2)
    p.add_argument("--noise_batches", type=int, default=5)
    p.add_argument("--unsir_noise_batches", type=int, default=20)
    p.add_argument("--opt", default="adam")
    p.add_argument("--noise_epochs", type=int, default=5)
    p.add_argument("--noise_steps", type=int, default=8)
    p.add_argument("--noise_lr", type=float, default=0.1)
    p.add_argument("--noise_eval_bn", action="store_true")
    p.add_argument("--impair_lr", type=float, default=0.02)
    p.add_argument("--repair_lr", type=float, default=0.01)
    return p.parse_args()


def main():
    a = parse()
    train_ds, valid_ds = C.load_cifar10()
    base = C.load_base_model()
    fkey, rkey = ("forget_valid_dl", "retain_valid_dl") if a.eval == "test" else ("val_forget_dl", "val_retain_dl")
    common = dict(opt_name=a.opt, noise_epochs=a.noise_epochs,
                  noise_steps=a.noise_steps, noise_lr=a.noise_lr, impair_lr=a.impair_lr,
                  repair_lr=a.repair_lr, noise_train_mode=not a.noise_eval_bn)
    runs = []
    for method in a.methods:
        for cls in a.classes:
            for seed in a.seeds:
                C.set_seed(seed)
                sp = C.class_split(train_ds, valid_ds, [cls], 1000, seed)
                if method == "unsir":
                    m = unsir_unlearn(copy.deepcopy(base), sp["retain_sub"], cls, seed=seed,
                                      noise_batches=a.unsir_noise_batches,
                                      device=torch.device(C.device), **common)
                else:
                    m = usifr_unlearn(copy.deepcopy(base), sp["retain_sub"], cls, alpha=a.alpha,
                                      repair_epochs=a.repair_epochs, seed=seed, noise_batches=a.noise_batches,
                                      device=torch.device(C.device), **common)
                r = dict(method=method, forget_class=cls, seed=seed, eval=a.eval,
                         Acc_f=C.evaluate(m, sp[fkey])["acc"] * 100,
                         Acc_r=C.evaluate(m, sp[rkey])["acc"] * 100,
                         W_D=C.weight_distance(m, base), time_s=m.unlearn_time)
                runs.append(r)
                print(json.dumps(r), flush=True)
    summary = {}
    for method in a.methods:
        rs = [r for r in runs if r["method"] == method]
        summary[method] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0]
                           for k in ("Acc_f", "Acc_r", "W_D", "time_s")}
        print(method, {k: f"{v[0]:.2f}+-{v[1]:.2f}" for k, v in summary[method].items()}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"args": vars(a), "runs": runs, "summary_mean_std": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
