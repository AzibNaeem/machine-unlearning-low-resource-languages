import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from dnf import dnf_unlearn

p = argparse.ArgumentParser()
p.add_argument("--forget_class", type=int, nargs="*", default=list(range(10)))
p.add_argument("--seed", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--noise_n", type=int, default=5120)
p.add_argument("--opt", default="adam")
p.add_argument("--batch_size", type=int, default=250)
p.add_argument("--impair_lr", type=float, default=0.02)
p.add_argument("--repair_lr", type=float, default=0.01)
p.add_argument("--n_retain", type=int, default=1000)
p.add_argument("--grid", type=int, nargs="*", default=[1000, 2500, 5120, 9000, 18000])
p.add_argument("--opts", nargs="*", default=["adam", "sgd"])
p.add_argument("--tune", action="store_true")
p.add_argument("--out", default=None)
a = p.parse_args()

train_ds, valid_ds = C.load_cifar10()
base = C.load_base_model()


def run(fc, seed, noise_n, opt, split_eval):
    C.set_seed(seed)
    sp = C.class_split(train_ds, valid_ds, [fc], a.n_retain, seed)
    fdl, rdl = (sp["val_forget_dl"], sp["val_retain_dl"]) if split_eval == "val" else (sp["forget_valid_dl"], sp["retain_valid_dl"])
    m = dnf_unlearn(base, sp["retain_sub"], fc, noise_n=noise_n, batch_size=a.batch_size,
                    impair_lr=a.impair_lr, repair_lr=a.repair_lr, opt_name=opt, seed=seed)
    return {"forget_class": fc, "seed": seed, "noise_n": noise_n, "opt": opt,
            "Acc_f": 100 * C.evaluate(m, fdl)["acc"], "Acc_r": 100 * C.evaluate(m, rdl)["acc"],
            "W_D": C.weight_distance(m, base), "time": m.unlearn_time,
            "base_Acc_f": 100 * C.evaluate(base, fdl)["acc"], "base_Acc_r": 100 * C.evaluate(base, rdl)["acc"]}


if a.tune:
    rows = []
    for opt in a.opts:
        for n in a.grid:
            rs = [run(fc, s, n, opt, "val") for fc in a.forget_class for s in a.seed]
            rows += rs
            print(f"{opt} noise_n={n}: val Acc_f {np.mean([r['Acc_f'] for r in rs]):.2f} "
                  f"Acc_r {np.mean([r['Acc_r'] for r in rs]):.2f} W_D {np.mean([r['W_D'] for r in rs]):.2f}", flush=True)
    out = a.out or os.path.join(C.ROOT, "outputs", "dnf_tune.json")
else:
    rows = []
    for fc in a.forget_class:
        for s in a.seed:
            r = run(fc, s, a.noise_n, a.opt, "test")
            rows.append(r)
            print(f"class {fc} seed {s}: Acc_f {r['Acc_f']:.2f} Acc_r {r['Acc_r']:.2f} W_D {r['W_D']:.2f} time {r['time']:.2f}s", flush=True)
    f, r_, w, t = ([r[k] for r in rows] for k in ["Acc_f", "Acc_r", "W_D", "time"])
    print(f"mean+-std over {len(rows)} runs: Acc_f {np.mean(f):.2f}+-{np.std(f):.2f} Acc_r {np.mean(r_):.2f}+-{np.std(r_):.2f} "
          f"W_D {np.mean(w):.2f}+-{np.std(w):.2f} time {np.mean(t):.2f}s | base Acc_r {np.mean([r['base_Acc_r'] for r in rows]):.2f}")
    out = a.out or os.path.join(C.ROOT, "outputs", "dnf_results.json")
json.dump(rows, open(out, "w"), indent=1)
