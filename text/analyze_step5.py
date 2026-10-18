import json
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.DataFrame([json.loads(l) for l in open(path)])
for c, d in (("learn", False), ("alpha", 0.0), ("repair_epochs", 1)):
    df[c] = df[c].fillna(d) if c in df else d
keys = ["opt", "lr", "mode", "scale", "label", "pos0", "learn", "alpha", "repair_epochs"]
cols = ["val_acc_f", "val_acc_r", "test_acc_f", "test_acc_r", "w_d"]

g = df.groupby(keys + ["mult"])[cols].mean().round(2)
n = df.groupby(keys + ["mult"])["fc"].nunique().rename("classes")
print(g.join(n).to_string())

print("\npaired vs noise_n=0 control (same lr), mean over classes of control Acc_f minus noise Acc_f; positive = noise forgets more")
for k, grp in df.groupby(keys):
    ctrl = grp[grp.mult == 0].set_index(["seed", "fc"])
    for m, sub in grp[grp.mult > 0].groupby("mult"):
        s = sub.set_index(["seed", "fc"])
        d = {sp: (ctrl[f"{sp}_acc_f"] - s[f"{sp}_acc_f"]).dropna() for sp in ("val", "test")}
        r = (s["val_acc_r"] - ctrl["val_acc_r"]).dropna()
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        print(k, "mult", m, f"d_val_acc_f {d['val'].mean():.2f}+-{se(d['val']):.2f}",
              f"d_test_acc_f {d['test'].mean():.2f}+-{se(d['test']):.2f}", f"d_val_acc_r {r.mean():.2f}", f"n={len(r)}")
