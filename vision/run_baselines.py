import argparse
import json
import os
import time
import numpy as np
from vision.common import (ROOT, load_cifar10, load_base_model, class_split, evaluate,
                           weight_distance, set_seed)
from vision.baselines import finetune, neggrad, retrain

OUT = os.path.join(ROOT, 'outputs', 'baseline_results.json')
GRIDS = {
    'finetune': [dict(lr=lr, epochs=e) for lr in (1e-3, 1e-2) for e in (1, 5)],
    'neggrad': [dict(lr=lr, epochs=e) for lr in (1e-5, 1e-4, 1e-3) for e in (1, 3)],
}


def run_method(name, base, s, cfg, retrain_epochs):
    if name == 'finetune':
        return finetune(base, s['retain_sub'], **cfg)
    if name == 'neggrad':
        return neggrad(base, s['forget_train'], **cfg)
    return retrain(s['retain_train'], epochs=retrain_epochs)


def tune(name, base, train_ds, valid_ds, cls):
    s = class_split(train_ds, valid_ds, [cls], 1000, 0)
    best = None
    for cfg in GRIDS[name]:
        set_seed(0)
        m = run_method(name, base, s, cfg, 0)
        f, r = evaluate(m, s['val_forget_dl'])['acc'], evaluate(m, s['val_retain_dl'])['acc']
        score = r if f <= 0.02 else -f
        print(name, cfg, f'val_f {f:.3f} val_r {r:.3f}', flush=True)
        if best is None or score > best[0]:
            best = (score, cfg)
    return best[1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--retrain_epochs', type=int, default=40)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--methods', nargs='+', default=['finetune', 'neggrad', 'retrain'])
    a = ap.parse_args()
    train_ds, valid_ds = load_cifar10()
    base = load_base_model()
    cfgs = {m: tune(m, base, train_ds, valid_ds, 0) for m in a.methods if m in GRIDS}
    print('chosen', cfgs, flush=True)
    rows = []
    for cls in a.classes:
        for seed in a.seeds:
            s = class_split(train_ds, valid_ds, [cls], 1000, seed)
            for name in a.methods:
                set_seed(seed)
                t = time.time()
                m = run_method(name, base, s, cfgs.get(name), a.retrain_epochs)
                dt = time.time() - t
                row = dict(method=name, cls=cls, seed=seed,
                           Acc_f=evaluate(m, s['forget_valid_dl'])['acc'],
                           Acc_r=evaluate(m, s['retain_valid_dl'])['acc'],
                           W_D=weight_distance(m, base), time=dt)
                rows.append(row)
                print(row, flush=True)
                json.dump({'config': cfgs, 'retrain_epochs': a.retrain_epochs, 'rows': rows}, open(a.out, 'w'), indent=1)
    summary = {}
    for name in a.methods:
        r = [x for x in rows if x['method'] == name]
        summary[name] = {k: (float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r], ddof=1))) for k in ('Acc_f', 'Acc_r', 'W_D', 'time')}
    json.dump({'config': cfgs, 'retrain_epochs': a.retrain_epochs, 'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1)
    for n, v in summary.items():
        print(n, {k: f'{m:.4f}+-{sd:.4f}' for k, (m, sd) in v.items()})
