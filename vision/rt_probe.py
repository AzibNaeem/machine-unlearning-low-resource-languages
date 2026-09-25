"""Linear probe + relearn-time (R_t) for unlearned CIFAR-10 ResNet18 models. Import-only use of common."""
import copy, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import common as C
from vision.dnf import dnf_unlearn
from vision.usifr import unsir_unlearn, usifr_unlearn
from vision.baselines import finetune, neggrad, retrain

OUT = os.path.join(C.ROOT, 'outputs', 'cleanup', 'rt_probe', 'results.json')
CLASSES = [0, 1, 2, 3, 4]
SEED = 0
dev = torch.device(C.device)
train_ds, valid_ds = C.load_cifar10()
base = C.load_base_model()


def val_indices(seed=SEED, n_val=200):
    """replicates class_split's permutation: first n_val per class = held-out val (never used by any method)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for c in range(10):
        idx = (train_ds.y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)].tolist()
        out[c] = idx[:n_val]
    return out


VAL = val_indices()
PX = torch.cat([train_ds.x[VAL[c]] for c in range(10)])
PY = torch.cat([train_ds.y[VAL[c]] for c in range(10)])


@torch.no_grad()
def feats(model, x):
    m = copy.deepcopy(model).eval()
    m.fc = nn.Identity()
    out = []
    for i in range(0, len(x), 500):
        out.append(m(x[i:i + 500].to(dev)).cpu())
    return torch.cat(out).numpy()


def probe(model, fc):
    ftr, fte = feats(model, PX), feats(model, valid_ds.x)
    sc = StandardScaler().fit(ftr)
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(sc.transform(ftr), PY.numpy())
    pred = clf.predict(sc.transform(fte))
    y = valid_ds.y.numpy()
    fm = y == fc
    return 100 * (pred[fm] == fc).mean(), 100 * (pred[~fm] == y[~fm]).mean()


def relearn(model, sp, fc, base_facc, max_ep=30):
    """fine-tune on 100 forget-class train imgs + 100 imgs per retain class (from retain_sub); Adam 1e-4, bs 50."""
    m = copy.deepcopy(model)
    fx = torch.stack([x for x, y in sp['forget_train'][:100]]); fy = torch.full((100,), fc)
    rs = sp['retain_sub']; cnt = {}; rx, ry = [], []
    for x, y in rs:
        if cnt.get(y, 0) < 100:
            cnt[y] = cnt.get(y, 0) + 1; rx.append(x); ry.append(y)
    X = torch.cat([fx, torch.stack(rx)]); Y = torch.cat([fy, torch.tensor(ry)])
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    g = torch.Generator().manual_seed(0)
    accs = [100 * C.evaluate(m, sp['forget_valid_dl'])['acc']]
    for ep in range(max_ep):
        m.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 50):
            b = perm[i:i + 50]
            loss = F.cross_entropy(m(X[b].to(dev)), Y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        accs.append(100 * C.evaluate(m, sp['forget_valid_dl'])['acc'])
    first = lambda thr: next((e for e, a in enumerate(accs) if a >= thr), None)  # index 0 = before any relearning
    return accs, first(50.0), first(0.9 * base_facc)


def methods(sp, fc):
    rs = sp['retain_sub']
    d = lambda o, n: (lambda: dnf_unlearn(base, rs, fc, noise_n=n, batch_size=250, impair_lr=0.02, repair_lr=0.01, opt_name=o, seed=SEED))
    ucom = dict(opt_name=None, noise_epochs=5, noise_steps=8, noise_lr=0.1, impair_lr=0.02, repair_lr=0.01, noise_train_mode=True)
    def un(o, nb): return lambda: unsir_unlearn(copy.deepcopy(base), rs, fc, seed=SEED, noise_batches=nb, device=dev, **{**ucom, 'opt_name': o})
    def us(o, nb): return lambda: usifr_unlearn(copy.deepcopy(base), rs, fc, alpha=10.0, repair_epochs=2, seed=SEED, noise_batches=nb, device=dev, **{**ucom, 'opt_name': o})
    return {
        'DNF-SGD n18000': d('sgd', 18000), 'DNF-Adam n18000': d('adam', 18000),
        'DNF-SGD n0 (control)': d('sgd', 0), 'DNF-Adam n0 (control)': d('adam', 0),
        'UNSIR-SGD nb72': un('sgd', 72), 'UNSIR-Adam nb20': un('adam', 20),
        'USIFR-SGD nb72': us('sgd', 72), 'USIFR-Adam nb5': us('adam', 5),
        'FineTune': lambda: finetune(base, rs, lr=0.01, epochs=5),
        'NegGrad': lambda: neggrad(base, sp['forget_train'], lr=1e-3, epochs=3),
        'Retrain40': lambda: retrain(sp['retain_train'], epochs=40),
        'Base': lambda: copy.deepcopy(base),
    }


rows = json.load(open(OUT))['rows'] if os.path.exists(OUT) and '--resume' in sys.argv else []
done = {(r['method'], r['cls']) for r in rows}
for fc in CLASSES:
    sp = C.class_split(train_ds, valid_ds, [fc], 1000, SEED)
    base_facc = 100 * C.evaluate(base, sp['forget_valid_dl'])['acc']
    for name, fn in methods(sp, fc).items():
        if (name, fc) in done:
            continue
        C.set_seed(SEED); t = time.time()
        m = fn(); m.eval()
        pf, pr = probe(m, fc)
        accs, r50, r90 = relearn(m, sp, fc, base_facc)
        row = dict(method=name, cls=fc, Acc_f=100 * C.evaluate(m, sp['forget_valid_dl'])['acc'],
                   Acc_r=100 * C.evaluate(m, sp['retain_valid_dl'])['acc'], W_D=C.weight_distance(m, base),
                   probe_f=pf, probe_r=pr, base_Acc_f=base_facc, Rt50=r50, Rt90base=r90, relearn_curve=accs, sec=time.time() - t)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'relearn_curve'}), flush=True)
        json.dump({'rows': rows}, open(OUT, 'w'), indent=1)
print('DONE', flush=True)
