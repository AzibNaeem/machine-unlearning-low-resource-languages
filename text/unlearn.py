import argparse
import copy
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import get_loaders
from embed_inject import embed, forward_from_embeddings
from embed_stats import compute_stats, load_model, make_noise
from train_baseline import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT = os.path.join(ROOT, "outputs", "final_run", "text", "baseline_{lang}_seed{seed}.pt")
OUT = os.path.join(ROOT, "outputs", "step5")
L = 64


def set_len(n):
    global L
    L = n


def retain_subset(ids, mask, y, fc, per_class, seed):
    g = np.random.RandomState(seed)
    keep = []
    for c in y.unique().tolist():
        if c == fc:
            continue
        idx = (y == c).nonzero().squeeze(1).numpy()
        keep += g.choice(idx, min(per_class, len(idx)), replace=False).tolist()
    keep = torch.tensor(sorted(keep))
    return ids[keep][:, :L], mask[keep][:, :L], y[keep]


def build_resample(model, r_ids, r_mask, r_y, n, seed):
    """Step-matched control arm: n extra REAL retain examples (resampled with replacement,
    true labels), embedded the same way noise is, to isolate extra-steps-alone from noise identity."""
    g = torch.Generator().manual_seed(seed + 999)
    idx = torch.randint(len(r_y), (n,), generator=g)
    with torch.no_grad():
        emb = embed(model, r_ids[idx].to(device)).cpu()
    return (emb, r_mask[idx].clone()), r_y[idx].clone()


def build_noise(model, stats, lengths, n, mode, scale, noise_pos0, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    noise = make_noise((n, L, 768), stats, mode=mode, generator=g, scale=scale)
    if not noise_pos0:
        with torch.no_grad():
            bos = embed(model.eval(), torch.tensor([[0]], device=device))[0, 0]
        noise[:, 0] = bos
    lens = lengths[torch.randint(len(lengths), (n,), generator=torch.Generator().manual_seed(seed))].clamp(max=L)
    mask = (torch.arange(L)[None] < lens[:, None]).long()
    return noise.cpu(), mask


def learn_noise(model, noise, fc, alpha, steps, lr, bs=64, loss_kind="alg1", reg=0.009):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x, m = noise[0].clone(), noise[1]
    for i in range(0, len(x), bs):
        n = x[i:i + bs].to(device).requires_grad_(True)
        mk = m[i:i + bs].to(device)
        opt = torch.optim.Adam([n], lr=lr)
        tgt = torch.full((len(n),), fc, device=device)
        for _ in range(steps):
            logits = forward_from_embeddings(model, torch.cat([n[:, :1].detach(), n[:, 1:]], 1), mk)
            dist = (logits.softmax(-1) - F.one_hot(tgt, logits.shape[-1])).norm(dim=-1).mean()
            if loss_kind == "unsir":
                loss = -F.cross_entropy(logits, tgt) + reg * n.pow(2).sum((1, 2)).mean()
            else:
                loss = F.cross_entropy(logits, tgt) - alpha * dist
            opt.zero_grad()
            loss.backward()
            opt.step()
        x[i:i + bs] = n.detach().cpu()
    for p in model.parameters():
        p.requires_grad_(True)
    return x, m


def make_opt(model, name, lr):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def impair_epoch(model, opt, real, noise, noise_y, bs, g):
    ids, mask, y = real
    n_r = len(y)
    n_n = 0 if noise is None else len(noise_y)
    model.train()
    for b in torch.randperm(n_r + n_n, generator=g).split(bs):
        r, n = b[b < n_r], b[b >= n_r] - n_r
        parts, masks, ys = [], [], []
        if len(r):
            parts.append(embed(model, ids[r].to(device)))
            masks.append(mask[r])
            ys.append(y[r])
        if len(n):
            parts.append(noise[0][n].to(device))
            masks.append(noise[1][n])
            ys.append(noise_y[n])
        logits = forward_from_embeddings(model, torch.cat(parts), torch.cat(masks).to(device))
        loss = F.cross_entropy(logits, torch.cat(ys).to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def repair_epoch(model, opt, real, bs, g):
    ids, mask, y = real
    model.train()
    for b in torch.randperm(len(y), generator=g).split(bs):
        loss = F.cross_entropy(model(input_ids=ids[b].to(device), attention_mask=mask[b].to(device)).logits, y[b].to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def dnf_text(base, real, noise, noise_y, cfg, seed):
    model = copy.deepcopy(base)
    set_seed(seed)  # also seeds global/CUDA RNG, which dropout during impair/repair draws from
    g = torch.Generator().manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    opt = make_opt(model, cfg["opt"], cfg["lr"])
    for _ in range(cfg["impair_epochs"]):
        impair_epoch(model, opt, real, noise, noise_y, cfg["bs"], g)
    opt = make_opt(model, cfg["opt"], cfg["lr"])
    for _ in range(cfg["repair_epochs"]):
        repair_epoch(model, opt, real, cfg["bs"], g)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model.eval(), time.time() - t0


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, ys = [], []
    for ids, mask, y in loader:
        preds.append(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu())
        ys.append(y)
    return torch.cat(preds), torch.cat(ys)


def score(model, loader, fc):
    p, y = predict(model, loader)
    f = y == fc
    return {"acc_f": 100 * (p[f] == fc).float().mean().item(), "acc_r": 100 * (p[~f] == y[~f]).float().mean().item(),
            "n_f": int(f.sum())}


def weight_distance(a, b):
    return torch.sqrt(sum(((p - q) ** 2).sum() for p, q in zip(a.parameters(), b.parameters()))).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="eng_Latn")
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--classes", type=int, nargs="+", default=list(range(7)))
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    p.add_argument("--opts", nargs="+", default=["adamw"])
    p.add_argument("--noise_mult", type=float, nargs="+", default=[0, 1, 2])
    p.add_argument("--modes", nargs="+", default=["per_dim"])
    p.add_argument("--scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--label", choices=["forget", "random"], default="forget")
    p.add_argument("--fill", choices=["noise", "resample"], default="noise",
                   help="resample = step-matched control: extra slots filled with real retain data (true labels) instead of noise")
    p.add_argument("--noise_pos0", action="store_true")
    p.add_argument("--per_class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--learn", action="store_true")
    p.add_argument("--noise_loss", choices=["alg1", "unsir"], default="alg1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--repair_epochs", type=int, default=None)
    p.add_argument("--noise_steps", type=int, default=30)
    p.add_argument("--noise_lr", type=float, default=0.01)
    p.add_argument("--n_learn", type=int, default=256)
    p.add_argument("--tag", default="grid")
    a = p.parse_args()

    set_len(a.max_len)
    loaders, _ = get_loaders(a.lang, 32, verbose=False)
    ids, mask, y = loaders["train"].dataset.tensors
    lengths = mask.sum(1)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.tag}.jsonl")
    out = open(path, "a")

    for seed in a.seeds:
        base = load_model(CKPT.format(lang=a.lang, seed=seed))
        base_cpu = copy.deepcopy(base).cpu()
        for fc in a.classes:
            r_ids, r_mask, r_y = retain_subset(ids, mask, y, fc, a.per_class, seed)
            keep = y != fc
            rl = DataLoader(TensorDataset(ids[keep], mask[keep], y[keep]), batch_size=32)
            stats = compute_stats(base, rl)
            for mult, mode, scale, opt, lr in itertools.product(a.noise_mult, a.modes, a.scales, a.opts, a.lrs):
                n = int(mult * len(r_y))
                t_learn = 0.0
                if a.fill == "resample":
                    nz, ny = (build_resample(base, r_ids, r_mask, r_y, n, seed) if n else (None, None))
                else:
                    nz = build_noise(base, stats, lengths, n, mode, scale, a.noise_pos0, seed) if n else None
                    if nz is not None and a.learn:
                        t0 = time.time()
                        k = min(a.n_learn, n)
                        pool = learn_noise(copy.deepcopy(base), (nz[0][:k], nz[1][:k]), fc, a.alpha, a.noise_steps, a.noise_lr, loss_kind=a.noise_loss)
                        rep = -(-n // k)
                        nz = (pool[0].repeat(rep, 1, 1)[:n], pool[1].repeat(rep, 1)[:n])
                        t_learn = time.time() - t0
                    if a.label == "forget":
                        ny = torch.full((n,), fc)
                    else:
                        others = torch.tensor([c for c in range(7) if c != fc])
                        ny = others[torch.randint(len(others), (n,), generator=torch.Generator().manual_seed(seed))]
                cfg = {"opt": opt, "lr": lr, "bs": a.bs, "impair_epochs": a.epochs, "repair_epochs": a.repair_epochs or a.epochs}
                model, t = dnf_text(base, (r_ids, r_mask, r_y), nz, ny, cfg, seed)
                row = {"lang": a.lang, "seed": seed, "fc": fc, "mult": mult, "mode": mode, "scale": scale, "label": a.label,
                       "fill": a.fill, "pos0": a.noise_pos0, "learn": a.learn, "noise_loss": a.noise_loss, "alpha": a.alpha, **cfg, "time": t + t_learn, "w_d": weight_distance(model, base),
                       **{f"{s}_{k}": v for s, ld in (("val", loaders["validation"]), ("test", loaders["test"]))
                          for k, v in score(model, ld, fc).items()}}
                out.write(json.dumps(row) + "\n")
                out.flush()
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ("seed", "fc", "mult", "lr", "val_acc_f", "val_acc_r", "test_acc_f", "test_acc_r", "w_d", "time")})
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
