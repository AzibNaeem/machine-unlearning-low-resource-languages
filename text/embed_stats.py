import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from data import MODEL_NAME, get_loaders
from embed_inject import forward_from_embeddings as forward_from_embeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT = os.path.join(ROOT, "outputs", "final_run", "text", "baseline_eng_Latn_seed0.pt")
OUT = os.path.join(ROOT, "outputs", "step4", "B")
D = 768


def load_model(ckpt=None, num_labels=7):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def _dist(x):
    return {"mean": x.mean().item(), "std": x.std().item(), "min": x.min().item(), "max": x.max().item()}


def _pct(x):
    q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=x.dtype))
    return {"mean": x.mean().item(), "std": x.std().item(), "min": x.min().item(), "max": x.max().item(),
            **{f"p{p}": v.item() for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)}}


@torch.no_grad()
def compute_stats(model, loader):
    model.eval()
    real, pad, lens = [], [], []
    for ids, mask, _ in loader:
        e = model.roberta.embeddings(input_ids=ids.to(device)).double().cpu()
        m = mask.bool()
        real.append(e[m])
        pad.append(e[~m])
        lens.append(mask.sum(1))
    real, pad, lens = torch.cat(real), torch.cat(pad), torch.cat(lens).double()
    mean, std = real.mean(0), real.std(0)
    norms = real.norm(dim=1)
    ln = model.roberta.embeddings.LayerNorm
    gamma, beta = ln.weight.detach().double().cpu(), ln.bias.detach().double().cpu()
    order = std.abs().argsort(descending=True)
    sq = (real ** 2).mean(0)
    top_share = {k: (sq[sq.argsort(descending=True)[:k]].sum() / sq.sum()).item() for k in (1, 3, 10)}
    seq_len = loader.dataset.tensors[0].shape[1]
    cos_to_mean = F.cosine_similarity(real, mean.expand_as(real), dim=1)
    return {
        "mean": mean, "std": std, "gamma": gamma, "beta": beta,
        "norm_mean": norms.mean().item(), "norm_std": norms.std().item(),
        "scalar_mean": real.mean().item(), "scalar_std": real.std().item(), "rms": real.pow(2).mean().sqrt().item(),
        "n_real_tokens": real.shape[0], "n_pad_tokens": pad.shape[0], "seq_len": seq_len,
        "summary": {
            "real": {
                "per_dim_mean": _dist(mean), "per_dim_std": _dist(std), "token_norm": _pct(norms),
                "mean_vector_norm": mean.norm().item(), "cos_to_mean_vector": _dist(cos_to_mean),
                "outlier_dims_by_std": [{"dim": i.item(), "std": std[i].item(), "mean": mean[i].item()} for i in order[:6]],
                "median_dim_std": std.median().item(), "top_k_dims_energy_share": top_share,
                "expected_norm_N01": math.sqrt(D), "norm_ratio_real_over_N01": norms.mean().item() / math.sqrt(D),
            },
            "pad": {
                "per_dim_mean": _dist(pad.mean(0)), "per_dim_std": _dist(pad.std(0)), "token_norm": _pct(pad.norm(dim=1)),
                "scalar_std": pad.std().item(), "cos_real_mean_vs_pad_mean": F.cosine_similarity(mean, pad.mean(0), dim=0).item(),
            },
            "layernorm": {
                "gamma": _dist(gamma), "beta": _dist(beta),
                "gamma_top_dims": [{"dim": i.item(), "gamma": gamma[i].item(), "beta": beta[i].item()}
                                   for i in gamma.abs().argsort(descending=True)[:5]],
                "eps": ln.eps,
            },
            "lengths": {"mean": lens.mean().item(), "std": lens.std().item(), "min": lens.min().item(), "max": lens.max().item(),
                        "padded_len": seq_len, "real_fraction_of_padded": (lens.mean() / seq_len).item()},
            "dilution": {
                "mean_norm_over_padded_positions": ((real.shape[0] * norms.mean() + pad.shape[0] * pad.norm(dim=1).mean())
                                                    / (real.shape[0] + pad.shape[0])).item(),
                "pad_fraction_of_positions": pad.shape[0] / (pad.shape[0] + real.shape[0]),
            },
        },
    }


def make_noise(shape, stats, mode="match_norm", generator=None, device=device, scale=1.0):
    z = torch.randn(shape, generator=generator, device=device)
    mean, std = stats["mean"].float().to(device), stats["std"].float().to(device)
    if mode == "n01":
        return z * scale
    if mode == "match_norm":
        return z * (scale * stats["norm_mean"] / math.sqrt(shape[-1]))
    if mode == "std":
        return z * (scale * stats["scalar_std"]) + stats["scalar_mean"]
    if mode == "per_dim":
        return z * (scale * std) + mean
    if mode == "ln":
        g, b = stats["gamma"].float().to(device), stats["beta"].float().to(device)
        return F.layer_norm(z, (shape[-1],), eps=1e-5) * g + b
    raise ValueError(mode)


@torch.no_grad()
def noise_behaviour(model, stats, n=512, seq_len=32, bs=32, seed=0, ident_scale=None):
    g = torch.Generator(device=device).manual_seed(seed)
    cands = {"N01": None, "match_norm": {"mode": "match_norm"}, "std": {"mode": "std"},
             "per_dim": {"mode": "per_dim"}, "ln": {"mode": "ln"}}
    for s in (0.25, 0.5, 2.0):
        cands[f"match_norm_x{s}"] = {"mode": "match_norm", "scale": s}
    out = {}
    for name, kw in cands.items():
        probs = []
        for _ in range(n // bs):
            shape = (bs, seq_len, D)
            x = torch.randn(shape, generator=g, device=device) if kw is None else make_noise(shape, stats, generator=g, **kw)
            probs.append(forward_from_embeds(model, x).softmax(-1).cpu())
        p = torch.cat(probs)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
        pred = p.argmax(-1)
        out[name] = {
            "pred_hist": torch.bincount(pred, minlength=p.shape[1]).tolist(),
            "conf_mean": p.max(-1).values.mean().item(), "conf_std": p.max(-1).values.std().item(),
            "entropy_mean": ent.mean().item(), "entropy_max_possible": math.log(p.shape[1]),
            "mean_prob_per_class": p.mean(0).tolist(),
        }
    return out


@torch.no_grad()
def api_check(model, loader):
    correct = total = 0
    for ids, mask, y in loader:
        ids, mask = ids.to(device), mask.to(device)
        x = model.roberta.embeddings(input_ids=ids)
        ref = model(input_ids=ids, attention_mask=mask).logits
        mine = forward_from_embeds(model, x, mask)
        correct += (mine.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return {"acc_via_embed_path": correct / total, "max_abs_logit_diff_last_batch": (ref - mine).abs().max().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="eng_Latn")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--n", type=int, default=512)
    a = p.parse_args()

    loaders, _ = get_loaders(a.lang, 32, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name, ckpt in (("finetuned", a.ckpt), ("pretrained", None)):
        model = load_model(ckpt)
        st = compute_stats(model, loaders["train"])
        res[name] = {"n_real_tokens": st["n_real_tokens"], "n_pad_tokens": st["n_pad_tokens"], **st["summary"]}
        if ckpt:
            res[name]["api_check_test"] = api_check(model, loaders["test"])
            res[name]["noise_behaviour"] = noise_behaviour(model, st, a.n)
            torch.save({k: st[k] for k in ("mean", "std", "gamma", "beta", "norm_mean", "scalar_mean", "scalar_std")},
                       os.path.join(OUT, "stats_finetuned.pt"))
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("real", "lengths")} for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
