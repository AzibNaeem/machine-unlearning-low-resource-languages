import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import MODEL_NAME, encode, load_sib200
from embed_inject import embed, forward_from_embeddings

device = torch.device("cuda")
GB = 2**30
HIDDEN = 768


def load_model(ckpt):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def timed(step, warm=1, reps=3):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    return {
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
        "step_s": round((time.time() - t0) / reps, 4),
    }


def guarded(fn):
    try:
        return fn()
    except torch.OutOfMemoryError:
        return {"oom": True}
    finally:
        cleanup()


def batch_of(data, B, L, tok, cat_to_id, split):
    ds = encode(split, tok, cat_to_id, L)
    ids, mask, y = ds.tensors
    idx = torch.arange(B) % len(ids)
    return ids[idx].to(device), mask[idx].to(device), y[idx].to(device)


def impair_step(model, opt, kind, ids, mask, y, amp):
    B, L = ids.shape

    def step():
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if kind == "plain":
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            else:
                h = B // 2
                real = embed(model, ids[:h])
                noise = torch.randn(B - h, L, HIDDEN, device=device)
                x = torch.cat([real, noise])
                m = torch.cat([mask[:h], torch.ones(B - h, L, dtype=mask.dtype, device=device)])
                logits = forward_from_embeddings(model, x, m)
                loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return step


def run_impair(model, B, L, kind, batch, amp=False):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    try:
        return timed(impair_step(model, opt, kind, *batch, amp))
    finally:
        del opt
        model.zero_grad(set_to_none=True)


def run_noise_learning(model, n, L, amp=False):
    y = torch.zeros(n, dtype=torch.long, device=device)
    mask = torch.ones(n, L, dtype=torch.long, device=device)
    noise = torch.randn(n, L, HIDDEN, device=device, requires_grad=True)
    opt = torch.optim.Adam([noise], lr=0.1)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = forward_from_embeddings(model, noise, mask).float()
        onehot = F.one_hot(y, logits.size(-1)).float()
        loss = F.cross_entropy(logits, y) - 0.1 * (logits.softmax(-1) - onehot).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return timed(step)


def largest_batch(model, L, cap=4096):
    lo, hi = 0, None
    n = 16
    while n <= cap:
        r = guarded(lambda: run_noise_learning(model, n, L))
        if r.get("oom"):
            hi = n
            break
        lo = n
        n *= 2
    if hi is None:
        return lo, True
    while hi - lo > 8:
        mid = (lo + hi) // 2 // 8 * 8
        if mid <= lo:
            break
        if guarded(lambda: run_noise_learning(model, mid, L)).get("oom"):
            hi = mid
        else:
            lo = mid
    return lo, False


def token_lengths(tok, split):
    enc = tok(split["text"], truncation=True, max_length=512)
    lens = torch.tensor([len(x) for x in enc["input_ids"]], dtype=torch.float)
    q = lambda p: float(lens.quantile(p))
    return {"n": len(lens), "mean": round(float(lens.mean()), 2), "median": q(0.5), "p95": q(0.95),
            "p99": q(0.99), "max": int(lens.max()), "min": int(lens.min())}


def storage_table(free_gb_after_model):
    rows = {}
    for L in (32, 64, 164):
        per = L * HIDDEN * 4
        rows[L] = {"bytes_per_sample": per, "mb_per_sample": round(per / 2**20, 4),
                   "n": {N: round(N * per / GB, 3) for N in (600, 1200, 5000, 18000)},
                   "max_samples_in_10gb": int(10 * GB // per)}
    return rows


def transfer_time(B, L):
    cpu = torch.randn(B, L, HIDDEN).pin_memory()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        cpu.to(device, non_blocking=True)
    torch.cuda.synchronize()
    return round((time.time() - t0) / 20 * 1000, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/final_run/text/baseline_eng_Latn_seed0.pt")
    p.add_argument("--out_dir", default="outputs/step4/C")
    p.add_argument("--lang", default="eng_Latn")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, "vram.json")
    res = {"gpu": torch.cuda.get_device_name(0), "total_gb": round(torch.cuda.get_device_properties(0).total_memory / GB, 2)}

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    ds = load_sib200(a.lang)
    cat_to_id = {c: i for i, c in enumerate(sorted(set(ds["train"]["category"])))}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    res["token_lengths_train"] = token_lengths(tok, ds["train"])
    print("token lengths", res["token_lengths_train"], flush=True)
    res["noise_storage_gb"] = storage_table(None)
    res["transfer_ms_B64_L164_pinned"] = None
    save()

    model = load_model(a.ckpt)
    torch.cuda.synchronize()
    res["after_model_load_alloc_gb"] = round(torch.cuda.memory_allocated() / GB, 3)
    res["transfer_ms_B64_L164_pinned"] = transfer_time(64, 164)
    print("model loaded", res["after_model_load_alloc_gb"], flush=True)

    res["impair"] = {}
    for L in (164, 64, 32):
        for B in (16, 32):
            batch = batch_of(ds["train"], B, L, tok, cat_to_id, ds["train"])
            r = {}
            for kind in ("plain", "mixed_noise"):
                r[kind] = guarded(lambda: run_impair(model, B, L, kind, batch))
                print("impair", B, L, kind, r[kind], flush=True)
            if not any(v.get("oom") for v in r.values()):
                r["extra_alloc_gb"] = round(r["mixed_noise"]["peak_alloc_gb"] - r["plain"]["peak_alloc_gb"], 3)
            r["noise_tensor_mb"] = round(B // 2 * L * HIDDEN * 4 / 2**20, 3)
            res["impair"][f"B{B}_L{L}"] = r
            save()

    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    res["noise_learning"] = {}
    for L in (32, 64, 164):
        for n in (16, 64, 256):
            r = guarded(lambda: run_noise_learning(model, n, L))
            res["noise_learning"][f"n{n}_L{L}"] = r
            print("noise", n, L, r, flush=True)
            save()
    res["largest_noise_batch"] = {}
    for L in (32, 64, 164):
        n, capped = largest_batch(model, L)
        res["largest_noise_batch"][f"L{L}"] = {"n": n, "hit_search_cap_4096": capped}
        print("largest", L, n, capped, flush=True)
        save()

    res["bf16"] = {}
    r = guarded(lambda: run_noise_learning(model, 64, 164, amp=True))
    res["bf16"]["noise_learning_n64_L164"] = {"fp32": res["noise_learning"]["n64_L164"], "bf16": r}
    print("bf16 noise", r, flush=True)
    r256 = guarded(lambda: run_noise_learning(model, 256, 64, amp=True))
    res["bf16"]["noise_learning_n256_L64"] = {"fp32": res["noise_learning"]["n256_L64"], "bf16": r256}
    save()

    for q in model.parameters():
        q.requires_grad_(True)
    for B in (16, 32):
        batch = batch_of(ds["train"], B, 164, tok, cat_to_id, ds["train"])
        r = {k: guarded(lambda: run_impair(model, B, 164, k, batch, amp=True)) for k in ("plain", "mixed_noise")}
        res["bf16"][f"impair_B{B}_L164"] = {"fp32": res["impair"][f"B{B}_L164"], "bf16": r}
        print("bf16 impair", B, r, flush=True)
        save()
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
