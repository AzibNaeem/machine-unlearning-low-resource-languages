import os
import sys

import torch
from transformers import AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from data import MODEL_NAME, get_loaders  # noqa: E402
from embed_inject import embed, forward_from_embeddings  # noqa: E402

CKPT = os.path.join(HERE, "..", "..", "outputs", "final_run", "text", "baseline_eng_Latn_seed0.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def bypass_logits(model, ids, mask):
    return forward_from_embeddings(model, embed(model, ids), mask)


def main():
    model = load_model()
    loaders, _ = get_loaders("eng_Latn", batch_size=16, verbose=False)
    val = loaders["validation"]
    ids, mask, y = (t.to(device) for t in next(iter(val)))

    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask).logits
    diff = (bypass_logits(model, ids, mask) - ref).abs().max().item()
    report("1 bypass == normal forward (16 real val sentences, full padding)", diff < 1e-5, f"max_abs_diff={diff:.3e}")

    ids_all = torch.cat([b[0] for b in val])[:32].to(device)
    mask_all = torch.cat([b[1] for b in val])[:32].to(device)
    diffs, same = [], True
    for i in range(0, 32, 16):
        a, m = ids_all[i:i + 16], mask_all[i:i + 16]
        with torch.no_grad():
            r = model(input_ids=a, attention_mask=m).logits
        b = bypass_logits(model, a, m)
        diffs.append((b - r).abs().max().item())
        same &= bool((b.argmax(-1) == r.argmax(-1)).all())
    report("2 32 val sentences, argmax identical", same and max(diffs) < 1e-5, f"max_abs_diff={max(diffs):.3e}")

    for p in model.parameters():
        p.requires_grad_(False)
    noise = torch.randn(4, ids.shape[1], 768, device=device, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(forward_from_embeddings(model, noise, mask[:4]), y[:4])
    loss.backward()
    g = noise.grad
    ok = g is not None and bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0
    report("3 gradient reaches noise with frozen params", ok, f"grad_abs_mean={g.abs().mean().item():.3e}")
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        e = embed(model, ids, None)
        e2 = e.clone()
        pad = mask == 0
        e2[pad] = torch.randn_like(e2[pad]) * 10
        d = (forward_from_embeddings(model, e2, mask) - forward_from_embeddings(model, e, mask)).abs().max().item()
    report("4 padded-position embeds do not affect logits", pad.any().item() and d < 1e-5, f"max_abs_diff={d:.3e}")

    c_norm = c_byp = n = 0
    with torch.no_grad():
        for a, m, t in val:
            a, m, t = a.to(device), m.to(device), t.to(device)
            c_norm += (model(input_ids=a, attention_mask=m).logits.argmax(-1) == t).sum().item()
            c_byp += (bypass_logits(model, a, m).argmax(-1) == t).sum().item()
            n += len(t)
    report("5 full validation accuracy, bypass == normal", c_norm == c_byp, f"normal={100*c_norm/n:.2f} bypass={100*c_byp/n:.2f} n={n}")

    print("ALL PASS" if all(results) else "SOME FAILED")


if __name__ == "__main__":
    main()
