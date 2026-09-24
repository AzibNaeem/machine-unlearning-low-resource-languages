import argparse
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from data import MODEL_NAME, get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        pred = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="eng_Latn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--out_dir", default="outputs")
    a = p.parse_args()

    set_seed(a.seed)
    loaders, cat_to_id = get_loaders(a.lang, a.batch_size, seed=a.seed, verbose=a.seed == 0)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(cat_to_id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(loaders["train"])
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"baseline_{a.lang}_seed{a.seed}.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_val, best_epoch = -1.0, -1
    for ep in range(1, a.epochs + 1):
        model.train()
        loss_sum = 0.0
        for ids, mask, y in loaders["train"]:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
        val = evaluate(model, loaders["validation"])
        print(f"epoch {ep} train_loss {loss_sum / len(loaders['train']):.4f} val_acc {val:.4f}", flush=True)
        if val > best_val:
            best_val, best_epoch = val, ep
            torch.save(model.state_dict(), path)
    train_time = time.time() - t0

    model.load_state_dict(torch.load(path, map_location=device))
    test = evaluate(model, loaders["test"])
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    print(f"RESULT lang={a.lang} seed={a.seed} lr={a.lr} epochs={a.epochs} best_epoch={best_epoch} "
          f"val_acc={best_val:.4f} test_acc={test:.4f} peak_vram_gb={peak:.2f} time_s={train_time:.0f}")


if __name__ == "__main__":
    main()
