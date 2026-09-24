import csv
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

MAX_LEN = 164
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LOCAL_MODEL = os.path.join(ROOT, "xlmr")
MODEL_NAME = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "xlm-roberta-base"
LOCAL_SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def load_sib200(lang):
    # HF Davlan/sib200 was rate-limited from this machine; the local TSVs are the original dadelani/sib-200 release
    d = os.path.join(ROOT, "sib200", lang)
    if os.path.isdir(d):
        out = {}
        for split, f in LOCAL_SPLITS.items():
            with open(os.path.join(d, f + ".tsv"), encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            out[split] = {"text": [r["text"] for r in rows], "category": [r["category"] for r in rows]}
        return out
    from datasets import load_dataset
    return {k: v.to_dict() for k, v in load_dataset("Davlan/sib200", lang).items()}


def encode(split, tok, cat_to_id, max_len=MAX_LEN):
    enc = tok(split["text"], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    labels = torch.tensor([cat_to_id[c] for c in split["category"]])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


def get_loaders(lang="eng_Latn", batch_size=16, max_len=MAX_LEN, seed=0, verbose=True):
    ds = load_sib200(lang)
    cats = sorted(set(ds["train"]["category"]))
    cat_to_id = {c: i for i, c in enumerate(cats)}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if verbose:
        print("label mapping:", cat_to_id)
        for name, split in ds.items():
            print(name, len(split["text"]), dict(sorted(Counter(split["category"]).items())))
    sets = {k: encode(v, tok, cat_to_id, max_len) for k, v in ds.items()}
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(sets["train"], batch_size=batch_size, shuffle=True, generator=g),
        "validation": DataLoader(sets["validation"], batch_size=batch_size),
        "test": DataLoader(sets["test"], batch_size=batch_size),
    }
    return loaders, cat_to_id
