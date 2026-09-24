import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
CACHE = os.path.join(DATA_DIR, 'cifar10_tensors.pt')
BASE_CKPT = os.path.join(ROOT, 'outputs', 'cifar10_resnet18.pt')
EXTRACTED = r'C:\Users\Azib\AppData\Local\Temp\claude\d--University-FYP\b827bfb8-7153-4872-b3bb-281ec0b34ad3\scratchpad\data\cifar10'
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
NUM_WORKERS = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class TensorPairs(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.targets = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _decode_folder(root):
    from PIL import Image
    classes = sorted(os.listdir(root))
    xs, ys = [], []
    for c, name in enumerate(classes):
        d = os.path.join(root, name)
        for f in sorted(os.listdir(d)):
            xs.append(np.asarray(Image.open(os.path.join(d, f)).convert('RGB')))
            ys.append(c)
    return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous(), torch.tensor(ys)


def _build_cache():
    if os.path.isdir(EXTRACTED):
        xtr, ytr = _decode_folder(os.path.join(EXTRACTED, 'train'))
        xte, yte = _decode_folder(os.path.join(EXTRACTED, 'test'))
    else:
        from torchvision.datasets import CIFAR10
        tr = CIFAR10(DATA_DIR, train=True, download=True)
        te = CIFAR10(DATA_DIR, train=False, download=True)
        xtr, ytr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(tr.targets)
        xte, yte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(te.targets)
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save({'xtr': xtr, 'ytr': ytr, 'xte': xte, 'yte': yte}, CACHE)


def load_cifar10(batch_size=256):
    if not os.path.exists(CACHE):
        _build_cache()
    d = torch.load(CACHE)
    norm = lambda x: (x.float() / 255 - MEAN) / STD
    return TensorPairs(norm(d['xtr']), d['ytr']), TensorPairs(norm(d['xte']), d['yte'])


def make_loader(ds, batch_size=256, shuffle=False):
    return DataLoader(ds, batch_size, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return {'loss': loss / n, 'acc': correct / n}


def weight_distance(model, ref_model):
    sq = 0.0
    for p, q in zip(model.parameters(), ref_model.parameters()):
        sq += (p.detach() - q.detach()).pow(2).sum().item()
    return sq ** 0.5


def new_model():
    return resnet18(num_classes=10).to(device)


def load_base_model():
    model = new_model()
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device))
    return model.eval()


def fit(model, train_dl, epochs, lr, valid_dl=None, weight_decay=1e-4, grad_clip=0.1, log=True):
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())
        if valid_dl is not None:
            r = evaluate(model, valid_dl)
            sched.step(r['loss'])
            if log:
                print(f"epoch {epoch} train_loss {np.mean(losses):.4f} val_loss {r['loss']:.4f} val_acc {r['acc']:.4f}", flush=True)
    return model


def train_base_model(epochs=40, batch_size=256, lr=0.01, seed=100, overwrite=False):
    if os.path.exists(BASE_CKPT) and not overwrite:
        raise FileExistsError(f'{BASE_CKPT} exists; base training is not bit-reproducible. Pass overwrite=True (CLI: --retrain) to replace it.')
    set_seed(seed)
    train_ds, valid_ds = load_cifar10()
    model = new_model()
    fit(model, make_loader(train_ds, batch_size, True), epochs, lr, make_loader(valid_ds, batch_size * 2))
    os.makedirs(os.path.dirname(BASE_CKPT), exist_ok=True)
    torch.save(model.state_dict(), BASE_CKPT)
    return evaluate(model, make_loader(valid_ds, batch_size * 2))


def class_split(train_ds, valid_ds, forget_classes, n_retain_per_class=1000, seed=0, n_val_per_class=200, batch_size=256):
    """retain_sub/forget_train/retain_train: lists of (x, y) train samples. val_* are held out of
    train data (disjoint from retain_sub, forget_train, retain_train) for tuning; *_valid_dl use the test split."""
    forget_classes = list(forget_classes)
    g = torch.Generator().manual_seed(seed)
    ytr = train_ds.y
    retain_sub, retain_train, forget_train, val_f, val_r = [], [], [], [], []
    for c in range(10):
        idx = (ytr == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)].tolist()
        val, rest = idx[:n_val_per_class], idx[n_val_per_class:]
        pairs = lambda ii: [train_ds[i] for i in ii]
        if c in forget_classes:
            val_f += pairs(val)
            forget_train += pairs(rest)
        else:
            val_r += pairs(val)
            retain_train += pairs(rest)
            retain_sub += pairs(rest[:n_retain_per_class])
    yte = valid_ds.y
    fmask = torch.isin(yte, torch.tensor(forget_classes))
    fv = [valid_ds[i] for i in fmask.nonzero().squeeze(1).tolist()]
    rv = [valid_ds[i] for i in (~fmask).nonzero().squeeze(1).tolist()]
    return {
        'retain_sub': retain_sub, 'retain_train': retain_train, 'forget_train': forget_train,
        'forget_valid_dl': make_loader(fv, batch_size), 'retain_valid_dl': make_loader(rv, batch_size * 2),
        'val_forget_dl': make_loader(val_f, batch_size), 'val_retain_dl': make_loader(val_r, batch_size * 2),
        'forget_classes': forget_classes,
    }


if __name__ == '__main__':
    import sys
    r = train_base_model(overwrite='--retrain' in sys.argv)
    print('base test', r)
    open(os.path.join(ROOT, 'outputs', 'COMMON_READY'), 'w').close()
