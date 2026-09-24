import copy
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def as_tensors(data):
    if isinstance(data, (tuple, list)) and len(data) == 2 and torch.is_tensor(data[0]):
        return data[0], torch.as_tensor(data[1])
    if hasattr(data, "tensors"):
        return data.tensors[0], data.tensors[1]
    xs, ys = zip(*[data[i] for i in range(len(data))])
    return torch.stack(xs), torch.as_tensor(ys)


def run_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def dnf_unlearn(model, retain_sub, forget_class, noise_n=5120, batch_size=250,
                impair_lr=0.02, repair_lr=0.01, impair_epochs=1, repair_epochs=1,
                opt_name="adam", noise_std=1.0, seed=0, device=None, inplace=False):
    device = device or next(model.parameters()).device
    x, y = as_tensors(retain_sub)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((noise_n,) + tuple(x.shape[1:]), generator=g) * noise_std
    noise_y = torch.full((noise_n,), int(forget_class), dtype=y.dtype)
    impair_ds = TensorDataset(torch.cat([x.cpu(), noise]), torch.cat([y.cpu(), noise_y]))
    repair_ds = TensorDataset(x.cpu(), y.cpu())

    def make_opt(lr):
        if opt_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if not inplace:
        model = copy.deepcopy(model)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    impair_dl = DataLoader(impair_ds, batch_size, shuffle=True, generator=g)
    repair_dl = DataLoader(repair_ds, batch_size, shuffle=True, generator=g)
    opt = make_opt(impair_lr)
    for _ in range(impair_epochs):
        run_epoch(model, impair_dl, opt, device)
    opt = make_opt(repair_lr)
    for _ in range(repair_epochs):
        run_epoch(model, repair_dl, opt, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.unlearn_time = time.time() - t0
    return model
