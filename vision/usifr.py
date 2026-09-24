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


def make_opt(params, lr, opt_name):
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9)


def run_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def usifr_noise_loss(logits, target, alpha):
    ce = F.cross_entropy(logits, target)
    onehot = F.one_hot(target, logits.shape[1]).float()
    dist = (logits.softmax(1) - onehot).norm(dim=1).mean()
    return ce - alpha * dist


def unsir_noise_loss(logits, target, inputs, lam=0.1):
    return -F.cross_entropy(logits, target) + lam * inputs.pow(2).sum((1, 2, 3)).mean()


def learn_noise(model, forget_class, shape, loss_fn, batch_size=250, epochs=5, steps=8,
                lr=0.1, seed=0, device=None, train_mode=True):
    """Optimise one noise batch against frozen weights; the batch is the noise dataset."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((batch_size,) + tuple(shape), generator=g).to(device).requires_grad_(True)
    target = torch.full((batch_size,), int(forget_class), device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.train(train_mode)
    opt = torch.optim.Adam([noise], lr=lr)
    for _ in range(epochs * steps):
        loss = loss_fn(model(noise), target, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return noise.detach().cpu()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _unlearn(model, retain_sub, forget_class, noise_loss, noise_batches, batch_size, impair_lr,
             repair_lr, repair_epochs, opt_name, noise_epochs, noise_steps, noise_lr, seed,
             device, inplace, noise_train_mode):
    device = device or next(model.parameters()).device
    x, y = as_tensors(retain_sub)
    x, y = x.cpu(), y.cpu()
    if not inplace:
        model = copy.deepcopy(model)
    model.to(device)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    _sync(device)
    t0 = time.time()

    noise = learn_noise(model, forget_class, x.shape[1:], noise_loss, batch_size, noise_epochs,
                        noise_steps, noise_lr, seed, device, noise_train_mode)
    noise = noise.repeat(noise_batches, 1, 1, 1)
    noise_y = torch.full((len(noise),), int(forget_class), dtype=y.dtype)
    impair_dl = DataLoader(TensorDataset(torch.cat([x, noise]), torch.cat([y, noise_y])),
                           batch_size, shuffle=True, generator=g)
    repair_dl = DataLoader(TensorDataset(x, y), batch_size, shuffle=True, generator=g)

    opt = make_opt(model.parameters(), impair_lr, opt_name)
    run_epoch(model, impair_dl, opt, device)
    opt = make_opt(model.parameters(), repair_lr, opt_name)
    for _ in range(repair_epochs):
        run_epoch(model, repair_dl, opt, device)
    _sync(device)
    model.unlearn_time = time.time() - t0
    return model


def unsir_unlearn(model, retain_sub, forget_class, noise_batches=20, batch_size=250,
                  impair_lr=0.02, repair_lr=0.01, opt_name="adam", noise_epochs=5,
                  noise_steps=8, noise_lr=0.1, lam=0.1, seed=0, device=None, inplace=False,
                  noise_train_mode=True):
    loss = lambda out, t, n: unsir_noise_loss(out, t, n, lam)
    return _unlearn(model, retain_sub, forget_class, loss, noise_batches, batch_size, impair_lr,
                    repair_lr, 1, opt_name, noise_epochs, noise_steps, noise_lr, seed, device,
                    inplace, noise_train_mode)


def usifr_unlearn(model, retain_sub, forget_class, alpha=1.0, repair_epochs=2, noise_batches=20,
                  batch_size=250, impair_lr=0.02, repair_lr=0.01, opt_name="adam",
                  noise_epochs=5, noise_steps=8, noise_lr=0.1, seed=0, device=None,
                  inplace=False, noise_train_mode=True):
    loss = lambda out, t, n: usifr_noise_loss(out, t, alpha)
    return _unlearn(model, retain_sub, forget_class, loss, noise_batches, batch_size, impair_lr,
                    repair_lr, repair_epochs, opt_name, noise_epochs, noise_steps, noise_lr,
                    seed, device, inplace, noise_train_mode)
