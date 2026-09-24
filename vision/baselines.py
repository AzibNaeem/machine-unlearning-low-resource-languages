import copy
import time
import torch
import torch.nn.functional as F
from vision.common import device, new_model, make_loader, fit


def finetune(model, retain_data, lr=0.01, epochs=5, batch_size=256):
    model = copy.deepcopy(model)
    dl = make_loader(retain_data, batch_size, True)
    opt = torch.optim.Adam(model.parameters(), lr)
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def neggrad(model, forget_data, lr=1e-4, epochs=1, batch_size=256):
    model = copy.deepcopy(model)
    dl = make_loader(forget_data, batch_size, True)
    opt = torch.optim.Adam(model.parameters(), lr)
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = -F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def retrain(retain_data, epochs=40, lr=0.01, batch_size=256, valid_dl=None):
    model = new_model()
    return fit(model, make_loader(retain_data, batch_size, True), epochs, lr, valid_dl, log=False)
