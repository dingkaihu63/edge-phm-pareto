"""Quantify attention distribution shift on XJTU across train/val/test bearings."""

import os
import numpy as np
import pandas as pd
import torch

from prepare_data import load_xjtu
from torch_common import build_model, predict_attention

data = load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False)
model = build_model(data["x_train"].shape[1], data["x_train"].shape[2], lstm_units_1=96, lstm_units_2=48, dropout_rate=0.10, seed=1)
model.load_state_dict(torch.load(os.path.join("..", "results", "models_torch", "xjtu_full.pt"), map_location="cpu"))
model.eval()

def metrics(x):
    _, attn = predict_attention(model, x)
    attn = np.asarray(attn).reshape(len(x), -1)
    ent = -np.sum(attn * np.log(attn + 1e-12), axis=1)
    energy = np.sum(attn ** 2, axis=1)
    maxw = attn.max(axis=1)
    late = attn[:, -int(attn.shape[1] * 0.25):].mean(axis=1)
    return ent, energy, maxw, late, attn

parts = {}
for split in ["train", "val", "test"]:
    parts[split] = metrics(data[f"x_{split}"])
    for name, arr in zip(["entropy", "energy", "max", "late"], parts[split][:4]):
        print(split, name, round(float(arr.mean()), 4), "+/-", round(float(arr.std()), 4))

# KL between average attention profiles (normalized to probability-like).
avg = {s: parts[s][4].mean(axis=0) for s in ["train", "val", "test"]}
def kl(a, b):
    a = a + 1e-12
    b = b + 1e-12
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum(a * np.log(a / b)))

print("KL train->val", round(kl(avg["train"], avg["val"]), 4))
print("KL train->test", round(kl(avg["train"], avg["test"]), 4))
print("KL val->test", round(kl(avg["val"], avg["test"]), 4))
