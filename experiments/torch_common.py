"""PyTorch implementation of the proposed framework and deep baselines."""

from __future__ import annotations

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sota_baselines import PatchTSTModel, TimesNetModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETERMINISTIC = os.environ.get("EDGE_PHM_DETERMINISTIC", "0") == "1"
torch.backends.cudnn.benchmark = not DETERMINISTIC
torch.backends.cudnn.deterministic = DETERMINISTIC
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ProposedModel(nn.Module):
    def __init__(
        self,
        time_steps: int,
        n_features: int,
        attention: str = "sigmoid",
        mc_dropout: bool = True,
        lstm_units_1: int = 64,
        lstm_units_2: int = 32,
        dropout_rate: float = 0.15,
        attn_temperature: float = 0.5,
        attn_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.mc_dropout = mc_dropout
        self._force_mc_dropout = False
        self.drop_p1 = dropout_rate if mc_dropout else 0.0
        self.drop_p2 = dropout_rate * 0.5 if mc_dropout else 0.0
        self.attn_temperature = attn_temperature
        self.attn_normalize = attn_normalize
        self.lstm1 = nn.LSTM(n_features, lstm_units_1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units_1, lstm_units_2, batch_first=True)
        self.attn_score = nn.Linear(lstm_units_2, 1, bias=True)
        self.dense_hidden = nn.Linear(lstm_units_2, 16, bias=True)
        self.output = nn.Linear(16, 1, bias=True)

    def _attention(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.attn_score(h)
        if self.attention == "sigmoid":
            if self.attn_temperature != 1.0:
                score = score / self.attn_temperature
            alpha = torch.sigmoid(score)
            if self.attn_normalize:
                alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        elif self.attention == "softmax":
            alpha = F.softmax(score, dim=1)
        elif self.attention in {"mean", "none"}:
            alpha = torch.ones_like(score) / score.shape[1]
        else:
            raise ValueError(self.attention)
        return score, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, _ = self.lstm1(x)
        h2, _ = self.lstm2(h1)
        if self.attention == "none":
            c = h2[:, -1, :]
        else:
            _, alpha = self._attention(h2)
            c = (h2 * alpha).sum(dim=1)
        dropout_active = self.training or self._force_mc_dropout
        c = F.dropout(c, p=self.drop_p1, training=dropout_active)
        d = F.relu(self.dense_hidden(c))
        d = F.dropout(d, p=self.drop_p2, training=dropout_active)
        return self.output(d)

    def set_mc_dropout(self, enabled: bool) -> None:
        """Enable stochastic dropout during evaluation-only MC sampling."""
        self._force_mc_dropout = bool(enabled and self.mc_dropout)

    def forward_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1, _ = self.lstm1(x)
        h2, _ = self.lstm2(h1)
        _, alpha = self._attention(h2)
        c = (h2 * alpha).sum(dim=1)
        c = F.dropout(c, p=self.drop_p1, training=False)
        d = F.relu(self.dense_hidden(c))
        d = F.dropout(d, p=self.drop_p2, training=False)
        return self.output(d), alpha

def _softmax_attention_block(h: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    score = linear(h)
    alpha = F.softmax(score, dim=1)
    return (h * alpha).sum(dim=1)


class DeepBaseline(nn.Module):
    def __init__(self, kind: str, time_steps: int, n_features: int) -> None:
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.dense = nn.Linear(32, 16)
        elif kind == "bilstm":
            self.lstm1 = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.attn = nn.Linear(32, 1)
            self.dense = nn.Linear(32, 16)
        elif kind == "gru":
            self.gru1 = nn.GRU(n_features, 64, batch_first=True)
            self.gru2 = nn.GRU(64, 32, batch_first=True)
            self.attn = nn.Linear(32, 1)
            self.dense = nn.Linear(32, 16)
        elif kind == "transformer":
            self.embed = nn.Linear(n_features, 64)
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
            self.norm1 = nn.LayerNorm(64)
            self.ffn = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
            self.norm2 = nn.LayerNorm(64)
            self.dense = nn.Linear(64, 16)
        elif kind == "tcn":
            self.convs = nn.ModuleList()
            self.dilations = [1, 2, 4, 8, 8]
            for i, d in enumerate(self.dilations):
                in_ch = n_features if i == 0 else 32
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, 32, 3, padding="same", dilation=d),
                        nn.ReLU(),
                        nn.Conv1d(32, 32, 3, padding="same", dilation=d),
                    )
                )
            self.dense = nn.Linear(32, 16)
        else:
            raise ValueError(kind)
        self.output = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lstm":
            h1, _ = self.lstm1(x)
            h2, _ = self.lstm2(h1)
            z = h2[:, -1, :]
            z = F.relu(self.dense(z))
        elif self.kind == "bilstm":
            h1, _ = self.lstm1(x)
            h2, _ = self.lstm2(h1)
            z = _softmax_attention_block(h2, self.attn)
            z = F.relu(self.dense(z))
        elif self.kind == "gru":
            h1, _ = self.gru1(x)
            h2, _ = self.gru2(h1)
            z = _softmax_attention_block(h2, self.attn)
            z = F.relu(self.dense(z))
        elif self.kind == "transformer":
            z = F.relu(self.embed(x))
            a, _ = self.attn(z, z, z)
            z = self.norm1(z + a)
            z = self.norm2(z + self.ffn(z))
            z = z.mean(dim=1)
            z = F.relu(self.dense(z))
        elif self.kind == "tcn":
            z = x.transpose(1, 2)
            for i, (conv, dilation) in enumerate(zip(self.convs, self.dilations)):
                y = conv(z)
                if i > 0:
                    y = y + z
                z = F.relu(y)
            z = z.transpose(1, 2).mean(dim=1)
            z = F.relu(self.dense(z))
        else:
            raise ValueError(self.kind)
        return self.output(z)

def build_model(
    time_steps: int,
    n_features: int,
    attention: str = "sigmoid",
    mc_dropout: bool = True,
    dropout_rate: float = 0.15,
    attn_temperature: float = 0.5,
    attn_normalize: bool = True,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    seed: int = 42,
) -> ProposedModel:
    set_seed(seed)
    return ProposedModel(
        time_steps,
        n_features,
        attention=attention,
        mc_dropout=mc_dropout,
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        attn_temperature=attn_temperature,
        attn_normalize=attn_normalize,
    ).to(DEVICE)


def build_deep_baseline(kind: str, time_steps: int, n_features: int, seed: int = 42):
    set_seed(seed)
    if kind == 'patchtst':
        return PatchTSTModel(time_steps, n_features).to(DEVICE)
    if kind == 'timesnet':
        return TimesNetModel(time_steps, n_features).to(DEVICE)
    return DeepBaseline(kind, time_steps, n_features).to(DEVICE)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    use_class_weight: bool = True,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    verbose: int = 0,
    seed: int = 42,
    pos_weight_scale: float = 1.0,
    balanced_sampling: bool = False,
    grad_clip: float = 0.0,
) -> Dict[str, list]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    x_tr = torch.from_numpy(x_train).float().to(DEVICE)
    y_tr = torch.from_numpy(y_train).float().to(DEVICE)
    x_va = torch.from_numpy(x_val).float().to(DEVICE)
    y_va = torch.from_numpy(y_val).float().to(DEVICE)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    # Balanced sampling already equalizes class exposure. Combining it with
    # inverse-frequency BCE weights would correct the same imbalance twice.
    use_weighted_loss = use_class_weight and not balanced_sampling
    pw = (
        max(1.0, neg / max(pos, 1)) * pos_weight_scale
        if use_weighted_loss
        else 1.0
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    if balanced_sampling and pos > 0 and neg > 0:
        w = np.where(y_train == 1, 0.5 / pos, 0.5 / neg)
        sample_weights = torch.from_numpy(w).double().to(DEVICE)
    else:
        sample_weights = None

    n = len(x_train)
    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        if sample_weights is not None:
            perm = torch.multinomial(sample_weights, n, replacement=True)
        else:
            perm = torch.randperm(n, device=DEVICE)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(x_tr[idx]).squeeze(1)
            loss = loss_fn(logits, y_tr[idx])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count
        model.eval()
        with torch.no_grad():
            vlogits = []
            for i in range(0, len(x_va), batch_size):
                vlogits.append(model(x_va[i:i + batch_size]).squeeze(1))
            vlogits = torch.cat(vlogits)
            val_loss = loss_fn(vlogits, y_va).item()
        sched.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    if hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(False)
    xt = torch.from_numpy(x).float().to(DEVICE)
    out = []
    for i in range(0, len(xt), batch_size):
        out.append(torch.sigmoid(model(xt[i:i + batch_size])).squeeze(1))
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: np.ndarray,
    samples: int = 50,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(True)
    xt = torch.from_numpy(x).float().to(DEVICE)
    preds = []
    try:
        for _ in range(samples):
            out = []
            for i in range(0, len(xt), batch_size):
                out.append(torch.sigmoid(model(xt[i:i + batch_size])).squeeze(1))
            preds.append(torch.cat(out).cpu().numpy())
    finally:
        if hasattr(model, "set_mc_dropout"):
            model.set_mc_dropout(False)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


@torch.no_grad()
def predict_attention(
    model: ProposedModel,
    x: np.ndarray,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    xt = torch.from_numpy(x).float().to(DEVICE)
    ps, alphas = [], []
    for i in range(0, len(xt), batch_size):
        logits, alpha = model.forward_attention(xt[i:i + batch_size])
        ps.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        alphas.append(alpha.squeeze(2).cpu().numpy())
    return np.concatenate(ps), np.concatenate(alphas)


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(
    cls,
    path: str,
    time_steps: int,
    n_features: int,
    **kwargs,
):
    model = cls(time_steps, n_features, **kwargs).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
