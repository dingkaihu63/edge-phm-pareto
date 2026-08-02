"""Compact PatchTST and TimesNet baselines for the unified benchmark protocol."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        time_steps: int,
        n_features: int,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.register_buffer("pos", self._positional(d_model, max_len=512))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def _positional(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        patches = x.unfold(1, self.patch_len, self.stride)
        b, pn, f, _ = patches.shape
        emb = self.patch_embed(patches)
        seq = emb.reshape(b, pn * f, -1)
        seq = seq + self.pos[:, : seq.shape[1], :]
        out = self.encoder(seq)
        z = self.norm(out).mean(dim=1)
        return self.head(z)


class TimesNetModel(nn.Module):
    def __init__(
        self,
        time_steps: int,
        n_features: int,
        periods=(4, 8),
        hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.periods = periods
        self.hidden = hidden
        self.convs = nn.ModuleList()
        for _ in periods:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, hidden, kernel_size=(3, 3), padding=(1, 1), bias=False),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1), bias=False),
                    nn.GELU(),
                )
            )
        self.period_weight = nn.Parameter(torch.zeros(len(periods)))
        self.norm = nn.LayerNorm(n_features * hidden)
        self.head1 = nn.Linear(n_features * hidden, 16)
        self.head = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        xt = x.permute(0, 2, 1)
        xf = torch.fft.rfft(xt, dim=2, norm="ortho")
        amp = xf.abs().mean(dim=1)
        outs = []
        for i, p in enumerate(self.periods):
            n = (t + p - 1) // p
            pad = n * p - t
            xp = F.pad(xt, (0, pad)) if pad else xt
            folded = xp.reshape(b, f, n, p).unsqueeze(1)
            y = self.convs[i](folded.reshape(b, 1, f * n, p))
            y = y.reshape(b, self.hidden, f, n, p).permute(0, 2, 1, 3, 4)
            y = y.reshape(b, f, self.hidden, n * p)[:, :, :, :t]
            outs.append(y.mean(dim=3))
        z = torch.stack(outs, dim=0)
        w = F.softmax(self.period_weight, dim=0)
        z = (z * w.view(-1, 1, 1, 1)).sum(dim=0)
        z = z.reshape(b, f * self.hidden)
        z = self.norm(z)
        z = F.relu(self.head1(self.dropout(z)))
        return self.head(z)