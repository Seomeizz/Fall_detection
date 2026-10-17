"""Temporal Transformer architecture compatible with the 8-class notebook checkpoint."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .labels import FEATURE_DIM, NUM_CLASSES, SEQUENCE_LENGTH


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerClassifier(nn.Module):
    """Input [B, 30, 51]; output logits for the eight mapped action classes."""
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        num_classes: int = NUM_CLASSES,
        window_size: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.positional = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.window_size, self.feature_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}")
        h = self.input_proj(x)
        h = self.positional(h)
        h = self.encoder(h)
        h = self.norm(h)
        h = h.mean(dim=1)
        return self.classifier(h)


def build_transformer_from_checkpoint_metadata(checkpoint: dict) -> TemporalTransformerClassifier:
    """Build the project Transformer using metadata emitted by the notebook."""
    class_names = list(checkpoint.get("class_names", []))
    if class_names:
        from .labels import CLASS_NAMES
        if class_names != CLASS_NAMES:
            raise ValueError(
                "Checkpoint class order does not match this project. "
                f"Checkpoint={class_names}, project={CLASS_NAMES}"
            )
    return TemporalTransformerClassifier(
        feature_dim=int(checkpoint.get("feature_dim", FEATURE_DIM)),
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.15,
        num_classes=len(class_names) if class_names else NUM_CLASSES,
        window_size=int(checkpoint.get("sequence_length", SEQUENCE_LENGTH)),
    )
