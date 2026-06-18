"""SigLIP image encoder + MiniLM text encoder + trained MLP head classifier backend."""

from __future__ import annotations

import time
from typing import Protocol

import torch
from torch import nn

from .base_model import BaseVisionModel, PredictionResult

IMAGE_EMBED_DIM = 1152
TEXT_EMBED_DIM = 384


class ClassifierHead(nn.Module):
    """Trainable MLP fusing a frozen image embedding and a frozen text embedding."""

    def __init__(
        self,
        image_dim: int = IMAGE_EMBED_DIM,
        text_dim: int = TEXT_EMBED_DIM,
        hidden_dims: tuple[int, int] = (512, 128),
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(image_dim + text_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image_embed, text_embed], dim=-1)
        return self.net(fused).squeeze(-1)
