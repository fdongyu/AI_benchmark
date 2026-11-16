#!/usr/bin/env python3
"""Fourier neural operator used as an RDycore emulator baseline."""

from __future__ import annotations

from torch import nn
from neuralop.models import FNO


class RDycoreFNO(nn.Module):
    """Full-domain 2-D FNO that predicts a normalized one-hour state increment."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        hidden_channels: int = 24,
        modes_height: int = 12,
        modes_width: int = 20,
        layers: int = 4,
        lifting_channels: int = 48,
        projection_channels: int = 48,
        domain_padding: float = 0.0625,
    ) -> None:
        super().__init__()
        self.operator = FNO(
            n_modes=(int(modes_height), int(modes_width)),
            hidden_channels=int(hidden_channels),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            lifting_channels=int(lifting_channels),
            projection_channels=int(projection_channels),
            n_layers=int(layers),
            domain_padding=float(domain_padding),
            domain_padding_mode="one-sided",
            fno_skip="linear",
            factorization=None,
            implementation="factorized",
        )

    def forward(self, x):
        return self.operator(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
