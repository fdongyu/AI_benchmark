"""Compact full-resolution neural operator with event adapters."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.block(x) + self.skip(x), inplace=True)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        self.block = ResidualBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class EventAdapter(nn.Module):
    """Small zero-output pointwise residual head for event adaptation."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        if width <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, width, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, channels, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return prediction
        return prediction + self.net(prediction)


class HoustonPINO(nn.Module):
    """U-Net-like grid operator that predicts a normalized conservative-state increment.

    The architecture accepts arbitrary image sizes. It is intentionally compact so a
    full 256 x 512 sample fits on one A100, while tiling remains supported by the data
    pipeline. Physics is imposed outside the network on dimensional states.
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 3,
        base_channels: int = 24,
        depth: int = 4,
        dropout: float = 0.0,
        adapter_width: int = 8,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        widths = [base_channels * (2**level) for level in range(depth)]
        self.stem = ResidualBlock(in_channels, widths[0], dropout)
        self.down = nn.ModuleList(
            DownBlock(widths[index], widths[index + 1], dropout)
            for index in range(depth - 1)
        )
        self.up = nn.ModuleList(
            UpBlock(widths[index], widths[index - 1], widths[index - 1], dropout)
            for index in range(depth - 1, 0, -1)
        )
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        self.adapter = EventAdapter(out_channels, adapter_width)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W], found {tuple(features.shape)}"
            )
        skips: List[torch.Tensor] = [self.stem(features)]
        x = skips[0]
        for block in self.down:
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = block(x, skip)
        return self.adapter(self.head(x))

    def freeze_backbone(self) -> None:
        """Freeze the operator and retain only the event adapter as trainable."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(True)

    def freeze_adapter(self) -> None:
        """Keep the event adapter neutral while fitting the shared operator."""

        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.adapter.parameters()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
