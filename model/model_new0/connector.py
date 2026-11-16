"""Working template for a new emulator; copy this folder before editing."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn


class ExampleModel(nn.Module):
    """Small example that preserves map height and width."""

    def __init__(self, in_channels: int, out_channels: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, values):
        return self.network(values)


def create_model(
    *, in_channels: int, out_channels: int, settings: Mapping[str, Any]
) -> nn.Module:
    width = int(settings.get("width", 16))
    if width < 1:
        raise ValueError("width must be positive")
    return ExampleModel(in_channels, out_channels, width)


def model_details(*, model: nn.Module, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "short_description": "Example only; replace with the published emulator",
        "width": int(settings.get("width", 16)),
        "active_benchmark_model": False,
    }


def source_files() -> tuple[str, ...]:
    return ()
