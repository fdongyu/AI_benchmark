"""Build the retained U-Net architecture."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from vendor.baselines.unet_model import RDycoreUNet


def create_model(
    *, in_channels: int, out_channels: int, settings: Mapping[str, Any]
) -> nn.Module:
    base_channels = int(settings.get("base_channels", 16))
    if base_channels < 1:
        raise ValueError("base_channels must be positive")
    return RDycoreUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


def model_details(*, model: nn.Module, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "short_description": "Three-level U-Net architecture baseline",
        "base_channels": int(settings.get("base_channels", 16)),
    }


def source_files() -> tuple[str, ...]:
    return ("src/vendor/baselines/unet_model.py",)
