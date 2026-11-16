"""Build the retained small residual U-Net architecture."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from vendor.pino.model import HoustonPINO


def create_model(
    *, in_channels: int, out_channels: int, settings: Mapping[str, Any]
) -> nn.Module:
    model = HoustonPINO(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=int(settings.get("base_channels", 14)),
        depth=int(settings.get("depth", 4)),
        dropout=float(settings.get("dropout", 0.0)),
        adapter_width=int(settings.get("adapter_width", 0)),
    )
    model.freeze_adapter()
    return model


def model_details(*, model: nn.Module, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "short_description": "Small depth-4 residual U-Net",
        "base_channels": int(settings.get("base_channels", 14)),
        "depth": int(settings.get("depth", 4)),
        "dropout": float(settings.get("dropout", 0.0)),
        "adapter_width": int(settings.get("adapter_width", 0)),
        "adapter_training": "absent",
    }


def source_files() -> tuple[str, ...]:
    return ("src/vendor/pino/model.py",)
