"""Build the retained Fourier Neural Operator architecture."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from vendor.baselines.fno_model import RDycoreFNO


DEFAULTS = {
    "hidden_channels": 24,
    "modes_height": 12,
    "modes_width": 20,
    "layers": 4,
    "lifting_channels": 48,
    "projection_channels": 48,
    "domain_padding": 0.0625,
}


def _resolved_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    values = {**DEFAULTS, **dict(settings)}
    for name in (
        "hidden_channels",
        "modes_height",
        "modes_width",
        "layers",
        "lifting_channels",
        "projection_channels",
    ):
        values[name] = int(values[name])
        if values[name] < 1:
            raise ValueError(f"{name} must be positive")
    values["domain_padding"] = float(values["domain_padding"])
    if values["domain_padding"] < 0.0:
        raise ValueError("domain_padding must be non-negative")
    return values


def create_model(
    *, in_channels: int, out_channels: int, settings: Mapping[str, Any]
) -> nn.Module:
    return RDycoreFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        **_resolved_settings(settings),
    )


def model_details(*, model: nn.Module, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "short_description": "Four-layer full-domain Fourier Neural Operator",
        **_resolved_settings(settings),
    }


def source_files() -> tuple[str, ...]:
    return ("src/vendor/baselines/fno_model.py",)
