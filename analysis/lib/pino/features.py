"""Canonical forcing/static feature construction and state normalization."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def _stat(
    normalization: Mapping[str, Any], key: str, field: str, reference: torch.Tensor
) -> torch.Tensor:
    values = normalization[key][field]
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device).view(
        1, -1, 1, 1
    )


def normalize_state(
    state: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    mean = _stat(normalization, "state", "mean", state)
    std = _stat(normalization, "state", "std", state).clamp_min(1.0e-8)
    return (state - mean) / std


def normalize_delta(
    delta: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    mean = _stat(normalization, "state_delta", "mean", delta)
    std = _stat(normalization, "state_delta", "std", delta).clamp_min(1.0e-8)
    return (delta - mean) / std


def denormalize_delta(
    delta: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    mean = _stat(normalization, "state_delta", "mean", delta)
    std = _stat(normalization, "state_delta", "std", delta).clamp_min(1.0e-8)
    return delta * std + mean


def coordinate_fields(
    reference: torch.Tensor,
    tile_origin: Optional[torch.Tensor] = None,
    global_shape: Optional[torch.Tensor] = None,
    halo: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized global coordinates, including correct tile offsets."""

    batch, _, height, width = reference.shape
    if tile_origin is None and global_shape is None:
        y = torch.linspace(
            1.0, -1.0, height, dtype=reference.dtype, device=reference.device
        )
        x = torch.linspace(
            -1.0, 1.0, width, dtype=reference.dtype, device=reference.device
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return (
            xx.view(1, 1, height, width).expand(batch, -1, -1, -1),
            yy.view(1, 1, height, width).expand(batch, -1, -1, -1),
        )
    if tile_origin is None or global_shape is None:
        raise ValueError("tile_origin and global_shape must be supplied together")
    origins = tile_origin.to(device=reference.device, dtype=reference.dtype).reshape(
        batch, 2
    )
    shapes = global_shape.to(device=reference.device, dtype=reference.dtype).reshape(
        batch, 2
    )
    if halo is None:
        halos = torch.zeros(batch, device=reference.device, dtype=reference.dtype)
    else:
        halos = halo.to(device=reference.device, dtype=reference.dtype).reshape(batch)
    rows = torch.arange(height, dtype=reference.dtype, device=reference.device)
    columns = torch.arange(width, dtype=reference.dtype, device=reference.device)
    global_y = rows.view(1, height) + origins[:, 0:1] - halos[:, None]
    global_x = columns.view(1, width) + origins[:, 1:2] - halos[:, None]
    yy = 1.0 - 2.0 * global_y / (shapes[:, 0:1] - 1.0).clamp_min(1.0)
    xx = -1.0 + 2.0 * global_x / (shapes[:, 1:2] - 1.0).clamp_min(1.0)
    return (
        xx[:, None, None, :].expand(batch, 1, height, width),
        yy[:, None, :, None].expand(batch, 1, height, width),
    )


def make_features(
    current_state: torch.Tensor,
    rainfall_mm_h: torch.Tensor,
    ocean_height_m: torch.Tensor,
    bed_elevation: torch.Tensor,
    domain_mask: torch.Tensor,
    ocean_boundary_mask: torch.Tensor,
    normalization: Mapping[str, Any],
    manning: Optional[torch.Tensor] = None,
    tile_origin: Optional[torch.Tensor] = None,
    global_shape: Optional[torch.Tensor] = None,
    halo: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the 11-channel operator input in one documented order.

    Channels are normalized h/hu/hv, log rainfall, ocean stage, bed, domain,
    ocean-boundary mask, Manning n/0.05, normalized x, and normalized y.
    """

    if current_state.ndim != 4 or current_state.shape[1] != 3:
        raise ValueError("current_state must have shape [B,3,H,W]")
    batch = current_state.shape[0]
    domain = domain_mask.to(dtype=current_state.dtype)
    state = normalize_state(current_state, normalization) * domain

    rain_mean = _stat(normalization, "log1p_rainfall", "mean", current_state)
    rain_std = _stat(normalization, "log1p_rainfall", "std", current_state).clamp_min(
        1.0e-8
    )
    rain = (torch.log1p(rainfall_mm_h.clamp_min(0.0)) - rain_mean) / rain_std

    ocean_mean = _stat(normalization, "ocean_height", "mean", current_state)
    ocean_std = _stat(normalization, "ocean_height", "std", current_state).clamp_min(
        1.0e-8
    )
    ocean = ocean_height_m.reshape(batch, 1, 1, 1)
    ocean = ((ocean - ocean_mean) / ocean_std).expand_as(domain)

    bed_mean = _stat(normalization, "bed_elevation", "mean", current_state)
    bed_std = _stat(normalization, "bed_elevation", "std", current_state).clamp_min(
        1.0e-8
    )
    bed = (bed_elevation - bed_mean) / bed_std

    if manning is None:
        manning = torch.full_like(domain, 0.015)
    manning_feature = manning / 0.05
    xx, yy = coordinate_fields(
        current_state,
        tile_origin=tile_origin,
        global_shape=global_shape,
        halo=halo,
    )
    return torch.cat(
        (
            state,
            rain * domain,
            ocean * domain,
            bed * domain,
            domain,
            ocean_boundary_mask.to(current_state.dtype),
            manning_feature * domain,
            xx * domain,
            yy * domain,
        ),
        dim=1,
    )
