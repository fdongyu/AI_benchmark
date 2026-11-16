"""Differentiable two-dimensional shallow-water physics for PINO training.

The routines in this module operate on conservative states ordered as
``(h, qx, qy) == (water depth, h*u, h*v)`` with shape ``[B, 3, H, W]``.
Rainfall is supplied in millimetres per hour and converted to SI units inside
the residual.  Spatial derivatives use closed, mask-aware finite-volume faces
for conservative advection.  The pressure and bed-slope terms are combined as
``g*h*grad(h + bed)`` so a lake at rest is preserved discretely, even over
non-flat topography.

These losses are intended to regularize a learned time step.  They are not a
replacement for a shock-capturing production shallow-water solver.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor


ScalarLike = Union[float, int, Tensor]
FieldLike = Union[float, int, Tensor]

MM_PER_HOUR_TO_M_PER_SECOND = 1.0e-3 / 3600.0


def _validate_state(state: Tensor, name: str) -> None:
    if not isinstance(state, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape [B, 3, H, W]; got {tuple(state.shape)}"
        )
    if not state.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _field_like(value: FieldLike, state: Tensor, name: str) -> Tensor:
    """Convert a scalar/spatial field to a broadcast ``[B, H, W]`` tensor."""

    batch, _, height, width = state.shape
    field = torch.as_tensor(value, dtype=state.dtype, device=state.device)

    if field.ndim == 0:
        field = field.reshape(1, 1, 1)
    elif field.ndim == 1:
        if field.shape[0] not in (1, batch):
            raise ValueError(
                f"{name} with one dimension must have length 1 or B={batch}; "
                f"got {tuple(field.shape)}"
            )
        field = field.reshape(field.shape[0], 1, 1)
    elif field.ndim == 2:
        field = field.unsqueeze(0)
    elif field.ndim == 4:
        if field.shape[1] != 1:
            raise ValueError(
                f"{name} with four dimensions must have one channel; "
                f"got {tuple(field.shape)}"
            )
        field = field[:, 0]
    elif field.ndim != 3:
        raise ValueError(
            f"{name} must be scalar, [H,W], [B,H,W], or [B,1,H,W]; "
            f"got {tuple(field.shape)}"
        )

    try:
        return field.expand(batch, height, width)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(field.shape)} cannot broadcast to "
            f"[{batch}, {height}, {width}]"
        ) from exc


def _positive_scalar(value: ScalarLike, reference: Tensor, name: str) -> Tensor:
    scalar = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar; got shape {tuple(scalar.shape)}")
    scalar = scalar.reshape(())
    if scalar.detach().item() <= 0.0:
        raise ValueError(f"{name} must be positive; got {scalar.detach().item()}")
    return scalar


def _domain_mask(domain_mask: Optional[FieldLike], state: Tensor) -> Tensor:
    if domain_mask is None:
        return torch.ones(
            (state.shape[0], state.shape[2], state.shape[3]),
            dtype=torch.bool,
            device=state.device,
        )
    return _field_like(domain_mask, state, "domain_mask") > 0.5


def project_wet_dry(
    state: Tensor,
    domain_mask: Optional[FieldLike] = None,
    *,
    dry_tolerance: float = 1.0e-6,
    max_velocity: Optional[float] = None,
) -> Tensor:
    """Project a conservative state onto nonnegative wet/dry states.

    Negative depths and depths at or below ``dry_tolerance`` become zero.
    Momentum is zeroed in those dry cells and outside ``domain_mask``.  When
    ``max_velocity`` is supplied, momentum magnitude is additionally limited
    to ``h * max_velocity`` to prevent singular near-dry velocities.  The
    operation is differentiable almost everywhere and does not modify its
    input in place.
    """

    _validate_state(state, "state")
    if dry_tolerance < 0.0:
        raise ValueError("dry_tolerance must be nonnegative")
    if max_velocity is not None and max_velocity <= 0.0:
        raise ValueError("max_velocity must be positive when supplied")

    mask = _domain_mask(domain_mask, state)
    h = torch.clamp_min(state[:, 0], 0.0)
    wet = (h > dry_tolerance) & mask
    zero = torch.zeros((), dtype=state.dtype, device=state.device)

    h = torch.where(wet, h, zero)
    qx = torch.where(wet, state[:, 1], zero)
    qy = torch.where(wet, state[:, 2], zero)
    if max_velocity is not None:
        speed = torch.linalg.vector_norm(torch.stack((qx, qy), dim=1), dim=1)
        limit = h * float(max_velocity)
        scale = torch.clamp(
            limit / speed.clamp_min(torch.finfo(state.dtype).eps), max=1.0
        )
        qx = qx * scale
        qy = qy * scale
    return torch.stack((h, qx, qy), dim=1)


def straight_through_wet_dry(
    state: Tensor,
    domain_mask: Optional[FieldLike] = None,
    *,
    dry_tolerance: float = 1.0e-6,
    max_velocity: Optional[float] = None,
) -> Tensor:
    """Apply the hard admissibility projection with identity backward gradient.

    Forward rollouts remain nonnegative and velocity limited, while supervised
    and continuity losses can still teach a currently dry/negative proposal to
    become wet.  This is a straight-through estimator, not a differentiable
    hydrodynamic flux approximation.
    """

    projected = project_wet_dry(
        state,
        domain_mask,
        dry_tolerance=dry_tolerance,
        max_velocity=max_velocity,
    )
    return state + (projected - state).detach()


def rainfall_to_m_per_s(rainfall_mm_h: FieldLike, state: Tensor) -> Tensor:
    """Broadcast rainfall to ``[B,H,W]`` and convert mm/hour to m/second."""

    _validate_state(state, "state")
    return _field_like(rainfall_mm_h, state, "rainfall_mm_h") * (
        MM_PER_HOUR_TO_M_PER_SECOND
    )


def _closed_face_divergence(
    flux_x: Tensor,
    flux_y: Tensor,
    mask: Tensor,
    dx: Tensor,
    dy: Tensor,
    y_axis_sign: float,
) -> Tensor:
    """Finite-volume divergence with zero flux through masked/exterior faces."""

    zero_x = torch.zeros_like(flux_x[..., :1])
    valid_x = mask[..., :-1] & mask[..., 1:]
    inner_x = 0.5 * (flux_x[..., :-1] + flux_x[..., 1:])
    inner_x = inner_x * valid_x.to(inner_x.dtype)
    faces_x = torch.cat((zero_x, inner_x, zero_x), dim=-1)
    divergence_x = (faces_x[..., 1:] - faces_x[..., :-1]) / dx

    zero_y = torch.zeros_like(flux_y[..., :1, :])
    valid_y = mask[..., :-1, :] & mask[..., 1:, :]
    inner_y = 0.5 * (flux_y[..., :-1, :] + flux_y[..., 1:, :])
    inner_y = inner_y * valid_y.to(inner_y.dtype)
    faces_y = torch.cat((zero_y, inner_y, zero_y), dim=-2)
    divergence_y = (
        y_axis_sign * (faces_y[..., 1:, :] - faces_y[..., :-1, :]) / dy
    )

    return (divergence_x + divergence_y) * mask.to(flux_x.dtype)


def _masked_gradient(
    field: Tensor,
    mask: Tensor,
    spacing: Tensor,
    dimension: int,
    axis_sign: float = 1.0,
) -> Tensor:
    """Centered interior and one-sided boundary gradient without crossing masks."""

    if dimension == -1:
        false_edge = torch.zeros_like(mask[..., :1])
        right = torch.cat((field[..., 1:], field[..., -1:]), dim=-1)
        left = torch.cat((field[..., :1], field[..., :-1]), dim=-1)
        right_valid = torch.cat((mask[..., 1:], false_edge), dim=-1)
        left_valid = torch.cat((false_edge, mask[..., :-1]), dim=-1)
    elif dimension == -2:
        false_edge = torch.zeros_like(mask[..., :1, :])
        right = torch.cat((field[..., 1:, :], field[..., -1:, :]), dim=-2)
        left = torch.cat((field[..., :1, :], field[..., :-1, :]), dim=-2)
        right_valid = torch.cat((mask[..., 1:, :], false_edge), dim=-2)
        left_valid = torch.cat((false_edge, mask[..., :-1, :]), dim=-2)
    else:
        raise ValueError("dimension must be -1 (x) or -2 (y)")

    both = right_valid & left_valid
    centered = (right - left) / (2.0 * spacing)
    forward = (right - field) / spacing
    backward = (field - left) / spacing
    gradient = torch.where(
        both,
        centered,
        torch.where(right_valid, forward, torch.where(left_valid, backward, 0.0)),
    )
    return axis_sign * gradient * mask.to(field.dtype)


def swe_residual(
    state_t: Tensor,
    state_tp1: Tensor,
    rainfall_mm_h: FieldLike,
    bed: FieldLike,
    manning: FieldLike,
    domain_mask: Optional[FieldLike],
    dt: ScalarLike,
    dx: ScalarLike,
    dy: ScalarLike,
    *,
    gravity: float = 9.81,
    dry_tolerance: float = 1.0e-6,
    project_states: bool = True,
    y_axis_sign: float = 1.0,
) -> Tensor:
    """Return local 2-D SWE residuals with shape ``[B,3,H,W]``.

    The three channels are water mass, x momentum, and y momentum.  Rainfall
    enters mass only, corresponding to added water with zero horizontal
    momentum.  Momentum advection is conservative.  Combining pressure and
    topographic forcing through the free-surface gradient makes a stationary
    constant free surface an exact discrete equilibrium.

    Exterior and mask-interface faces are treated as closed.  Open-ocean and
    river boundary conditions should be imposed separately by the training
    objective.
    """

    _validate_state(state_t, "state_t")
    _validate_state(state_tp1, "state_tp1")
    if state_t.shape != state_tp1.shape:
        raise ValueError(
            "state_t and state_tp1 must have identical shapes; got "
            f"{tuple(state_t.shape)} and {tuple(state_tp1.shape)}"
        )
    if state_t.device != state_tp1.device or state_t.dtype != state_tp1.dtype:
        raise ValueError("state_t and state_tp1 must share dtype and device")
    if gravity <= 0.0:
        raise ValueError("gravity must be positive")
    if dry_tolerance <= 0.0:
        raise ValueError("dry_tolerance must be positive")
    if y_axis_sign not in (-1.0, 1.0):
        raise ValueError("y_axis_sign must be -1 or +1")

    mask = _domain_mask(domain_mask, state_t)
    dt_tensor = _positive_scalar(dt, state_t, "dt")
    dx_tensor = _positive_scalar(dx, state_t, "dx")
    dy_tensor = _positive_scalar(dy, state_t, "dy")

    if project_states:
        first = project_wet_dry(state_t, mask, dry_tolerance=dry_tolerance)
        second = project_wet_dry(state_tp1, mask, dry_tolerance=dry_tolerance)
    else:
        first = state_t * mask[:, None].to(state_t.dtype)
        second = state_tp1 * mask[:, None].to(state_tp1.dtype)

    midpoint = 0.5 * (first + second)
    h, qx, qy = midpoint.unbind(dim=1)
    wet = (h > dry_tolerance) & mask
    h_safe = torch.clamp_min(h, dry_tolerance)
    zero = torch.zeros((), dtype=state_t.dtype, device=state_t.device)
    u = torch.where(wet, qx / h_safe, zero)
    v = torch.where(wet, qy / h_safe, zero)

    rain = rainfall_to_m_per_s(rainfall_mm_h, state_t)
    bed_field = _field_like(bed, state_t, "bed")
    manning_field = _field_like(manning, state_t, "manning")
    if torch.any(manning_field.detach() < 0.0):
        raise ValueError("manning must be nonnegative")

    mass_divergence = _closed_face_divergence(
        qx, qy, mask, dx_tensor, dy_tensor, y_axis_sign
    )
    momentum_x_advection = _closed_face_divergence(
        qx * u, qx * v, mask, dx_tensor, dy_tensor, y_axis_sign
    )
    momentum_y_advection = _closed_face_divergence(
        qy * u, qy * v, mask, dx_tensor, dy_tensor, y_axis_sign
    )

    free_surface = h + bed_field
    surface_gradient_x = _masked_gradient(free_surface, mask, dx_tensor, dimension=-1)
    surface_gradient_y = _masked_gradient(
        free_surface,
        mask,
        dy_tensor,
        dimension=-2,
        axis_sign=y_axis_sign,
    )

    # ``linalg.vector_norm`` defines a zero gradient at zero discharge.  A
    # direct ``sqrt(qx**2 + qy**2)`` produces NaN gradients for still water.
    discharge_magnitude = torch.linalg.vector_norm(torch.stack((qx, qy), dim=1), dim=1)
    friction_factor = (
        gravity * manning_field.square() * discharge_magnitude / h_safe.pow(7.0 / 3.0)
    )
    friction_x = torch.where(wet, friction_factor * qx, zero)
    friction_y = torch.where(wet, friction_factor * qy, zero)

    mask_float = mask.to(state_t.dtype)
    residual_h = (
        (second[:, 0] - first[:, 0]) / dt_tensor + mass_divergence - rain
    ) * mask_float
    residual_qx = (
        (second[:, 1] - first[:, 1]) / dt_tensor
        + momentum_x_advection
        + gravity * h * surface_gradient_x
        + friction_x
    ) * mask_float
    residual_qy = (
        (second[:, 2] - first[:, 2]) / dt_tensor
        + momentum_y_advection
        + gravity * h * surface_gradient_y
        + friction_y
    ) * mask_float

    return torch.stack((residual_h, residual_qx, residual_qy), dim=1)


def _masked_mean_square(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values.square() * weights).sum() / weights.sum().clamp_min(1.0)


def physics_loss(
    state_t: Tensor,
    state_tp1: Tensor,
    rainfall_mm_h: FieldLike,
    bed: FieldLike,
    manning: FieldLike,
    domain_mask: Optional[FieldLike],
    dt: ScalarLike,
    dx: ScalarLike,
    dy: ScalarLike,
    *,
    evaluation_mask: Optional[FieldLike] = None,
    momentum_evaluation_mask: Optional[FieldLike] = None,
    gravity: float = 9.81,
    dry_tolerance: float = 1.0e-6,
    y_axis_sign: float = 1.0,
    component_weights: Sequence[float] = (1.0, 1.0, 1.0),
    component_scales: Sequence[float] = (1.0, 1.0, 1.0),
    positivity_weight: float = 1.0,
    dry_momentum_weight: float = 1.0,
    wet_only_momentum: bool = True,
    project_states: bool = True,
    return_components: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Compute a mask-aware SWE regularization loss.

    ``component_scales`` should normally be set from training-set statistics so
    mass and momentum residuals have comparable numerical magnitudes.  When
    ``return_components`` is true, a dictionary containing ``loss`` and each
    unweighted diagnostic component is returned.
    """

    if len(component_weights) != 3 or len(component_scales) != 3:
        raise ValueError("component_weights and component_scales must have length 3")
    if any(weight < 0.0 for weight in component_weights):
        raise ValueError("component_weights must be nonnegative")
    if any(scale <= 0.0 for scale in component_scales):
        raise ValueError("component_scales must be positive")
    if positivity_weight < 0.0 or dry_momentum_weight < 0.0:
        raise ValueError("penalty weights must be nonnegative")

    residual = swe_residual(
        state_t,
        state_tp1,
        rainfall_mm_h,
        bed,
        manning,
        domain_mask,
        dt,
        dx,
        dy,
        gravity=gravity,
        dry_tolerance=dry_tolerance,
        project_states=project_states,
        y_axis_sign=y_axis_sign,
    )
    # ``domain_mask`` controls the flux stencil, while ``evaluation_mask`` can
    # restrict the objective to a tile interior.  Keeping those concepts
    # separate lets halo cells supply valid derivatives without being counted
    # repeatedly in the loss.
    mask = _domain_mask(domain_mask, state_t)
    if evaluation_mask is not None:
        mask = mask & _domain_mask(evaluation_mask, state_t)
    momentum_mask = mask
    if momentum_evaluation_mask is not None:
        momentum_mask = momentum_mask & _domain_mask(
            momentum_evaluation_mask, state_t
        )
    if wet_only_momentum:
        midpoint_depth = 0.5 * (
            torch.clamp_min(state_t[:, 0], 0.0) + torch.clamp_min(state_tp1[:, 0], 0.0)
        )
        momentum_mask = momentum_mask & (midpoint_depth > dry_tolerance)

    mass = _masked_mean_square(residual[:, 0] / component_scales[0], mask)
    momentum_x = _masked_mean_square(
        residual[:, 1] / component_scales[1], momentum_mask
    )
    momentum_y = _masked_mean_square(
        residual[:, 2] / component_scales[2], momentum_mask
    )

    h_next = state_tp1[:, 0]
    positivity = _masked_mean_square(torch.relu(-h_next), mask)
    dry_next = mask & (h_next <= dry_tolerance)
    dry_momentum = (
        (state_tp1[:, 1].square() + state_tp1[:, 2].square())
        * dry_next.to(state_tp1.dtype)
    ).sum() / dry_next.to(state_tp1.dtype).sum().clamp_min(1.0)

    total = (
        component_weights[0] * mass
        + component_weights[1] * momentum_x
        + component_weights[2] * momentum_y
        + positivity_weight * positivity
        + dry_momentum_weight * dry_momentum
    )
    if not return_components:
        return total
    return {
        "loss": total,
        "mass": mass,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "positivity": positivity,
        "dry_momentum": dry_momentum,
    }


def global_mass_balance(
    state_t: Tensor,
    state_tp1: Tensor,
    rainfall_mm_h: FieldLike,
    domain_mask: Optional[FieldLike],
    dt: ScalarLike,
    dx: ScalarLike,
    dy: ScalarLike,
    *,
    net_boundary_outflow_m3_s: Optional[FieldLike] = None,
    dry_tolerance: float = 1.0e-6,
    project_states: bool = True,
) -> Dict[str, Tensor]:
    """Return per-sample water-volume closure diagnostics.

    ``net_boundary_outflow_m3_s`` is optional, positive out of the domain, and
    may be scalar or length ``B``.  The closure convention is

    ``storage change - rainfall volume + boundary outflow volume = 0``.
    """

    _validate_state(state_t, "state_t")
    _validate_state(state_tp1, "state_tp1")
    if state_t.shape != state_tp1.shape:
        raise ValueError("state_t and state_tp1 must have identical shapes")

    mask = _domain_mask(domain_mask, state_t)
    dt_tensor = _positive_scalar(dt, state_t, "dt")
    dx_tensor = _positive_scalar(dx, state_t, "dx")
    dy_tensor = _positive_scalar(dy, state_t, "dy")
    area = dx_tensor * dy_tensor

    if project_states:
        first = project_wet_dry(state_t, mask, dry_tolerance=dry_tolerance)
        second = project_wet_dry(state_tp1, mask, dry_tolerance=dry_tolerance)
    else:
        first = state_t
        second = state_tp1

    mask_float = mask.to(state_t.dtype)
    initial_volume = (first[:, 0] * mask_float).sum(dim=(-2, -1)) * area
    final_volume = (second[:, 0] * mask_float).sum(dim=(-2, -1)) * area
    storage_change = final_volume - initial_volume

    rain = rainfall_to_m_per_s(rainfall_mm_h, state_t)
    rainfall_volume = (rain * mask_float).sum(dim=(-2, -1)) * area * dt_tensor

    if net_boundary_outflow_m3_s is None:
        boundary_rate = torch.zeros_like(initial_volume)
    else:
        boundary_rate = torch.as_tensor(
            net_boundary_outflow_m3_s,
            dtype=state_t.dtype,
            device=state_t.device,
        )
        if boundary_rate.ndim == 0:
            boundary_rate = boundary_rate.expand_as(initial_volume)
        elif boundary_rate.ndim == 1 and boundary_rate.shape[0] in (
            1,
            state_t.shape[0],
        ):
            boundary_rate = boundary_rate.expand_as(initial_volume)
        else:
            raise ValueError("net_boundary_outflow_m3_s must be scalar or length B")
    boundary_volume = boundary_rate * dt_tensor
    closure = storage_change - rainfall_volume + boundary_volume
    scale = torch.maximum(
        storage_change.abs(),
        torch.maximum(rainfall_volume.abs(), boundary_volume.abs()),
    ).clamp_min(torch.finfo(state_t.dtype).eps)

    return {
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "storage_change_m3": storage_change,
        "rainfall_volume_m3": rainfall_volume,
        "boundary_outflow_volume_m3": boundary_volume,
        "closure_error_m3": closure,
        "relative_closure_error": closure / scale,
    }


# A descriptive alias for callers that prefer the full equation name.
shallow_water_residual = swe_residual
