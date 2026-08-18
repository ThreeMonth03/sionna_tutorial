"""Geometry helpers corresponding to the vectors in TR 38.901 Section 7.1."""

from __future__ import annotations

from typing import Any

from .constants import TWO_PI


def unit_sphere_vector(zenith_rad: Any, azimuth_rad: Any, *, xp: Any) -> Any:
    """Convert spherical zenith/azimuth angles to Cartesian unit vectors."""

    theta = xp.asarray(zenith_rad)
    phi = xp.asarray(azimuth_rad)
    return xp.stack(
        [
            xp.sin(theta) * xp.cos(phi),
            xp.sin(theta) * xp.sin(phi),
            xp.cos(theta),
        ],
        axis=-1,
    )


def spatial_phase(
    positions_m: Any,
    direction_vectors: Any,
    wavelength_m: float,
    *,
    xp: Any,
    complex_dtype: Any,
) -> Any:
    """Compute the phase seen by every antenna at every ray direction.

    ``direction_vectors`` has shape ``[..., 3]`` and positions have shape
    ``[num_antennas, 3]``. The returned shape is ``[..., num_antennas]``.
    """

    positions = xp.asarray(positions_m)
    directions = xp.asarray(direction_vectors)
    dot = xp.einsum("...d,ad->...a", directions, positions)
    return xp.exp(1j * TWO_PI * dot / wavelength_m).astype(complex_dtype, copy=False)


def doppler_phase(
    direction_vectors: Any,
    velocity_mps: Any,
    sample_times_s: Any,
    wavelength_m: float,
    *,
    xp: Any,
    complex_dtype: Any,
) -> Any:
    """Compute ``exp(j 2π ν t)`` for a moving endpoint.

    Directions have shape ``[batch, rays, 3]`` (or a broadcast-compatible
    variant), velocity has shape ``[batch, 3]``, and output has shape
    ``[batch, rays, time]``.
    """

    directions = xp.asarray(direction_vectors)
    velocity = xp.asarray(velocity_mps)
    times = xp.asarray(sample_times_s)
    projected_speed = xp.einsum("brd,bd->br", directions, velocity)
    doppler_hz = projected_speed / wavelength_m
    return xp.exp(1j * TWO_PI * doppler_hz[..., None] * times[None, None, :]).astype(
        complex_dtype,
        copy=False,
    )
