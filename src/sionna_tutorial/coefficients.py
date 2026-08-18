"""Deterministic construction of the small-scale CDL propagation channel.

SYSTEM LOCATION
---------------

Tx antenna array -> [THIS MODULE: wireless propagation h(t, tau)] -> Rx array

The module does not process information bits and does not implement a receiver.
It converts a standardized profile plus sampled physical rays into

```text
coefficients [batch, rx_antenna, tx_antenna, delayed_path, time].
```

METHOD
------

For every ray, the implementation follows the structure of TR 38.901 Eq.
7.5-28 and combines:

```text
receive element field
x 2x2 polarization coupling
x transmit element field
x receive array-position phase
x transmit array-position phase
x Doppler time phase
x ray amplitude
```

Twenty rays are summed to form one delayed MIMO path/cluster. The code omits
global coordinate-system rotations and path loss; those belong to a larger
scenario model rather than this small-scale channel tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .arrays import AntennaArray, element_field_components
from .backend import Backend
from .constants import RAYS_PER_CLUSTER, SPEED_OF_LIGHT
from .geometry import doppler_phase, spatial_phase, unit_sphere_vector
from .profiles import CDLProfile
from .rays import CDLRandomState

MovingEnd = Literal["tx", "rx"]


@dataclass(frozen=True)
class ChannelCoefficients:
    """Sparse MIMO channel impulse response and path delays.

    `coefficients` has shape `[batch, rx_ant, tx_ant, path, time]`.
    `delays_s` has shape `[path]`.

    This is the generated channel itself. Applying it to transmitted data and
    estimating it at a receiver are later, separate operations.
    """

    coefficients: object
    delays_s: np.ndarray


def _polarization_matrix(phases: object, xpr_linear: float, *, xp: object, cdtype: object) -> object:
    """Build one 2x2 Jones/polarization matrix for every sampled ray."""

    co_theta = xp.exp(1j * phases[..., 0])
    cross_theta_phi = xp.exp(1j * phases[..., 1]) / xp.sqrt(xpr_linear)
    cross_phi_theta = xp.exp(1j * phases[..., 2]) / xp.sqrt(xpr_linear)
    co_phi = xp.exp(1j * phases[..., 3])
    row_theta = xp.stack([co_theta, cross_theta_phi], axis=-1)
    row_phi = xp.stack([cross_phi_theta, co_phi], axis=-1)
    return xp.stack([row_theta, row_phi], axis=-2).astype(cdtype, copy=False)


def _normalize_velocity(velocity_mps: np.ndarray, batch_size: int) -> np.ndarray:
    """Convert one shared velocity or one velocity per batch into `[batch, 3]`."""

    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if velocity.shape == (3,):
        return np.broadcast_to(velocity, (batch_size, 3)).copy()
    if velocity.shape == (batch_size, 3):
        return velocity
    raise ValueError("velocity_mps must have shape [3] or [batch, 3]")


def _cluster_coefficients(
    *,
    cluster_index: int,
    profile: CDLProfile,
    random_state: CDLRandomState,
    tx_array: AntennaArray,
    rx_array: AntennaArray,
    velocity_mps: object,
    sample_times_s: object,
    wavelength_m: float,
    backend: Backend,
    real_dtype: object,
    complex_dtype: object,
    moving_end: MovingEnd,
) -> object:
    """Sum 20 rays into one delayed MIMO path for one CDL cluster.

    Returns `[batch, rx_antenna, tx_antenna, time]`.
    """

    xp = backend.xp
    angles = random_state.angles

    # ------------------------------------------------------------------
    # Step 1: select this cluster's per-ray random physical state.
    # Every angle tensor below has shape [batch, ray].
    # ------------------------------------------------------------------
    aod = backend.asarray(angles.aod_rad[:, cluster_index], dtype=real_dtype)
    aoa = backend.asarray(angles.aoa_rad[:, cluster_index], dtype=real_dtype)
    zod = backend.asarray(angles.zod_rad[:, cluster_index], dtype=real_dtype)
    zoa = backend.asarray(angles.zoa_rad[:, cluster_index], dtype=real_dtype)
    phases = backend.asarray(
        random_state.polarization_phases_rad[:, cluster_index],
        dtype=real_dtype,
    )

    # ------------------------------------------------------------------
    # Step 2: convert departure/arrival angles into 3D propagation vectors.
    # Shape: [batch, ray, 3].
    # ------------------------------------------------------------------
    tx_direction = unit_sphere_vector(zod, aod, xp=xp)
    rx_direction = unit_sphere_vector(zoa, aoa, xp=xp)

    # ------------------------------------------------------------------
    # Step 3: evaluate Tx/Rx antenna element fields and polarization.
    # tx_field: [batch, ray, tx_port, polarization_component]
    # rx_field: [batch, ray, rx_port, polarization_component]
    # ------------------------------------------------------------------
    tx_field = element_field_components(
        zod,
        aod,
        tx_array.slant_angles_rad,
        pattern=tx_array.pattern,
        xp=xp,
    ).astype(complex_dtype, copy=False)
    rx_field = element_field_components(
        zoa,
        aoa,
        rx_array.slant_angles_rad,
        pattern=rx_array.pattern,
        xp=xp,
    ).astype(complex_dtype, copy=False)

    polarization = _polarization_matrix(
        phases,
        profile.xpr_linear,
        xp=xp,
        cdtype=complex_dtype,
    )

    # Contract F_rx^T P F_tx for every ray and antenna pair.
    # field_coupling: [batch, ray, rx_port, tx_port].
    field_coupling = xp.einsum(
        "brui,brij,brvj->bruv",
        rx_field,
        polarization,
        tx_field,
        optimize=True,
    )

    # ------------------------------------------------------------------
    # Step 4: convert physical antenna positions into complex spatial phase.
    # A plane wave reaches different array elements at different phases.
    # ------------------------------------------------------------------
    rx_spatial = spatial_phase(
        rx_array.positions_m,
        rx_direction,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    tx_spatial = spatial_phase(
        tx_array.positions_m,
        tx_direction,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    spatial = rx_spatial[..., :, None] * tx_spatial[..., None, :]

    # ------------------------------------------------------------------
    # Step 5: generate each ray's time evolution from mobility/Doppler.
    # time_phase: [batch, ray, time].
    # ------------------------------------------------------------------
    moving_direction = rx_direction if moving_end == "rx" else tx_direction
    time_phase = doppler_phase(
        moving_direction,
        velocity_mps,
        sample_times_s,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )

    # ------------------------------------------------------------------
    # Step 6: multiply all per-ray factors and sum the 20-ray axis.
    # `bruv,brt->buvt` removes `r` and returns one delayed MIMO path.
    # ------------------------------------------------------------------
    ray_amplitude = xp.sqrt(profile.powers_linear[cluster_index] / RAYS_PER_CLUSTER)
    return ray_amplitude * xp.einsum(
        "bruv,brt->buvt",
        field_coupling * spatial,
        time_phase,
        optimize=True,
    )


def _los_coefficient(
    *,
    profile: CDLProfile,
    tx_array: AntennaArray,
    rx_array: AntennaArray,
    velocity_mps: object,
    sample_times_s: object,
    wavelength_m: float,
    backend: Backend,
    real_dtype: object,
    complex_dtype: object,
    moving_end: MovingEnd,
) -> object:
    """Generate the deterministic specular LOS MIMO component for CDL-D/E."""

    if profile.los_angles_rad is None:
        raise ValueError("LOS coefficient requested for a non-LOS profile")

    xp = backend.xp
    aod, aoa, zod, zoa = profile.los_angles_rad
    batch_size = int(velocity_mps.shape[0])
    aod_b = xp.full((batch_size, 1), aod, dtype=real_dtype)
    aoa_b = xp.full((batch_size, 1), aoa, dtype=real_dtype)
    zod_b = xp.full((batch_size, 1), zod, dtype=real_dtype)
    zoa_b = xp.full((batch_size, 1), zoa, dtype=real_dtype)

    tx_direction = unit_sphere_vector(zod_b, aod_b, xp=xp)
    rx_direction = unit_sphere_vector(zoa_b, aoa_b, xp=xp)
    tx_field = element_field_components(
        zod_b,
        aod_b,
        tx_array.slant_angles_rad,
        pattern=tx_array.pattern,
        xp=xp,
    ).astype(complex_dtype, copy=False)
    rx_field = element_field_components(
        zoa_b,
        aoa_b,
        rx_array.slant_angles_rad,
        pattern=rx_array.pattern,
        xp=xp,
    ).astype(complex_dtype, copy=False)

    # The specular path uses the deterministic LOS polarization matrix.
    los_pol = xp.asarray(
        [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]],
        dtype=complex_dtype,
    )
    field_coupling = xp.einsum(
        "brui,ij,brvj->bruv",
        rx_field,
        los_pol,
        tx_field,
        optimize=True,
    )
    rx_spatial = spatial_phase(
        rx_array.positions_m,
        rx_direction,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    tx_spatial = spatial_phase(
        tx_array.positions_m,
        tx_direction,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    spatial = rx_spatial[..., :, None] * tx_spatial[..., None, :]
    moving_direction = rx_direction if moving_end == "rx" else tx_direction
    time_phase = doppler_phase(
        moving_direction,
        velocity_mps,
        sample_times_s,
        wavelength_m,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    return xp.einsum(
        "bruv,brt->buvt",
        field_coupling * spatial,
        time_phase,
        optimize=True,
    )


def generate_cdl_coefficients(
    profile: CDLProfile,
    random_state: CDLRandomState,
    tx_array: AntennaArray,
    rx_array: AntennaArray,
    *,
    carrier_frequency_hz: float,
    velocity_mps: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    num_time_steps: int = 1,
    sampling_frequency_hz: float = 1_000.0,
    moving_end: MovingEnd = "rx",
    backend: Backend,
    precision: Literal["single", "double"] = "single",
) -> ChannelCoefficients:
    """Generate a batch of CDL channel impulse responses `h(t, tau)`."""

    if carrier_frequency_hz <= 0 or sampling_frequency_hz <= 0:
        raise ValueError("frequencies must be positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")
    if moving_end not in {"tx", "rx"}:
        raise ValueError("moving_end must be 'tx' or 'rx'")

    batch_size = random_state.polarization_phases_rad.shape[0]
    expected = (batch_size, profile.num_clusters, RAYS_PER_CLUSTER, 4)
    if random_state.polarization_phases_rad.shape != expected:
        raise ValueError(f"polarization phases must have shape {expected}")

    # Choose floating/complex precision in the active NumPy or CuPy namespace.
    xp = backend.xp
    if precision == "single":
        real_dtype, complex_dtype = xp.float32, xp.complex64
    elif precision == "double":
        real_dtype, complex_dtype = xp.float64, xp.complex128
    else:
        raise ValueError("precision must be 'single' or 'double'")

    # Shared simulation quantities used by every cluster.
    velocity = backend.asarray(
        _normalize_velocity(np.asarray(velocity_mps), batch_size),
        dtype=real_dtype,
    )
    sample_times = xp.arange(num_time_steps, dtype=real_dtype) / sampling_frequency_hz
    wavelength = SPEED_OF_LIGHT / carrier_frequency_hz

    # Final sparse CIR tensor. The path axis remains explicit so delay is not
    # destroyed before a later convolution or CIR-to-CFR transformation.
    output = xp.empty(
        (
            batch_size,
            rx_array.num_antennas,
            tx_array.num_antennas,
            profile.num_clusters,
            num_time_steps,
        ),
        dtype=complex_dtype,
    )

    # Generate one delayed MIMO path at a time. This keeps the formula visible
    # and avoids materializing batch x cluster x ray x rx x tx x time at once.
    for cluster in range(profile.num_clusters):
        output[:, :, :, cluster, :] = _cluster_coefficients(
            cluster_index=cluster,
            profile=profile,
            random_state=random_state,
            tx_array=tx_array,
            rx_array=rx_array,
            velocity_mps=velocity,
            sample_times_s=sample_times,
            wavelength_m=wavelength,
            backend=backend,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            moving_end=moving_end,
        )

    # CDL-D/E combine diffuse power and a deterministic specular component
    # according to the profile's Rician K-factor.
    if profile.has_los:
        k = profile.k_factor_linear
        output *= xp.sqrt(1.0 / (k + 1.0))
        output[:, :, :, 0, :] += xp.sqrt(k / (k + 1.0)) * _los_coefficient(
            profile=profile,
            tx_array=tx_array,
            rx_array=rx_array,
            velocity_mps=velocity,
            sample_times_s=sample_times,
            wavelength_m=wavelength,
            backend=backend,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            moving_end=moving_end,
        )

    return ChannelCoefficients(coefficients=output, delays_s=profile.delays_s.copy())
