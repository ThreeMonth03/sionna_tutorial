"""Generate the twenty rays inside every CDL cluster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import RAY_OFFSETS, RAYS_PER_CLUSTER
from .profiles import CDLProfile


@dataclass(frozen=True)
class RayAngles:
    """Per-batch ray angles, all with shape ``[batch, cluster, ray]``."""

    aod_rad: np.ndarray
    aoa_rad: np.ndarray
    zod_rad: np.ndarray
    zoa_rad: np.ndarray


@dataclass(frozen=True)
class CDLRandomState:
    """All stochastic inputs needed by the deterministic coefficient engine."""

    angles: RayAngles
    polarization_phases_rad: np.ndarray


def _expanded_angles(cluster_angles_rad: np.ndarray, spread_deg: float) -> np.ndarray:
    offsets_rad = np.deg2rad(spread_deg * RAY_OFFSETS)
    return cluster_angles_rad[:, None] + offsets_rad[None, :]


def _shuffle_last_axis(values: np.ndarray, rng: np.random.Generator, batch_size: int) -> np.ndarray:
    clusters, rays = values.shape
    output = np.empty((batch_size, clusters, rays), dtype=np.float64)
    for batch in range(batch_size):
        for cluster in range(clusters):
            output[batch, cluster] = values[cluster, rng.permutation(rays)]
    return output


def sample_cdl_random_state(
    profile: CDLProfile,
    batch_size: int,
    *,
    seed: int | None = None,
) -> CDLRandomState:
    """Sample random coupling and four polarization phases per ray.

    Keeping this step in NumPy makes a realization exactly reproducible across
    the NumPy and CuPy coefficient engines. That is useful for learning and
    differential testing.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)

    base_aod = _expanded_angles(profile.aod_rad, profile.cluster_asd_deg)
    base_aoa = _expanded_angles(profile.aoa_rad, profile.cluster_asa_deg)
    base_zod = _expanded_angles(profile.zod_rad, profile.cluster_zsd_deg)
    base_zoa = _expanded_angles(profile.zoa_rad, profile.cluster_zsa_deg)

    angles = RayAngles(
        aod_rad=_shuffle_last_axis(base_aod, rng, batch_size),
        aoa_rad=_shuffle_last_axis(base_aoa, rng, batch_size),
        zod_rad=_shuffle_last_axis(base_zod, rng, batch_size),
        zoa_rad=_shuffle_last_axis(base_zoa, rng, batch_size),
    )
    phases = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(batch_size, profile.num_clusters, RAYS_PER_CLUSTER, 4),
    )
    return CDLRandomState(angles=angles, polarization_phases_rad=phases)
