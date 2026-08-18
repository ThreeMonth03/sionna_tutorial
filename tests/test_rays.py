from __future__ import annotations

import numpy as np

from sionna_tutorial.constants import RAY_OFFSETS, RAYS_PER_CLUSTER
from sionna_tutorial.profiles import load_cdl_profile
from sionna_tutorial.rays import sample_cdl_random_state


def test_ray_offsets_match_twenty_ray_model() -> None:
    assert RAYS_PER_CLUSTER == 20
    np.testing.assert_allclose(RAY_OFFSETS[0::2], -RAY_OFFSETS[1::2])


def test_random_state_is_reproducible() -> None:
    profile = load_cdl_profile("A")
    one = sample_cdl_random_state(profile, 2, seed=42)
    two = sample_cdl_random_state(profile, 2, seed=42)
    np.testing.assert_array_equal(one.angles.aoa_rad, two.angles.aoa_rad)
    np.testing.assert_array_equal(one.polarization_phases_rad, two.polarization_phases_rad)


def test_random_coupling_preserves_each_cluster_ray_set() -> None:
    profile = load_cdl_profile("B")
    state = sample_cdl_random_state(profile, 3, seed=4)
    expected = np.sort(
        profile.aoa_rad[0] + np.deg2rad(profile.cluster_asa_deg * RAY_OFFSETS)
    )
    for batch in range(3):
        np.testing.assert_allclose(np.sort(state.angles.aoa_rad[batch, 0]), expected)
