from __future__ import annotations

import numpy as np
import pytest

from sionna_tutorial.profiles import load_cdl_profile, load_tdl_profile


@pytest.mark.parametrize("model,count", [("A", 23), ("B", 23), ("C", 24), ("D", 13), ("E", 14)])
def test_cdl_profile_counts_and_normalization(model: str, count: int) -> None:
    profile = load_cdl_profile(model, 100e-9)
    assert profile.num_clusters == count
    assert profile.delays_s.shape == (count,)
    assert np.isclose(profile.powers_linear.sum(), 1.0)
    assert np.all(profile.powers_linear > 0.0)
    assert np.all(profile.delays_s >= 0.0)


@pytest.mark.parametrize("model,count", [("A", 23), ("B", 23), ("C", 24), ("D", 13), ("E", 14)])
def test_tdl_profile_counts_and_normalization(model: str, count: int) -> None:
    profile = load_tdl_profile(model, 100e-9)
    assert profile.num_taps == count
    assert np.isclose(profile.powers_linear.sum(), 1.0)
    assert profile.delays_s.shape == (count,)


def test_los_profiles_extract_specular_component() -> None:
    for model in ("D", "E"):
        cdl = load_cdl_profile(model)
        tdl = load_tdl_profile(model)
        assert cdl.has_los and tdl.has_los
        assert cdl.k_factor_linear > 0.0
        assert tdl.k_factor_linear > 0.0
        assert cdl.los_angles_rad is not None
        assert cdl.delays_s[0] == 0.0
        assert tdl.delays_s[0] == 0.0


def test_delay_spread_scales_delays() -> None:
    short = load_cdl_profile("A", 30e-9)
    long = load_cdl_profile("A", 300e-9)
    np.testing.assert_allclose(long.delays_s, short.delays_s * 10.0)


def test_invalid_model_and_delay() -> None:
    with pytest.raises(ValueError):
        load_cdl_profile("Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        load_tdl_profile("A", 0.0)
