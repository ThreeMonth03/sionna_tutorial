from __future__ import annotations

import numpy as np

from sionna_tutorial.arrays import AntennaArray, element_field_components
from sionna_tutorial.geometry import spatial_phase, unit_sphere_vector


def test_unit_sphere_axes() -> None:
    x = unit_sphere_vector(np.pi / 2.0, 0.0, xp=np)
    y = unit_sphere_vector(np.pi / 2.0, np.pi / 2.0, xp=np)
    z = unit_sphere_vector(0.0, 0.0, xp=np)
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0], atol=1e-12)


def test_ula_is_centered_and_half_wavelength_spaced() -> None:
    carrier = 3.0e9
    array = AntennaArray.ula(4, carrier)
    wavelength = 299_792_458.0 / carrier
    np.testing.assert_allclose(array.positions_m.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.diff(array.positions_m[:, 1]), wavelength / 2.0)


def test_broadside_phase_is_constant_for_y_axis_ula() -> None:
    carrier = 3.0e9
    array = AntennaArray.ula(4, carrier)
    direction = np.array([1.0, 0.0, 0.0])
    phase = spatial_phase(
        array.positions_m,
        direction,
        299_792_458.0 / carrier,
        xp=np,
        complex_dtype=np.complex128,
    )
    np.testing.assert_allclose(phase, np.ones(4), atol=1e-12)


def test_dual_polarized_array_duplicates_positions() -> None:
    array = AntennaArray.ula(3, 3.5e9, polarization="dual")
    assert array.num_antennas == 6
    np.testing.assert_allclose(array.positions_m[0::2], array.positions_m[1::2])
    np.testing.assert_allclose(np.rad2deg(array.slant_angles_rad[:2]), [-45.0, 45.0])


def test_38901_element_pattern_at_boresight_and_backoff() -> None:
    boresight = element_field_components(
        np.array([[np.pi / 2]]),
        np.array([[0.0]]),
        np.array([0.0]),
        pattern="tr38901",
        xp=np,
    )
    off_axis = element_field_components(
        np.array([[0.0]]),
        np.array([[np.pi]]),
        np.array([0.0]),
        pattern="tr38901",
        xp=np,
    )
    assert np.abs(boresight[..., 0]).item() > np.abs(off_axis[..., 0]).item()
