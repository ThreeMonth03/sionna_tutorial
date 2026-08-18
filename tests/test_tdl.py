from __future__ import annotations

import numpy as np

from sionna_tutorial import TDLChannel


def test_tdl_shape_dtype_and_reproducibility() -> None:
    channel = TDLChannel(
        "A",
        100e-9,
        3.5e9,
        num_rx_antennas=2,
        num_tx_antennas=3,
        num_sinusoids=16,
    )
    state = channel.sample_state(4, seed=5)
    one = channel.generate(
        4,
        num_time_steps=7,
        sampling_frequency_hz=1_000.0,
        speed_mps=8.0,
        random_state=state,
        backend="numpy",
    )
    two = channel.generate(
        4,
        num_time_steps=7,
        sampling_frequency_hz=1_000.0,
        speed_mps=8.0,
        random_state=state,
        backend="numpy",
    )
    assert one.coefficients.shape == (4, 2, 3, 23, 7)
    assert one.coefficients.dtype == np.complex64
    np.testing.assert_array_equal(one.coefficients, two.coefficients)


def test_zero_speed_is_time_invariant() -> None:
    channel = TDLChannel("C", 100e-9, 3.5e9)
    result = channel.generate(
        3,
        num_time_steps=8,
        sampling_frequency_hz=1_000.0,
        speed_mps=0.0,
        seed=7,
        backend="numpy",
    )
    np.testing.assert_allclose(result.coefficients[..., 0], result.coefficients[..., -1])


def test_los_tdl_has_expected_tap_count() -> None:
    channel = TDLChannel("E", 100e-9, 3.5e9)
    result = channel.generate(
        2,
        num_time_steps=2,
        sampling_frequency_hz=1_000.0,
        seed=9,
        backend="numpy",
    )
    assert result.coefficients.shape[3] == 14
