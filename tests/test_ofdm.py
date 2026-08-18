from __future__ import annotations

import numpy as np

from sionna_tutorial.backend import get_backend
from sionna_tutorial.coefficients import ChannelCoefficients
from sionna_tutorial.ofdm import (
    cir_to_frequency_response,
    qpsk_hard_demapper,
    qpsk_map,
    run_perfect_csi_ofdm_demo,
)


def test_zero_delay_cir_is_flat_in_frequency() -> None:
    coefficients = np.ones((2, 1, 1, 1, 3), dtype=np.complex64)
    channel = ChannelCoefficients(coefficients, np.array([0.0]))
    frequencies = np.linspace(-1e6, 1e6, 11)
    response = cir_to_frequency_response(channel, frequencies, backend=get_backend("numpy"))
    assert response.shape == (2, 1, 1, 3, 11)
    np.testing.assert_allclose(response, 1.0)


def test_two_path_frequency_response_matches_closed_form() -> None:
    coefficients = np.ones((1, 1, 1, 2, 1), dtype=np.complex128)
    delay = 1e-6
    channel = ChannelCoefficients(coefficients, np.array([0.0, delay]))
    frequencies = np.array([0.0, 1.0 / (2.0 * delay)])
    response = cir_to_frequency_response(channel, frequencies, backend=get_backend("numpy"))
    np.testing.assert_allclose(response[0, 0, 0, 0], [2.0, 0.0], atol=1e-12)


def test_qpsk_round_trip() -> None:
    bits = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]])
    symbols = qpsk_map(bits, xp=np)
    detected = qpsk_hard_demapper(symbols, xp=np)
    np.testing.assert_array_equal(detected, bits.astype(bool))


def test_identity_mimo_ofdm_has_zero_ber_at_high_snr() -> None:
    batch = 8
    identity = np.eye(2, dtype=np.complex64)
    coefficients = np.broadcast_to(identity[None, :, :, None, None], (batch, 2, 2, 1, 1)).copy()
    channel = ChannelCoefficients(coefficients, np.array([0.0]))
    result = run_perfect_csi_ofdm_demo(
        channel,
        backend=get_backend("numpy"),
        num_subcarriers=32,
        snr_db=80.0,
        seed=2,
    )
    assert result.bit_error_rate == 0.0
