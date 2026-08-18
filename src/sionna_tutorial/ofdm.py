"""Minimal frequency-domain application of a generated wireless channel.

SYSTEM LOCATION
---------------

uncoded QPSK -> frequency-domain subcarriers -> H[k] X[k] + N[k]
    -> perfect-CSI pseudo-inverse -> hard QPSK bits

`cdl.py` and `coefficients.py` generate the propagation channel `h(t, tau)`.
This module is where the tutorial first **applies** that channel to data.

IMPORTANT BOUNDARY
------------------

This is not a complete OFDM or 5G NR implementation. It does not create a
time-domain IFFT waveform or cyclic prefix, and it has no synchronization,
resource grid, pilot/DMRS, channel estimation, coding, soft demapping, or HARQ.
The equalizer is given the exact simulated channel, which is perfect CSI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import Backend
from .coefficients import ChannelCoefficients
from .constants import TWO_PI


@dataclass(frozen=True)
class OFDMDemoResult:
    """Intermediate and final values from the tiny channel-application demo."""

    transmitted_symbols: object
    received_symbols: object
    equalized_symbols: object
    detected_bits: object
    bit_error_rate: float
    channel_frequency_response: object


def subcarrier_frequencies(num_subcarriers: int, subcarrier_spacing_hz: float) -> np.ndarray:
    """Return baseband subcarrier frequencies centered around zero."""

    if num_subcarriers <= 0 or subcarrier_spacing_hz <= 0:
        raise ValueError("OFDM dimensions must be positive")
    indices = np.arange(num_subcarriers, dtype=np.float64) - num_subcarriers // 2
    return indices * subcarrier_spacing_hz


def cir_to_frequency_response(
    channel: ChannelCoefficients,
    frequencies_hz: np.ndarray,
    *,
    backend: Backend,
    normalize: bool = False,
) -> Any:
    """Transform sparse `h(t, tau)` paths into one MIMO matrix per subcarrier.

    Input
    -----
    `channel.coefficients`: `[batch, rx, tx, path, time]`

    Output
    ------
    `H`: `[batch, rx, tx, time, frequency]`

    For every frequency `f_k`, delayed paths are combined as
    `H[k,t] = sum_n h[n,t] exp(-j 2 pi f_k tau_n)`.
    """

    xp = backend.xp
    coefficients = channel.coefficients
    frequencies = backend.asarray(np.asarray(frequencies_hz, dtype=np.float64))
    delays = backend.asarray(channel.delays_s)

    # Phase contributed by every delayed path at every subcarrier frequency.
    phase = xp.exp(-1j * TWO_PI * delays[:, None] * frequencies[None, :])

    # Reduce the path axis: sparse CIR h(t, tau) -> frequency response H(f, t).
    response = xp.einsum("brupt,pk->brutk", coefficients, phase, optimize=True)
    if normalize:
        power = xp.mean(xp.abs(response) ** 2, axis=(-4, -3, -2, -1), keepdims=True)
        response = response / xp.sqrt(xp.maximum(power, xp.finfo(power.dtype).tiny))
    return response


def qpsk_map(bits: Any, *, xp: Any) -> Any:
    """Gray-map bit pairs to unit-energy QPSK symbols."""

    bits = xp.asarray(bits)
    if bits.shape[-1] != 2:
        raise ValueError("last bit dimension must have size 2")
    real = 1.0 - 2.0 * bits[..., 0]
    imag = 1.0 - 2.0 * bits[..., 1]
    return (real + 1j * imag) / xp.sqrt(2.0)


def qpsk_hard_demapper(symbols: Any, *, xp: Any) -> Any:
    """Return hard QPSK bit decisions from complex-symbol quadrants."""

    return xp.stack([xp.real(symbols) < 0.0, xp.imag(symbols) < 0.0], axis=-1)


def run_perfect_csi_ofdm_demo(
    channel: ChannelCoefficients,
    *,
    backend: Backend,
    num_subcarriers: int = 64,
    subcarrier_spacing_hz: float = 15_000.0,
    snr_db: float = 20.0,
    seed: int = 0,
    time_index: int = 0,
) -> OFDMDemoResult:
    """Apply a known MIMO channel to QPSK subcarriers and ideal-ZF equalize.

    The function is intentionally split into visible transmitter, channel, and
    receiver-like stages. It is a small application that gives physical meaning
    to the channel tensor, not a complete NR waveform.
    """

    xp = backend.xp
    h = channel.coefficients
    batch_size, num_rx, num_tx, _, num_times = h.shape
    if not 0 <= time_index < num_times:
        raise ValueError("time_index is outside the generated channel")
    if num_rx < num_tx:
        raise ValueError("ZF demo requires num_rx >= num_tx")

    # ------------------------------------------------------------------
    # 1. Minimal transmitter: random uncoded bits -> QPSK on subcarriers.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    bits_np = rng.integers(0, 2, size=(batch_size, num_tx, num_subcarriers, 2), dtype=np.int8)
    bits = backend.asarray(bits_np)
    x = qpsk_map(bits, xp=xp)

    # ------------------------------------------------------------------
    # 2. Channel representation: h(t, tau) -> H[k] for every subcarrier.
    # ------------------------------------------------------------------
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)
    response = cir_to_frequency_response(channel, frequencies, backend=backend, normalize=True)
    h_f = response[:, :, :, time_index, :]  # [batch, rx, tx, frequency]

    # ------------------------------------------------------------------
    # 3. Channel application: Y[k] = H[k] X[k].
    # ------------------------------------------------------------------
    y_clean = xp.einsum("brtk,btk->brk", h_f, x, optimize=True)

    # ------------------------------------------------------------------
    # 4. Add complex Gaussian noise at the requested SNR.
    # ------------------------------------------------------------------
    signal_power = float(backend.to_numpy(xp.mean(xp.abs(y_clean) ** 2)))
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    noise_np = (
        rng.standard_normal(y_clean.shape) + 1j * rng.standard_normal(y_clean.shape)
    ) * np.sqrt(noise_variance / 2.0)
    noise = backend.asarray(noise_np, dtype=y_clean.dtype)
    y = y_clean + noise

    # ------------------------------------------------------------------
    # 5. Ideal receiver: use the exact H[k], not an estimated H_hat[k].
    #    This is perfect CSI. A real receiver would first process pilots.
    # ------------------------------------------------------------------
    h_bk = xp.transpose(h_f, (0, 3, 1, 2))
    y_bk = xp.transpose(y, (0, 2, 1))
    h_pinv = xp.linalg.pinv(h_bk)
    x_hat_bk = xp.einsum("bktr,bkr->bkt", h_pinv, y_bk, optimize=True)
    x_hat = xp.transpose(x_hat_bk, (0, 2, 1))

    # ------------------------------------------------------------------
    # 6. Hard QPSK decisions and one small BER measurement.
    # ------------------------------------------------------------------
    detected = qpsk_hard_demapper(x_hat, xp=xp)
    bit_errors = backend.to_numpy(xp.count_nonzero(detected != bits))
    ber = float(bit_errors) / float(bits_np.size)

    return OFDMDemoResult(
        transmitted_symbols=x,
        received_symbols=y,
        equalized_symbols=x_hat,
        detected_bits=detected,
        bit_error_rate=ber,
        channel_frequency_response=response,
    )
