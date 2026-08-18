"""Tapped-delay-line fading generated with a transparent sum-of-sinusoids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backend import Backend, BackendName, get_backend
from .constants import SPEED_OF_LIGHT, TWO_PI
from .profiles import ModelLetter, TDLProfile, load_tdl_profile


@dataclass(frozen=True)
class TDLRandomState:
    """Phases and sinusoid directions used by the SoS fading process."""

    phases_rad: np.ndarray
    arrival_angles_rad: np.ndarray
    los_phase_rad: np.ndarray | None


@dataclass(frozen=True)
class TDLResult:
    """TDL coefficients shaped ``[batch, rx, tx, tap, time]``."""

    coefficients: object
    delays_s: np.ndarray


def sample_tdl_random_state(
    profile: TDLProfile,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    *,
    num_sinusoids: int = 32,
    seed: int | None = None,
) -> TDLRandomState:
    if min(batch_size, num_rx_antennas, num_tx_antennas, num_sinusoids) <= 0:
        raise ValueError("all dimensions must be positive")
    rng = np.random.default_rng(seed)
    shape = (
        batch_size,
        num_rx_antennas,
        num_tx_antennas,
        profile.num_taps,
        num_sinusoids,
    )
    phases = rng.uniform(-np.pi, np.pi, size=shape)
    arrival_angles = rng.uniform(-np.pi, np.pi, size=shape)
    los_phase = (
        rng.uniform(-np.pi, np.pi, size=(batch_size, num_rx_antennas, num_tx_antennas))
        if profile.has_los
        else None
    )
    return TDLRandomState(phases, arrival_angles, los_phase)


@dataclass
class TDLChannel:
    """Educational TDL A-E model.

    The standardized delay/power profiles are exact. The temporal fading uses
    a compact Jakes-style sum-of-sinusoids, chosen because the equation is easy
    to read. It is not a conformance replacement for every correlation option
    in TR 38.901.
    """

    model: ModelLetter
    delay_spread_s: float
    carrier_frequency_hz: float
    num_rx_antennas: int = 1
    num_tx_antennas: int = 1
    num_sinusoids: int = 32
    precision: Literal["single", "double"] = "single"

    def __post_init__(self) -> None:
        if self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive")
        self.profile = load_tdl_profile(self.model, self.delay_spread_s)

    def sample_state(self, batch_size: int, *, seed: int | None = None) -> TDLRandomState:
        return sample_tdl_random_state(
            self.profile,
            batch_size,
            self.num_rx_antennas,
            self.num_tx_antennas,
            num_sinusoids=self.num_sinusoids,
            seed=seed,
        )

    def generate(
        self,
        batch_size: int,
        *,
        num_time_steps: int,
        sampling_frequency_hz: float,
        speed_mps: float = 0.0,
        seed: int | None = None,
        random_state: TDLRandomState | None = None,
        backend: Backend | BackendName = "auto",
    ) -> TDLResult:
        if num_time_steps <= 0 or sampling_frequency_hz <= 0:
            raise ValueError("time dimensions and sampling frequency must be positive")
        if speed_mps < 0:
            raise ValueError("speed_mps must be non-negative")

        selected = get_backend(backend) if isinstance(backend, str) else backend
        xp = selected.xp
        state = random_state or self.sample_state(batch_size, seed=seed)
        if state.phases_rad.shape[0] != batch_size:
            raise ValueError("random_state batch dimension does not match batch_size")

        if self.precision == "single":
            rdtype, cdtype = xp.float32, xp.complex64
        elif self.precision == "double":
            rdtype, cdtype = xp.float64, xp.complex128
        else:
            raise ValueError("precision must be 'single' or 'double'")

        phases = selected.asarray(state.phases_rad, dtype=rdtype)
        angles = selected.asarray(state.arrival_angles_rad, dtype=rdtype)
        times = xp.arange(num_time_steps, dtype=rdtype) / sampling_frequency_hz
        max_doppler_hz = speed_mps / (SPEED_OF_LIGHT / self.carrier_frequency_hz)
        frequencies = max_doppler_hz * xp.cos(angles)
        oscillators = xp.exp(
            1j * (phases[..., None] + TWO_PI * frequencies[..., None] * times)
        ).astype(cdtype, copy=False)
        diffuse = xp.mean(oscillators, axis=-2) * xp.sqrt(xp.asarray(self.num_sinusoids, dtype=rdtype))
        diffuse *= xp.sqrt(
            selected.asarray(self.profile.powers_linear, dtype=rdtype)[None, None, None, :, None]
        )

        if self.profile.has_los:
            k = self.profile.k_factor_linear
            diffuse *= xp.sqrt(xp.asarray(1.0 / (k + 1.0), dtype=rdtype))
            if state.los_phase_rad is None:
                raise ValueError("LOS profile requires los_phase_rad")
            los_phase = selected.asarray(state.los_phase_rad, dtype=rdtype)
            los = xp.exp(1j * los_phase[..., None]).astype(cdtype, copy=False)
            los = xp.broadcast_to(los, (*los.shape[:-1], num_time_steps))
            diffuse[..., 0, :] += xp.sqrt(xp.asarray(k / (k + 1.0), dtype=rdtype)) * los

        return TDLResult(coefficients=diffuse, delays_s=self.profile.delays_s.copy())
