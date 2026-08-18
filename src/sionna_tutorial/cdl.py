"""High-level entry point for wireless-channel generation.

SYSTEM LOCATION
---------------

Tx antenna array -> [CDL propagation channel generated here] -> Rx antenna array

This module produces the channel impulse response `h(t, tau)`. It does not
transmit data through that channel, estimate the channel at a receiver, or
decode user bits. The tiny channel-application example lives in `ofdm.py`.

HIGH-LEVEL FLOW
---------------

1. Load a standardized CDL-A/B/C/D/E statistical profile.
2. Sample one or more random small-scale channel states.
3. Convert every cluster and ray into deterministic MIMO coefficients.
4. Return coefficients `[batch, rx, tx, path, time]` and path delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .arrays import AntennaArray
from .backend import Backend, BackendName, get_backend
from .coefficients import ChannelCoefficients, MovingEnd, generate_cdl_coefficients
from .profiles import CDLProfile, ModelLetter, load_cdl_profile
from .rays import CDLRandomState, sample_cdl_random_state


@dataclass
class CDLChannel:
    """Standards-profiled educational CDL-A/E MIMO propagation channel.

    `CDLChannel` is a channel generator, not a complete link simulator.

    Random-state sampling is deliberately separated from deterministic
    coefficient calculation. Reuse one `CDLRandomState` to compare NumPy and
    CuPy using the same physical rays and phases.
    """

    model: ModelLetter
    delay_spread_s: float
    carrier_frequency_hz: float
    tx_array: AntennaArray
    rx_array: AntennaArray
    moving_end: MovingEnd = "rx"
    precision: Literal["single", "double"] = "single"

    def __post_init__(self) -> None:
        # The profile is the average environment description: delays, powers,
        # cluster-center angles, spreads, XPR, and optional LOS metadata.
        self.profile: CDLProfile = load_cdl_profile(self.model, self.delay_spread_s)

    def sample_state(self, batch_size: int, *, seed: int | None = None) -> CDLRandomState:
        """Sample per-realization ray coupling and polarization phases."""

        return sample_cdl_random_state(self.profile, batch_size, seed=seed)

    def generate(
        self,
        batch_size: int,
        *,
        num_time_steps: int = 1,
        sampling_frequency_hz: float = 1_000.0,
        velocity_mps: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = None,
        random_state: CDLRandomState | None = None,
        backend: Backend | BackendName = "auto",
    ) -> ChannelCoefficients:
        """Generate a batch of propagation-channel impulse responses.

        Returns
        -------
        ChannelCoefficients
            `coefficients` has shape `[batch, rx, tx, path, time]` and
            `delays_s` has shape `[path]`.

        Notes
        -----
        This method performs channel generation only. Passing transmitted
        symbols through the returned channel is a later operation. A real
        receiver's channel estimation is later still and is not implemented by
        this method.
        """

        # Select the numerical array namespace. The same equations are used by
        # NumPy on CPU and CuPy on CUDA.
        selected = get_backend(backend) if isinstance(backend, str) else backend

        # Either sample a new physical realization or reuse one for exact
        # reproducibility and CPU/GPU differential tests.
        state = random_state or self.sample_state(batch_size, seed=seed)
        if state.polarization_phases_rad.shape[0] != batch_size:
            raise ValueError("random_state batch dimension does not match batch_size")

        # Convert the profile + sampled rays + arrays + motion into h(t, tau).
        return generate_cdl_coefficients(
            self.profile,
            state,
            self.tx_array,
            self.rx_array,
            carrier_frequency_hz=self.carrier_frequency_hz,
            velocity_mps=velocity_mps,
            num_time_steps=num_time_steps,
            sampling_frequency_hz=sampling_frequency_hz,
            moving_end=self.moving_end,
            backend=selected,
            precision=self.precision,
        )
