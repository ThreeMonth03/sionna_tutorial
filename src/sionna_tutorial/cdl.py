"""High-level educational CDL channel model."""

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
    """A standards-profiled, educational CDL A-E MIMO channel.

    This class deliberately separates random-state sampling from deterministic
    coefficient computation. Reuse a state to compare NumPy and CuPy exactly.
    """

    model: ModelLetter
    delay_spread_s: float
    carrier_frequency_hz: float
    tx_array: AntennaArray
    rx_array: AntennaArray
    moving_end: MovingEnd = "rx"
    precision: Literal["single", "double"] = "single"

    def __post_init__(self) -> None:
        self.profile: CDLProfile = load_cdl_profile(self.model, self.delay_spread_s)

    def sample_state(self, batch_size: int, *, seed: int | None = None) -> CDLRandomState:
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
        """Generate channel path coefficients and delays."""

        selected = get_backend(backend) if isinstance(backend, str) else backend
        state = random_state or self.sample_state(batch_size, seed=seed)
        if state.polarization_phases_rad.shape[0] != batch_size:
            raise ValueError("random_state batch dimension does not match batch_size")
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
