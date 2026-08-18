"""Example 05: Show how terminal speed changes a time-varying channel.

WHERE ARE WE IN THE COMMUNICATION SYSTEM?
-----------------------------------------

Tx antenna -> [time-varying wireless channel H(t, tau)] -> Rx antenna

This example remains inside channel generation. It does not run a receiver.

QUESTION
--------

Why does increasing terminal velocity make the complex channel fade faster?

INPUT -> OUTPUT
---------------

The same sampled CDL-B rays + velocities 0, 30, and 120 km/h
    -> one composite channel-magnitude trace for each speed

METHOD
------

The random channel state is sampled once and reused. Only velocity changes, so
the plot isolates the Doppler term instead of comparing unrelated channels.

IMPORTANT SIMPLIFICATION
------------------------

The example uses one Tx port and one Rx port, then sums all delayed paths into a
narrowband trace. It is a Doppler/coherence-time demonstration, not a complete
OFDM or receiver simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sionna_tutorial import AntennaArray, CDLChannel, get_backend
from sionna_tutorial.plotting import save_or_show


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy", "auto"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/05_mobility_doppler.png"))
    args = parser.parse_args()

    carrier = 3.5e9
    backend = get_backend(args.backend)
    channel = CDLChannel(
        "B",
        100e-9,
        carrier,
        tx_array=AntennaArray.ula(1, carrier),
        rx_array=AntennaArray.ula(1, carrier),
    )
    state = channel.sample_state(1, seed=8)
    sample_rate = 5_000.0
    num_steps = 500
    time_ms = np.arange(num_steps) / sample_rate * 1e3

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for speed_kmh in (0.0, 30.0, 120.0):
        result = channel.generate(
            1,
            num_time_steps=num_steps,
            sampling_frequency_hz=sample_rate,
            velocity_mps=(speed_kmh / 3.6, 0.0, 0.0),
            random_state=state,
            backend=backend,
        )
        # Sum paths to observe one composite narrowband channel trace.
        trace = backend.to_numpy(result.coefficients[0, 0, 0].sum(axis=0))
        ax.plot(time_ms, 20.0 * np.log10(np.abs(trace) + 1e-12), label=f"{speed_kmh:g} km/h")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Composite channel magnitude [dB]")
    ax.set_title("Mobility produces faster Doppler fading")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
