"""Example 01: Build one propagation-ray coefficient by hand.

WHERE ARE WE IN THE COMMUNICATION SYSTEM?
-----------------------------------------

PHY transmitter -> Tx antenna -> [one ray in the wireless channel] -> Rx side

This example is inside the propagation channel. It does not transmit bits or
run a receiver.

QUESTION
--------

Why does one plane wave have a different complex phase at every antenna, and
why does motion rotate that phase over time?

INPUT -> OUTPUT
---------------

Tx array positions + ray direction + wavelength + velocity + sampled time
    -> complex coefficient [tx_antenna, time]

WHAT TO READ
------------

`unit_sphere_vector()`, `spatial_phase()`, and `doppler_phase()` in
`src/sionna_tutorial/geometry.py`.

NOT MODELED
-----------

No multipath, delay spread, receive array, transmitted data, noise, channel
estimation, equalizer, coding, or protocol layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sionna_tutorial.arrays import AntennaArray
from sionna_tutorial.constants import SPEED_OF_LIGHT
from sionna_tutorial.geometry import doppler_phase, spatial_phase, unit_sphere_vector
from sionna_tutorial.plotting import save_or_show


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/01_single_path.png"))
    args = parser.parse_args()

    carrier = 3.5e9
    wavelength = SPEED_OF_LIGHT / carrier
    tx = AntennaArray.ula(4, carrier)
    direction = unit_sphere_vector(np.deg2rad(90.0), np.deg2rad(30.0), xp=np)
    array_response = spatial_phase(
        tx.positions_m,
        direction,
        wavelength,
        xp=np,
        complex_dtype=np.complex128,
    )

    times = np.linspace(0.0, 0.02, 400)
    direction_batch = np.broadcast_to(direction, (1, 1, 3))
    doppler = doppler_phase(
        direction_batch,
        np.array([[20.0, 0.0, 0.0]]),
        times,
        wavelength,
        xp=np,
        complex_dtype=np.complex128,
    )[0, 0]
    coefficient = array_response[:, None] * doppler[None, :]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for antenna in range(tx.num_antennas):
        axes[0].plot(times * 1e3, coefficient[antenna].real, label=f"Tx {antenna}")
        axes[1].plot(times * 1e3, np.unwrap(np.angle(coefficient[antenna])))
    axes[0].set_ylabel("Real{h(t)}")
    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Unwrapped phase [rad]")
    axes[1].set_xlabel("Time [ms]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("One path = array phase x Doppler phase")
    fig.tight_layout()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
