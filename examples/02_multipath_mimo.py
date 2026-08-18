"""Example 02: Inspect one MIMO matrix formed by many propagation rays.

WHERE ARE WE IN THE COMMUNICATION SYSTEM?
-----------------------------------------

Tx antenna array -> [multipath spatial channel] -> Rx antenna array

This example explains the physical origin of the matrix H in y = Hx + n.

QUESTION
--------

How do ray angles, antenna positions, polarization, and complex phase produce a
different coefficient for every Tx-Rx antenna pair?

INPUT -> OUTPUT
---------------

CDL-A profile + 8 Tx ports + 4 Rx ports + one random realization
    -> H [rx=4, tx=8] after delayed paths are summed for visualization

IMPORTANT SIMPLIFICATION
------------------------

The code sums all delayed paths into one matrix only to draw a heatmap. A
frequency-selective OFDM receiver normally preserves delay or uses one H[k] per
subcarrier.

NOT MODELED
-----------

No transmitted bits, OFDM waveform, noise, channel estimation, equalization,
coding, or protocol layer.
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/02_multipath_mimo.png"))
    args = parser.parse_args()

    carrier = 3.5e9
    backend = get_backend(args.backend)
    channel = CDLChannel(
        "A",
        100e-9,
        carrier,
        tx_array=AntennaArray.ula(8, carrier, pattern="tr38901"),
        rx_array=AntennaArray.ula(4, carrier, pattern="tr38901"),
    )
    result = channel.generate(1, seed=3, backend=backend)
    # Sum all delayed paths only for this visualization.
    matrix = backend.to_numpy(result.coefficients[0, :, :, :, 0].sum(axis=-1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    magnitude = axes[0].imshow(np.abs(matrix), aspect="auto")
    axes[0].set_title("|H| after summing paths")
    axes[0].set_xlabel("Tx antenna")
    axes[0].set_ylabel("Rx antenna")
    fig.colorbar(magnitude, ax=axes[0])
    phase = axes[1].imshow(np.angle(matrix), aspect="auto", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("angle(H) [rad]")
    axes[1].set_xlabel("Tx antenna")
    axes[1].set_ylabel("Rx antenna")
    fig.colorbar(phase, ax=axes[1])
    fig.suptitle(f"CDL-A 4x8 MIMO matrix ({backend.name})")
    fig.tight_layout()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
