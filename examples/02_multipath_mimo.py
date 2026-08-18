"""Inspect one MIMO channel matrix formed by many CDL rays."""

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
    axes[1].set_title("∠H [rad]")
    axes[1].set_xlabel("Tx antenna")
    axes[1].set_ylabel("Rx antenna")
    fig.colorbar(phase, ax=axes[1])
    fig.suptitle(f"CDL-A 4x8 MIMO matrix ({backend.name})")
    fig.tight_layout()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
