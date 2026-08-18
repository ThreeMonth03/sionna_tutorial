"""Example 04: Generate and visualize one 3GPP CDL MIMO channel realization.

WHERE ARE WE IN THE COMMUNICATION SYSTEM?
-----------------------------------------

Tx antenna array -> [3GPP clustered wireless channel] -> Rx antenna array

This is the main channel-generation example. Examples 01-03 introduce the
spatial phase, MIMO, and delay concepts that are combined here.

QUESTION
--------

How do cluster delay/power, 20 rays, four angular dimensions, antenna fields,
polarization, array geometry, random phase, and Doppler form H(t, tau)?

INPUT -> OUTPUT
---------------

CDL model + delay spread + carrier + Tx/Rx arrays + random state
    -> coefficients [batch=1, rx, tx, path, time=1]
    -> delays [path]

WHAT THE FIGURE SHOWS
---------------------

Left: realized power carried by every delayed CDL path.
Right: instantaneous MIMO magnitude after paths are summed for visualization.

NOT MODELED
-----------

No geographical UMi/UMa/RMa scenario, path loss, RF circuit, transmitted data,
pilot, channel estimation, detector, coding, or protocol layer.
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
    parser.add_argument("--model", choices=list("ABCDE"), default="C")
    parser.add_argument("--backend", choices=["numpy", "cupy", "auto"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/04_cdl_channel.png"))
    args = parser.parse_args()

    carrier = 3.5e9
    backend = get_backend(args.backend)
    channel = CDLChannel(
        args.model,
        300e-9,
        carrier,
        tx_array=AntennaArray.upa(2, 2, carrier, polarization="dual", pattern="tr38901"),
        rx_array=AntennaArray.ula(2, carrier, polarization="single", pattern="omni"),
    )
    result = channel.generate(
        1,
        num_time_steps=1,
        velocity_mps=(0.0, 0.0, 0.0),
        seed=11,
        backend=backend,
    )
    h = backend.to_numpy(result.coefficients[0, :, :, :, 0])
    path_power = np.sum(np.abs(h) ** 2, axis=(0, 1))
    matrix = h.sum(axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].stem(result.delays_s * 1e9, 10.0 * np.log10(path_power / path_power.sum()))
    axes[0].set_xlabel("Delay [ns]")
    axes[0].set_ylabel("Realization power [dB]")
    axes[0].set_title(f"CDL-{args.model} path powers")
    axes[0].grid(True, alpha=0.3)
    image = axes[1].imshow(np.abs(matrix), aspect="auto")
    axes[1].set_xlabel("Tx antenna port")
    axes[1].set_ylabel("Rx antenna port")
    axes[1].set_title("Instantaneous |H|")
    fig.colorbar(image, ax=axes[1])
    fig.suptitle(f"CDL-{args.model} ({backend.name})")
    fig.tight_layout()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
