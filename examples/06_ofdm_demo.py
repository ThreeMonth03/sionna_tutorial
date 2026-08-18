"""QPSK → CDL MIMO channel → perfect-CSI ZF equalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from sionna_tutorial import AntennaArray, CDLChannel, get_backend
from sionna_tutorial.ofdm import run_perfect_csi_ofdm_demo
from sionna_tutorial.plotting import plot_constellations, save_or_show


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy", "auto"], default="auto")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/06_ofdm_demo.png"))
    args = parser.parse_args()

    carrier = 3.5e9
    backend = get_backend(args.backend)
    channel = CDLChannel(
        "C",
        100e-9,
        carrier,
        tx_array=AntennaArray.ula(2, carrier),
        rx_array=AntennaArray.ula(4, carrier),
    )
    realization = channel.generate(64, seed=13, backend=backend)
    result = run_perfect_csi_ofdm_demo(
        realization,
        backend=backend,
        num_subcarriers=64,
        snr_db=args.snr_db,
        seed=14,
    )

    tx = backend.to_numpy(result.transmitted_symbols)
    rx = backend.to_numpy(result.received_symbols[:, 0])
    equalized = backend.to_numpy(result.equalized_symbols)
    fig = plot_constellations(tx, rx, equalized)
    fig.suptitle(
        f"CDL-C 2x4 OFDM demo · {backend.name} · SNR={args.snr_db:g} dB · "
        f"BER={result.bit_error_rate:.3e}",
        y=1.03,
    )
    save_or_show(fig, args.output)
    print(f"BER: {result.bit_error_rate:.6e}")


if __name__ == "__main__":
    main()
