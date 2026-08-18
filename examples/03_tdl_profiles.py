"""Plot the standardized TDL A-E power-delay profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sionna_tutorial.profiles import load_tdl_profile
from sionna_tutorial.plotting import save_or_show


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/03_tdl_profiles.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in "ABCDE":
        profile = load_tdl_profile(model, 100e-9)  # type: ignore[arg-type]
        ax.scatter(
            profile.delays_s * 1e9,
            10.0 * np.log10(profile.powers_linear),
            s=20,
            label=f"TDL-{model}",
        )
    ax.set_xlabel("Delay [ns]")
    ax.set_ylabel("Normalized diffuse tap power [dB]")
    ax.set_title("3GPP TR 38.901 TDL profiles, 100 ns delay spread")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_or_show(fig, args.output)


if __name__ == "__main__":
    main()
