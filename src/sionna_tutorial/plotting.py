"""Plot helpers kept outside the numerical core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_or_show(fig: Any, output: str | Path | None) -> None:
    if output is None:
        plt.show()
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}")


def plot_power_delay_profile(delays_s: np.ndarray, powers: np.ndarray, title: str) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    markerline, _stemlines, baseline = ax.stem(delays_s * 1e9, 10.0 * np.log10(powers))
    markerline.set_markersize(4)
    baseline.set_visible(False)
    ax.set_xlabel("Delay [ns]")
    ax.set_ylabel("Power [dB]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_constellations(tx: np.ndarray, rx: np.ndarray, equalized: np.ndarray) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        [tx, rx, equalized],
        ["Transmitted", "Received antenna 0", "ZF equalized"],
        strict=True,
    ):
        flat = np.asarray(data).reshape(-1)
        ax.scatter(flat.real, flat.imag, s=8, alpha=0.45)
        ax.axhline(0.0, linewidth=0.7)
        ax.axvline(0.0, linewidth=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
