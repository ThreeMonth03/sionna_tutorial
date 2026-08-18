"""Loading and normalizing the standardized TDL/CDL profile tables."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import numpy as np

ModelLetter = Literal["A", "B", "C", "D", "E"]


@dataclass(frozen=True)
class TDLProfile:
    model: ModelLetter
    has_los: bool
    delays_s: np.ndarray
    powers_linear: np.ndarray
    k_factor_linear: float
    los_power_fraction: float

    @property
    def num_taps(self) -> int:
        return int(self.delays_s.size)


@dataclass(frozen=True)
class CDLProfile:
    model: ModelLetter
    has_los: bool
    delays_s: np.ndarray
    powers_linear: np.ndarray
    aod_rad: np.ndarray
    aoa_rad: np.ndarray
    zod_rad: np.ndarray
    zoa_rad: np.ndarray
    cluster_asd_deg: float
    cluster_asa_deg: float
    cluster_zsd_deg: float
    cluster_zsa_deg: float
    xpr_linear: float
    k_factor_linear: float
    los_angles_rad: tuple[float, float, float, float] | None

    @property
    def num_clusters(self) -> int:
        return int(self.delays_s.size)


def _load_json(kind: str, model: ModelLetter) -> dict[str, object]:
    if model not in {"A", "B", "C", "D", "E"}:
        raise ValueError("model must be one of A, B, C, D, or E")
    path = files("sionna_tutorial.data.models").joinpath("models.json.gz")
    decoded = gzip.decompress(path.read_bytes()).decode("utf-8")
    all_models = json.loads(decoded)
    return all_models[f"{kind}-{model}"]


def _normalize_db_powers(powers_db: np.ndarray) -> np.ndarray:
    powers = np.power(10.0, powers_db / 10.0)
    return powers / powers.sum()


def load_tdl_profile(model: ModelLetter, delay_spread_s: float = 100e-9) -> TDLProfile:
    """Load a TDL A-E profile and scale normalized delays by RMS delay spread."""

    if delay_spread_s <= 0:
        raise ValueError("delay_spread_s must be positive")
    raw = _load_json("TDL", model)
    delays = np.asarray(raw["delays"], dtype=np.float64) * delay_spread_s
    powers = _normalize_db_powers(np.asarray(raw["powers"], dtype=np.float64))
    has_los = bool(raw["los"])

    k_factor = 0.0
    los_fraction = 0.0
    if has_los:
        los_power = powers[0]
        diffuse = powers[1:]
        diffuse_total = float(diffuse.sum())
        k_factor = float(los_power / diffuse_total)
        los_fraction = float(los_power)
        powers = diffuse / diffuse_total
        delays = delays[1:]

    return TDLProfile(
        model=model,
        has_los=has_los,
        delays_s=delays,
        powers_linear=powers,
        k_factor_linear=k_factor,
        los_power_fraction=los_fraction,
    )


def load_cdl_profile(model: ModelLetter, delay_spread_s: float = 100e-9) -> CDLProfile:
    """Load CDL A-E, including angle spreads and LOS K-factor handling."""

    if delay_spread_s <= 0:
        raise ValueError("delay_spread_s must be positive")
    raw = _load_json("CDL", model)
    delays = np.asarray(raw["delays"], dtype=np.float64) * delay_spread_s
    powers = _normalize_db_powers(np.asarray(raw["powers"], dtype=np.float64))
    aod = np.deg2rad(np.asarray(raw["aod"], dtype=np.float64))
    aoa = np.deg2rad(np.asarray(raw["aoa"], dtype=np.float64))
    zod = np.deg2rad(np.asarray(raw["zod"], dtype=np.float64))
    zoa = np.deg2rad(np.asarray(raw["zoa"], dtype=np.float64))
    has_los = bool(raw["los"])

    k_factor = 0.0
    los_angles: tuple[float, float, float, float] | None = None
    if has_los:
        los_power = powers[0]
        diffuse = powers[1:]
        diffuse_total = float(diffuse.sum())
        k_factor = float(los_power / diffuse_total)
        powers = diffuse / diffuse_total
        delays = delays[1:]
        los_angles = (float(aod[0]), float(aoa[0]), float(zod[0]), float(zoa[0]))
        aod, aoa, zod, zoa = aod[1:], aoa[1:], zod[1:], zoa[1:]

    expected = int(raw["num_clusters"])
    arrays = (delays, powers, aod, aoa, zod, zoa)
    if any(array.size != expected for array in arrays):
        raise ValueError(f"CDL-{model} profile is internally inconsistent")

    return CDLProfile(
        model=model,
        has_los=has_los,
        delays_s=delays,
        powers_linear=powers,
        aod_rad=aod,
        aoa_rad=aoa,
        zod_rad=zod,
        zoa_rad=zoa,
        cluster_asd_deg=float(raw["cASD"]),
        cluster_asa_deg=float(raw["cASA"]),
        cluster_zsd_deg=float(raw["cZSD"]),
        cluster_zsa_deg=float(raw["cZSA"]),
        xpr_linear=float(10.0 ** (float(raw["xpr"]) / 10.0)),
        k_factor_linear=k_factor,
        los_angles_rad=los_angles,
    )
