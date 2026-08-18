"""Antenna-array geometry and simplified element field patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constants import SPEED_OF_LIGHT

PatternName = Literal["omni", "tr38901"]


@dataclass(frozen=True)
class AntennaArray:
    """Positions and polarization slants for individual antenna ports.

    Positions are expressed in metres in a local Cartesian coordinate system.
    For dual polarization, each physical location occurs twice with different
    slant angles. This keeps the tensor dimensions explicit and easy to inspect.
    """

    positions_m: np.ndarray
    slant_angles_rad: np.ndarray
    pattern: PatternName = "omni"

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        slants = np.asarray(self.slant_angles_rad, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape [num_antennas, 3]")
        if slants.shape != (positions.shape[0],):
            raise ValueError("slant_angles_rad must have one value per antenna port")
        if self.pattern not in {"omni", "tr38901"}:
            raise ValueError("pattern must be 'omni' or 'tr38901'")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "slant_angles_rad", slants)

    @property
    def num_antennas(self) -> int:
        return int(self.positions_m.shape[0])

    @classmethod
    def ula(
        cls,
        num_elements: int,
        carrier_frequency_hz: float,
        spacing_wavelengths: float = 0.5,
        polarization: Literal["single", "dual"] = "single",
        pattern: PatternName = "omni",
    ) -> "AntennaArray":
        """Create a uniform linear array along the local y-axis."""

        if num_elements <= 0:
            raise ValueError("num_elements must be positive")
        wavelength = SPEED_OF_LIGHT / carrier_frequency_hz
        y = np.arange(num_elements, dtype=np.float64) * spacing_wavelengths * wavelength
        y -= y.mean()
        physical_positions = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=-1)
        return cls._with_polarization(physical_positions, polarization, pattern)

    @classmethod
    def upa(
        cls,
        num_rows: int,
        num_cols: int,
        carrier_frequency_hz: float,
        row_spacing_wavelengths: float = 0.5,
        col_spacing_wavelengths: float = 0.5,
        polarization: Literal["single", "dual"] = "single",
        pattern: PatternName = "omni",
    ) -> "AntennaArray":
        """Create a uniform planar array in the local y-z plane."""

        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows and num_cols must be positive")
        wavelength = SPEED_OF_LIGHT / carrier_frequency_hz
        y = np.arange(num_cols, dtype=np.float64) * col_spacing_wavelengths * wavelength
        z = np.arange(num_rows, dtype=np.float64) * row_spacing_wavelengths * wavelength
        y -= y.mean()
        z -= z.mean()
        yy, zz = np.meshgrid(y, z)
        physical_positions = np.stack(
            [np.zeros_like(yy), yy, zz],
            axis=-1,
        ).reshape(-1, 3)
        return cls._with_polarization(physical_positions, polarization, pattern)

    @classmethod
    def _with_polarization(
        cls,
        physical_positions: np.ndarray,
        polarization: Literal["single", "dual"],
        pattern: PatternName,
    ) -> "AntennaArray":
        if polarization == "single":
            return cls(
                positions_m=physical_positions,
                slant_angles_rad=np.zeros(physical_positions.shape[0]),
                pattern=pattern,
            )
        if polarization == "dual":
            positions = np.repeat(physical_positions, 2, axis=0)
            slants = np.tile(np.deg2rad(np.array([-45.0, 45.0])), physical_positions.shape[0])
            return cls(positions_m=positions, slant_angles_rad=slants, pattern=pattern)
        raise ValueError("polarization must be 'single' or 'dual'")


def element_field_components(
    zenith_rad: object,
    azimuth_rad: object,
    slant_angles_rad: object,
    *,
    pattern: PatternName,
    xp: object,
) -> object:
    """Return ``[F_theta, F_phi]`` for every ray and antenna port.

    The ``tr38901`` option implements the normalized single-element attenuation
    shape from TR 38.901. Absolute 8 dBi element gain is deliberately omitted,
    because this tutorial normalizes small-scale channel power.
    """

    zenith = xp.asarray(zenith_rad)[..., None]
    azimuth = xp.asarray(azimuth_rad)[..., None]
    slant = xp.asarray(slant_angles_rad)[None, None, :]

    if pattern == "omni":
        amplitude = xp.ones_like(zenith + azimuth)
    elif pattern == "tr38901":
        theta_deg = xp.rad2deg(zenith)
        phi_deg = (xp.rad2deg(azimuth) + 180.0) % 360.0 - 180.0
        vertical_loss_db = xp.minimum(12.0 * ((theta_deg - 90.0) / 65.0) ** 2, 30.0)
        horizontal_loss_db = xp.minimum(12.0 * (phi_deg / 65.0) ** 2, 30.0)
        total_loss_db = xp.minimum(vertical_loss_db + horizontal_loss_db, 30.0)
        amplitude = 10.0 ** (-total_loss_db / 20.0)
    else:  # pragma: no cover - protected by AntennaArray validation
        raise ValueError(f"unknown pattern: {pattern}")

    f_theta = amplitude * xp.cos(slant)
    f_phi = amplitude * xp.sin(slant)
    return xp.stack([f_theta, f_phi], axis=-1)
