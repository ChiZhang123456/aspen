from __future__ import annotations

"""Scattering-angle lookup for ASPEN elastic collisions.

The file `scattering_angle_distribution.txt` is treated as an inverse-CDF
table. A uniformly distributed random number in [0, 1] is interpolated onto
the table to obtain a LAB-frame scattering angle. The returned angle is then
used by `elastic_collision_after_scattering`.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cross_section import DEFAULT_CROSS_SECTION_DIR


@dataclass(frozen=True)
class ScatteringAngleTable:
    """Inverse-CDF table for LAB-frame scattering angles."""

    random_number: np.ndarray
    theta_lab_deg: np.ndarray
    source_file: Path


@lru_cache(maxsize=4)
def load_scattering_angle_table(
    filename: str | Path = DEFAULT_CROSS_SECTION_DIR / "scattering_angle_distribution.txt",
) -> ScatteringAngleTable:
    """Load and sort the inverse-CDF scattering-angle table."""
    path = Path(filename)
    data = np.loadtxt(path, comments="#", skiprows=7)
    good = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    random_number = np.clip(data[good, 0], 0.0, 1.0)
    theta_lab_deg = data[good, 1]
    order = np.argsort(random_number)
    return ScatteringAngleTable(
        random_number=random_number[order],
        theta_lab_deg=theta_lab_deg[order],
        source_file=path,
    )


def scattering_angle(
    random_number: float | np.ndarray,
    table_file: str | Path = DEFAULT_CROSS_SECTION_DIR / "scattering_angle_distribution.txt",
    unit: str = "deg",
) -> float | np.ndarray:
    """Return LAB-frame scattering angle from a random number.

    Parameters
    ----------
    random_number
        Scalar or array of values in [0, 1]. Values outside the interval are
        clipped to the valid range.
    unit
        `deg` for degrees or `rad` for radians.
    """
    table = load_scattering_angle_table(table_file)
    r = np.clip(np.asarray(random_number, dtype=np.float64), 0.0, 1.0)
    theta_deg = np.interp(r, table.random_number, table.theta_lab_deg)
    value = np.deg2rad(theta_deg) if unit == "rad" else theta_deg
    return float(value) if np.ndim(random_number) == 0 else value
