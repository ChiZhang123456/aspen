from __future__ import annotations

"""Electron stripping reaction for fast neutral H.

Example reaction: `H + O -> H+ + O + e-`. The projectile changes from neutral
H to H+.
"""

from pathlib import Path

import numpy as np

from ._reaction_common import chemical_collision_after_scattering, normalize_charge_state
from .cross_section import DEFAULT_CROSS_SECTION_DIR


def electron_stripping(
    projectile: str,
    target: str,
    projectile_velocity_m_s: np.ndarray | list[float] | tuple[float, float, float],
    target_density_m3: float,
    scattering_angle: float,
    charge_state: str | int | float = "neutral",
    angle_unit: str = "deg",
    azimuth_angle: float = 0.0,
    azimuth_unit: str = "deg",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return the post-reaction state for electron stripping."""
    _ = projectile
    if normalize_charge_state(charge_state) != "neutral":
        raise ValueError("electron_stripping requires charge_state='neutral'.")
    return chemical_collision_after_scattering(
        reaction="state_change",
        target=target,
        projectile_velocity_m_s=projectile_velocity_m_s,
        target_density_m3=target_density_m3,
        scattering_angle=scattering_angle,
        charge_state="neutral",
        charge_state_after="ion",
        angle_unit=angle_unit,
        azimuth_angle=azimuth_angle,
        azimuth_unit=azimuth_unit,
        cross_section_dir=cross_section_dir,
    )


def electron_striping(*args, **kwargs) -> dict[str, object]:
    """Backward-compatible alias for the common misspelling."""
    return electron_stripping(*args, **kwargs)
