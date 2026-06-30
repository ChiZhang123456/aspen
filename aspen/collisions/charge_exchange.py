from __future__ import annotations

"""Charge exchange reaction for H+ impact.

Example reaction: `H+ + O -> H + O+`. The projectile changes from ion to
neutral H-ENA.
"""

from pathlib import Path

import numpy as np

from ._reaction_common import chemical_collision_after_scattering, normalize_charge_state
from .cross_section import DEFAULT_CROSS_SECTION_DIR


def charge_exchange(
    projectile: str,
    target: str,
    projectile_velocity_m_s: np.ndarray | list[float] | tuple[float, float, float],
    target_density_m3: float,
    scattering_angle: float,
    charge_state: str | int | float = "ion",
    angle_unit: str = "deg",
    azimuth_angle: float = 0.0,
    azimuth_unit: str = "deg",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return the post-reaction state for H+ charge exchange."""
    _ = projectile
    if normalize_charge_state(charge_state) != "ion":
        raise ValueError("charge_exchange requires charge_state='ion'.")
    return chemical_collision_after_scattering(
        reaction="state_change",
        target=target,
        projectile_velocity_m_s=projectile_velocity_m_s,
        target_density_m3=target_density_m3,
        scattering_angle=scattering_angle,
        charge_state="ion",
        charge_state_after="neutral",
        angle_unit=angle_unit,
        azimuth_angle=azimuth_angle,
        azimuth_unit=azimuth_unit,
        cross_section_dir=cross_section_dir,
    )
