from __future__ import annotations

"""H Ly-alpha related excitation channel.

For neutral H impact, the projectile remains neutral. For H+ impact, this
model follows the ASPEN reaction notation used here and returns a neutral
H-ENA after the Ly-alpha producing channel.
"""

from pathlib import Path

import numpy as np

from ._reaction_common import chemical_collision_after_scattering, normalize_charge_state
from .cross_section import DEFAULT_CROSS_SECTION_DIR


def lyman_alpha_emission(
    projectile: str,
    target: str,
    projectile_velocity_m_s: np.ndarray | list[float] | tuple[float, float, float],
    target_density_m3: float,
    scattering_angle: float,
    charge_state: str | int | float,
    angle_unit: str = "deg",
    azimuth_angle: float = 0.0,
    azimuth_unit: str = "deg",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return the post-reaction state for the Ly-alpha emission channel."""
    _ = projectile
    before = normalize_charge_state(charge_state)
    return chemical_collision_after_scattering(
        reaction="lya",
        target=target,
        projectile_velocity_m_s=projectile_velocity_m_s,
        target_density_m3=target_density_m3,
        scattering_angle=scattering_angle,
        charge_state=before,
        charge_state_after="neutral",
        angle_unit=angle_unit,
        azimuth_angle=azimuth_angle,
        azimuth_unit=azimuth_unit,
        cross_section_dir=cross_section_dir,
    )
