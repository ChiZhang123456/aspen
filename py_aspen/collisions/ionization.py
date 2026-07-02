from __future__ import annotations

"""Impact ionization of a neutral target by H or H+.

Example reactions are `H + O -> H + O+ + e-` and
`H+ + O -> H+ + O+ + e-`. The projectile charge state is unchanged.
"""

from pathlib import Path

import numpy as np

from ._reaction_common import chemical_collision_after_scattering
from .cross_section import DEFAULT_CROSS_SECTION_DIR


def ionization(
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
    """Return the post-reaction state for target impact ionization."""
    _ = projectile
    return chemical_collision_after_scattering(
        reaction="ionization",
        target=target,
        projectile_velocity_m_s=projectile_velocity_m_s,
        target_density_m3=target_density_m3,
        scattering_angle=scattering_angle,
        charge_state=charge_state,
        charge_state_after=charge_state,
        angle_unit=angle_unit,
        azimuth_angle=azimuth_angle,
        azimuth_unit=azimuth_unit,
        cross_section_dir=cross_section_dir,
    )
