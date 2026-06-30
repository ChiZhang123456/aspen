from __future__ import annotations

"""Two-body elastic collision kinematics for H/H+ and neutral targets.

The target particle is assumed initially at rest. The Monte Carlo model
provides a projectile LAB-frame scattering angle. Given that angle, momentum
and kinetic-energy conservation determine the post-collision projectile speed,
projectile velocity vector, target recoil velocity, and energy transfer.

This file only handles elastic kinematics. It does not decide whether an
elastic collision occurs; that decision belongs to `collision_sampler.py`.
"""

import numpy as np

from aspen.constants import ELEMENTARY_CHARGE_C

from .cross_section import DEFAULT_CROSS_SECTION_DIR, cross_section, normalize_projectile, normalize_target


AMU_KG = 1.66053906660e-27
MASS_KG = {
    "H": 1.00782503223 * AMU_KG,
    "H+": 1.007276466621 * AMU_KG,
    "CO2": 44.0095 * AMU_KG,
    "O": 15.999 * AMU_KG,
    "N2": 28.0134 * AMU_KG,
}


def _normalize_particle_mass_name(name: str) -> str:
    """Normalize projectiles and neutral targets before mass lookup."""
    if name in ("H", "H+", "Hplus", "proton", "H-ENA", "ENA"):
        return normalize_projectile(name)
    return normalize_target(name)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a stable local basis around the incoming velocity direction."""
    e0 = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(e0, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e0, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return e0, e1, e2


def elastic_collision_after_scattering(
    projectile: str,
    target: str,
    projectile_velocity_m_s: np.ndarray | list[float] | tuple[float, float, float],
    scattering_angle: float,
    charge_state: str | int | float | None = None,
    target_density_m3: float | None = None,
    angle_unit: str = "deg",
    azimuth_angle: float = 0.0,
    azimuth_unit: str = "deg",
    cross_section_dir: str = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Compute projectile velocity and energy after a two-body elastic collision.

    The target neutral is assumed initially at rest. The input scattering angle is
    the projectile LAB-frame deflection angle. The azimuth angle selects the
    scattering plane around the incoming velocity direction.
    """
    if charge_state is None:
        projectile_name = _normalize_particle_mass_name(projectile)
        charge_state_before = "ion" if projectile_name == "H+" else "neutral"
    else:
        from ._reaction_common import normalize_charge_state, projectile_from_charge_state

        charge_state_before = normalize_charge_state(charge_state)
        projectile_name = projectile_from_charge_state(charge_state_before)
    target_name = _normalize_particle_mass_name(target)
    m1 = MASS_KG[projectile_name]
    m2 = MASS_KG[target_name]

    v0 = np.asarray(projectile_velocity_m_s, dtype=np.float64)
    speed0 = float(np.linalg.norm(v0))
    if speed0 <= 0.0:
        raise ValueError("projectile_velocity_m_s must have nonzero magnitude.")

    theta = np.deg2rad(scattering_angle) if angle_unit == "deg" else float(scattering_angle)
    phi = np.deg2rad(azimuth_angle) if azimuth_unit == "deg" else float(azimuth_angle)
    mass_ratio = m1 / m2

    # LAB-frame two-body elastic kinematics for target initially at rest.
    discriminant = 1.0 - (mass_ratio * np.sin(theta)) ** 2
    if discriminant < -1.0e-12:
        raise ValueError(
            "This LAB scattering angle is not kinematically allowed for the mass ratio."
        )
    discriminant = max(discriminant, 0.0)

    speed_ratio = (
        mass_ratio * np.cos(theta) + np.sqrt(discriminant)
    ) / (1.0 + mass_ratio)
    if speed_ratio < 0.0:
        speed_ratio = 0.0
    speed_after = speed0 * speed_ratio

    e0, e1, e2 = _orthonormal_basis(v0)
    # `theta` fixes the deflection from the initial velocity, and `phi` chooses
    # the scattering plane around that initial direction.
    direction_after = (
        np.cos(theta) * e0 + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )
    direction_after /= np.linalg.norm(direction_after)
    projectile_velocity_after = speed_after * direction_after

    target_velocity_after = (m1 / m2) * (v0 - projectile_velocity_after)
    energy_before_ev = 0.5 * m1 * speed0**2 / ELEMENTARY_CHARGE_C
    energy_after_ev = 0.5 * m1 * speed_after**2 / ELEMENTARY_CHARGE_C
    target_energy_after_ev = (
        0.5 * m2 * float(np.dot(target_velocity_after, target_velocity_after))
        / ELEMENTARY_CHARGE_C
    )
    cs = cross_section(projectile_name, target_name, "elastic", energy_before_ev, cross_section_dir)
    density = None if target_density_m3 is None else float(target_density_m3)

    return {
        "projectile_after": projectile_name,
        "target": target_name,
        "reaction": "elastic",
        "charge_state_after": charge_state_before,
        "target_density_m-3": density,
        "sigma_m2": float(cs["sigma_m2"]),
        "collision_coefficient_m-1": None if density is None else density * float(cs["sigma_m2"]),
        "scattering_angle_deg": float(np.rad2deg(theta)),
        "projectile_velocity_after_m_s": projectile_velocity_after,
        "target_velocity_after_m_s": target_velocity_after,
        "projectile_speed_after_m_s": speed_after,
        "projectile_energy_after_eV": float(energy_after_ev),
        "projectile_energy_loss_eV": float(energy_before_ev - energy_after_ev),
        "target_recoil_energy_eV": float(target_energy_after_ev),
    }
