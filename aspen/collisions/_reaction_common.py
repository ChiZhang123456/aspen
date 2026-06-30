from __future__ import annotations

"""Shared helpers for non-elastic ASPEN collision reactions."""

from pathlib import Path

import numpy as np

from aspen.constants import ELEMENTARY_CHARGE_C

from .cross_section import DEFAULT_CROSS_SECTION_DIR, cross_section, normalize_target
from .elastic_collision import MASS_KG, _orthonormal_basis


CHARGE_STATE_ALIASES = {
    "neutral": "neutral",
    "h": "neutral",
    "h-ena": "neutral",
    "ena": "neutral",
    "0": "neutral",
    "ion": "ion",
    "h+": "ion",
    "hplus": "ion",
    "proton": "ion",
    "+1": "ion",
    "1": "ion",
}


def normalize_charge_state(charge_state: str | int | float) -> str:
    """Normalize charge-state labels to `neutral` or `ion`."""
    key = str(charge_state).strip().lower()
    if key.endswith(".0"):
        key = key[:-2]
    if key not in CHARGE_STATE_ALIASES:
        raise ValueError("charge_state must be neutral, ion, H, H-ENA, H+, proton, 0, or 1.")
    return CHARGE_STATE_ALIASES[key]


def projectile_from_charge_state(charge_state: str | int | float) -> str:
    """Return the cross-section projectile name implied by the charge state."""
    state = normalize_charge_state(charge_state)
    return "H+" if state == "ion" else "H"


def _energy_from_velocity_ev(projectile: str, velocity_m_s: np.ndarray) -> float:
    mass = MASS_KG[projectile]
    speed = float(np.linalg.norm(velocity_m_s))
    return float(0.5 * mass * speed**2 / ELEMENTARY_CHARGE_C)


def _velocity_after_energy_loss(
    velocity_before_m_s: np.ndarray,
    speed_after_m_s: float,
    scattering_angle: float,
    angle_unit: str,
    azimuth_angle: float,
    azimuth_unit: str,
) -> tuple[np.ndarray, float]:
    theta = np.deg2rad(scattering_angle) if angle_unit == "deg" else float(scattering_angle)
    phi = np.deg2rad(azimuth_angle) if azimuth_unit == "deg" else float(azimuth_angle)
    e0, e1, e2 = _orthonormal_basis(velocity_before_m_s)
    direction_after = (
        np.cos(theta) * e0 + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )
    direction_after /= np.linalg.norm(direction_after)
    return speed_after_m_s * direction_after, float(theta)


def chemical_collision_after_scattering(
    reaction: str,
    target: str,
    projectile_velocity_m_s: np.ndarray | list[float] | tuple[float, float, float],
    target_density_m3: float,
    scattering_angle: float,
    charge_state: str | int | float,
    charge_state_after: str | int | float,
    angle_unit: str = "deg",
    azimuth_angle: float = 0.0,
    azimuth_unit: str = "deg",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Apply tabulated non-elastic energy loss plus sampled scattering angle."""
    projectile_before = projectile_from_charge_state(charge_state)
    target_name = normalize_target(target)
    velocity_before = np.asarray(projectile_velocity_m_s, dtype=np.float64)
    speed_before = float(np.linalg.norm(velocity_before))
    if speed_before <= 0.0:
        raise ValueError("projectile_velocity_m_s must have nonzero magnitude.")

    energy_before_ev = _energy_from_velocity_ev(projectile_before, velocity_before)
    cs = cross_section(projectile_before, target_name, reaction, energy_before_ev, cross_section_dir)
    energy_loss_ev = min(float(cs["energy_loss_ev"]), energy_before_ev)
    energy_after_ev = max(energy_before_ev - energy_loss_ev, 0.0)

    projectile_after = projectile_from_charge_state(charge_state_after)
    speed_after = float(
        np.sqrt(2.0 * energy_after_ev * ELEMENTARY_CHARGE_C / MASS_KG[projectile_after])
    )
    velocity_after, theta = _velocity_after_energy_loss(
        velocity_before,
        speed_after,
        scattering_angle,
        angle_unit,
        azimuth_angle,
        azimuth_unit,
    )

    density = float(target_density_m3)
    sigma_m2 = float(cs["sigma_m2"])
    return {
        "projectile_after": projectile_after,
        "target": target_name,
        "reaction": reaction,
        "charge_state_after": normalize_charge_state(charge_state_after),
        "target_density_m-3": density,
        "sigma_m2": sigma_m2,
        "collision_coefficient_m-1": density * sigma_m2,
        "scattering_angle_deg": float(np.rad2deg(theta)),
        "projectile_velocity_after_m_s": velocity_after,
        "projectile_speed_after_m_s": speed_after,
        "projectile_energy_after_eV": float(energy_after_ev),
        "projectile_energy_loss_eV": float(energy_before_ev - energy_after_ev),
        "reaction_energy_loss_eV": float(energy_loss_ev),
        "energy_deposit_eV": float(energy_loss_ev),
        "kinematic_model": "tabulated_energy_loss_plus_sampled_scattering_angle",
    }
