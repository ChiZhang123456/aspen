from __future__ import annotations

"""Numba accelerated adaptive transport step for ASPEN particles.

The cumulative value compared with the particle random number is an
optical-depth-like integral, `tau = integral alpha ds`, where
`alpha = sum_j n_j sum_k sigma_j,k`. Existing particle fields keep the name
`cumulative_collision_frequency` for compatibility with earlier code.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from aspen_zc.constants import ELEMENTARY_CHARGE_C, MARS_RADIUS_KM
from aspen_zc.cross_sections.collision_sampler import SAMPLE_REACTIONS
from aspen_zc.cross_sections.cross_section import (
    DEFAULT_CROSS_SECTION_DIR,
    PROJECTILE_INDEX,
    REACTION_INDEX,
    TARGETS,
    TARGET_INDEX,
    load_cross_sections,
    normalize_target,
)
from aspen_zc.cross_sections.elastic_collision import MASS_KG
from aspen_zc.cross_sections.particle_collision import apply_random_collision_to_particle
from aspen_zc.neutral_density_model import neutral_density
from aspen_zc.particle_initialization import Particle


H_MASS_KG = float(MASS_KG["H"])
HPLUS_MASS_KG = float(MASS_KG["H+"])

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]
        return lambda func: func


@njit(cache=True)
def _interp1d_numba(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    if x <= xp[0]:
        return fp[0]
    n = xp.size
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    dx = xp[hi] - xp[lo]
    if dx <= 0.0:
        return fp[lo]
    w = (x - xp[lo]) / dx
    return fp[lo] * (1.0 - w) + fp[hi] * w


@njit(cache=True)
def _energy_from_velocity_numba(velocity_m_s: np.ndarray, charge_state: int) -> float:
    mass = HPLUS_MASS_KG if charge_state == 1 else H_MASS_KG
    speed2 = (
        velocity_m_s[0] * velocity_m_s[0]
        + velocity_m_s[1] * velocity_m_s[1]
        + velocity_m_s[2] * velocity_m_s[2]
    )
    return 0.5 * mass * speed2 / ELEMENTARY_CHARGE_C


@njit(cache=True)
def _speed_numba(velocity_m_s: np.ndarray) -> float:
    return np.sqrt(
        velocity_m_s[0] * velocity_m_s[0]
        + velocity_m_s[1] * velocity_m_s[1]
        + velocity_m_s[2] * velocity_m_s[2]
    )


@njit(cache=True)
def _collision_alpha_numba(
    density_m3: np.ndarray,
    charge_state: int,
    energy_ev: float,
    energy_grid_ev: np.ndarray,
    sigma_m2: np.ndarray,
    target_indices: np.ndarray,
    reaction_indices: np.ndarray,
) -> float:
    projectile_index = 0 if charge_state == 1 else 1
    alpha = 0.0
    for j in range(target_indices.size):
        sigma_sum = 0.0
        target_index = target_indices[j]
        for k in range(reaction_indices.size):
            reaction_index = reaction_indices[k]
            sigma_sum += _interp1d_numba(
                energy_ev,
                energy_grid_ev,
                sigma_m2[projectile_index, target_index, reaction_index, :],
            )
        alpha += density_m3[j] * sigma_sum
    return alpha


@njit(cache=True)
def _advance_until_threshold_step_numba(
    position: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_state: int,
    cumulative_tau: float,
    threshold_r: float,
    density_m3: np.ndarray,
    energy_grid_ev: np.ndarray,
    sigma_m2: np.ndarray,
    target_indices: np.ndarray,
    reaction_indices: np.ndarray,
    safety_factor: float,
    min_step_m: float,
    max_step_m: float,
    position_scale_per_m: float,
) -> tuple[np.ndarray, float, float, float, float, float, float, int]:
    energy_ev = _energy_from_velocity_numba(velocity_m_s, charge_state)
    alpha = _collision_alpha_numba(
        density_m3,
        charge_state,
        energy_ev,
        energy_grid_ev,
        sigma_m2,
        target_indices,
        reaction_indices,
    )
    speed = _speed_numba(velocity_m_s)
    if alpha <= 0.0 or speed <= 0.0:
        return position.copy(), energy_ev, alpha, 0.0, 0.0, 0.0, cumulative_tau, 0

    mean_free_path_m = 1.0 / alpha
    step_m = safety_factor * mean_free_path_m
    if max_step_m > 0.0 and step_m > max_step_m:
        step_m = max_step_m
    if min_step_m > 0.0 and step_m < min_step_m:
        step_m = min_step_m

    remaining_tau = threshold_r - cumulative_tau
    collided = 0
    if remaining_tau <= 0.0:
        step_m = 0.0
        collided = 1
    else:
        tau_step = alpha * step_m
        if tau_step >= remaining_tau:
            step_m = remaining_tau / alpha
            collided = 1

    new_position = position.copy()
    if step_m > 0.0:
        new_position[0] += velocity_m_s[0] / speed * step_m * position_scale_per_m
        new_position[1] += velocity_m_s[1] / speed * step_m * position_scale_per_m
        new_position[2] += velocity_m_s[2] / speed * step_m * position_scale_per_m
    dt_s = step_m / speed
    local_frequency = speed * alpha
    cumulative_new = cumulative_tau + alpha * step_m
    return (
        new_position,
        energy_ev,
        alpha,
        local_frequency,
        mean_free_path_m,
        step_m,
        cumulative_new,
        collided,
    )


def _density_array(density_m3: Mapping[str, float], targets: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(density_m3[target]) for target in targets], dtype=np.float64)


def _target_indices(targets: tuple[str, ...]) -> np.ndarray:
    return np.asarray([TARGET_INDEX[normalize_target(target)] for target in targets], dtype=np.int64)


def _reaction_indices() -> np.ndarray:
    return np.asarray([REACTION_INDEX[reaction] for reaction in SAMPLE_REACTIONS], dtype=np.int64)


def advance_particle_numba_step(
    particle: Particle,
    density_m3: Mapping[str, float],
    targets: tuple[str, ...] = TARGETS,
    safety_factor: float = 0.2,
    min_step_m: float = 0.0,
    max_step_m: float | None = None,
    position_unit: str = "km",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Advance one particle by one adaptive numba-computed step."""
    target_keys = tuple(normalize_target(target) for target in targets)
    cs = load_cross_sections(str(Path(cross_section_dir)))
    scale = 1.0 if position_unit == "m" else 1.0 / 1000.0
    max_step = -1.0 if max_step_m is None else float(max_step_m)
    old_position = np.asarray(particle.position, dtype=np.float64).copy()
    out = _advance_until_threshold_step_numba(
        old_position,
        np.asarray(particle.velocity, dtype=np.float64),
        int(particle.charge_state),
        float(particle.cumulative_collision_frequency),
        float(particle.R),
        _density_array(density_m3, target_keys),
        cs.energy_ev,
        cs.sigma_m2,
        _target_indices(target_keys),
        _reaction_indices(),
        float(safety_factor),
        float(min_step_m),
        max_step,
        scale,
    )
    (
        new_position,
        energy_ev,
        alpha,
        local_frequency,
        mean_free_path_m,
        step_m,
        cumulative_new,
        collided,
    ) = out

    particle.position = new_position
    particle.local_collision_frequency = float(local_frequency)
    particle.cumulative_collision_frequency = float(cumulative_new)
    particle.iteration += 1
    return {
        "particle": particle,
        "old_position": old_position,
        "new_position": new_position.copy(),
        "energy_ev": float(energy_ev),
        "collision_coefficient_m-1": float(alpha),
        "local_collision_frequency_s-1": float(local_frequency),
        "mean_free_path_m": float(mean_free_path_m),
        "step_m": float(step_m),
        "cumulative_collision_frequency": float(cumulative_new),
        "threshold_R": float(particle.R),
        "collided": bool(collided),
        "position_unit": position_unit,
    }


def advance_particle_until_collision_numba(
    particle: Particle,
    density_m3: Mapping[str, float],
    targets: tuple[str, ...] = TARGETS,
    safety_factor: float = 0.2,
    min_step_m: float = 0.0,
    max_step_m: float | None = None,
    position_unit: str = "km",
    max_iterations: int = 10_000,
    rng: np.random.Generator | None = None,
    reset_after_collision: bool = True,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Iterate adaptive numba steps until cumulative tau exceeds particle.R."""
    history = []
    for _ in range(int(max_iterations)):
        step = advance_particle_numba_step(
            particle=particle,
            density_m3=density_m3,
            targets=targets,
            safety_factor=safety_factor,
            min_step_m=min_step_m,
            max_step_m=max_step_m,
            position_unit=position_unit,
            cross_section_dir=cross_section_dir,
        )
        history.append(step)
        if step["collided"]:
            cumulative_before_reset = particle.cumulative_collision_frequency
            collision = apply_random_collision_to_particle(
                particle=particle,
                density_m3=density_m3,
                rng=rng,
                targets=targets,
                cross_section_dir=cross_section_dir,
                collision_position=particle.position,
                increment_iteration=False,
            )
            if reset_after_collision:
                generator = rng if rng is not None else particle.rng
                if generator is None:
                    generator = np.random.default_rng()
                particle.cumulative_collision_frequency = 0.0
                particle.R = float(generator.random())
            return {
                "particle": particle,
                "collided": True,
                "step_history": history,
                "collision": collision,
                "cumulative_before_reset": float(cumulative_before_reset),
            }
    return {"particle": particle, "collided": False, "step_history": history}


def advance_particle_until_collision_from_position_numba(
    particle: Particle,
    targets: tuple[str, ...] = TARGETS,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    position_format: str = "cartesian_km",
    **kwargs,
) -> dict[str, object]:
    """Iterate to collision while recomputing density from particle.position.

    `position_format='cartesian_km'` means Mars-centered Cartesian position in
    km. `position_format='lon_lat_alt_km'` means `[lon_deg, lat_deg,
    altitude_km]`, useful only for local column tests because longitude and
    latitude are then not updated geometrically.
    """
    target_keys = tuple(normalize_target(target) for target in targets)
    max_iterations = int(kwargs.pop("max_iterations", 10_000))
    rng = kwargs.pop("rng", None)
    reset_after_collision = bool(kwargs.pop("reset_after_collision", True))
    history = []
    last_density = None
    for _ in range(max_iterations):
        lon_deg, lat_deg, altitude_km = position_to_lon_lat_alt(
            particle.position,
            position_format=position_format,
        )
        density = neutral_density(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            altitude_km=altitude_km,
            species=target_keys,
            solar=solar,
            ls=ls,
            include_hot_o=include_hot_o,
        )
        last_density = density
        step = advance_particle_numba_step(
            particle=particle,
            density_m3=density,
            targets=target_keys,
            **kwargs,
        )
        step["lon_deg"] = lon_deg
        step["lat_deg"] = lat_deg
        step["altitude_km"] = altitude_km
        history.append(step)
        if step["collided"]:
            cumulative_before_reset = particle.cumulative_collision_frequency
            collision = apply_random_collision_to_particle(
                particle=particle,
                density_m3=density,
                rng=rng,
                targets=target_keys,
                collision_position=particle.position,
                increment_iteration=False,
            )
            if reset_after_collision:
                generator = rng if rng is not None else particle.rng
                if generator is None:
                    generator = np.random.default_rng()
                particle.cumulative_collision_frequency = 0.0
                particle.R = float(generator.random())
            return {
                "particle": particle,
                "collided": True,
                "step_history": history,
                "collision": collision,
                "neutral_density_m-3": density,
                "cumulative_before_reset": float(cumulative_before_reset),
            }
    return {
        "particle": particle,
        "collided": False,
        "step_history": history,
        "neutral_density_m-3": last_density,
    }


def position_to_lon_lat_alt(
    position: object,
    position_format: str = "cartesian_km",
) -> tuple[float, float, float]:
    """Convert a particle position vector to lon, lat, altitude in km."""
    pos = np.asarray(position, dtype=float)
    if pos.shape != (3,):
        raise ValueError("position must be a 3-element vector.")
    fmt = position_format.strip().lower()
    if fmt == "lon_lat_alt_km":
        return float(pos[0] % 360.0), float(pos[1]), float(pos[2])
    if fmt == "cartesian_km":
        radius = float(np.linalg.norm(pos))
        if radius <= 0.0:
            raise ValueError("cartesian_km position must have nonzero radius.")
        lon = float(np.degrees(np.arctan2(pos[1], pos[0])) % 360.0)
        lat = float(np.degrees(np.arcsin(pos[2] / radius)))
        altitude = radius - MARS_RADIUS_KM
        return lon, lat, float(altitude)
    raise ValueError("position_format must be 'cartesian_km' or 'lon_lat_alt_km'.")
