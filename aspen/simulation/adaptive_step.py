from __future__ import annotations

"""Adaptive free-streaming step length for ASPEN particles.

This module intentionally excludes electromagnetic forces. A particle moves
along its current velocity direction during each step.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from aspen.collisions.collision_frequency import speed_from_energy
from aspen.collisions.cross_section import (
    DEFAULT_CROSS_SECTION_DIR,
    REACTIONS,
    TARGETS,
    cross_section,
    normalize_projectile,
    normalize_reaction,
    normalize_target,
)
from aspen.collisions.particle_collision import apply_random_collision_to_particle
from aspen.neutral_density_model import neutral_density, neutral_density_xyz
from aspen.particle_initialization import Particle

from .particle_state import (
    particle_energy_ev,
    reset_collision_random_number,
    should_stop_tracing,
)


def _validate_random_number(random_number: float) -> float:
    r = float(random_number)
    if not 0.0 < r < 1.0:
        raise ValueError("R must be in the open interval (0, 1) for -lambda * ln(R).")
    return r


def _as_positive_finite(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return out


def collision_coefficient(
    density_m3: Mapping[str, float],
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return alpha = sum_j n_j sum_k sigma_j,k in m^-1.

    The supported target list is currently controlled by the packaged
    cross-section tables, `CO2`, `O`, and `N2`. `O2` density exists in the
    neutral model, but O2 collision cross sections are not packaged yet.
    """
    projectile_key = normalize_projectile(projectile)
    target_keys = tuple(normalize_target(target) for target in targets)
    reaction_keys = tuple(normalize_reaction(reaction) for reaction in reactions)

    alpha_total = 0.0
    breakdown = []
    for target_key in target_keys:
        if target_key not in density_m3:
            raise KeyError(f"density_m3 is missing target {target_key!r}.")
        n_m3 = float(density_m3[target_key])
        sigma_sum = 0.0
        reaction_breakdown = []
        for reaction_key in reaction_keys:
            cs = cross_section(
                projectile=projectile_key,
                target=target_key,
                reaction=reaction_key,
                energy_ev=energy_ev,
                cross_section_dir=cross_section_dir,
            )
            sigma_m2 = float(cs["sigma_m2"])
            sigma_sum += sigma_m2
            reaction_breakdown.append(
                {
                    "reaction": reaction_key,
                    "sigma_m2": sigma_m2,
                    "energy_loss_ev": float(cs["energy_loss_ev"]),
                }
            )

        alpha_target = n_m3 * sigma_sum
        alpha_total += alpha_target
        breakdown.append(
            {
                "target": target_key,
                "density_m-3": n_m3,
                "sigma_sum_m2": sigma_sum,
                "alpha_m-1": alpha_target,
                "reactions": reaction_breakdown,
            }
        )

    return {
        "projectile": projectile_key,
        "energy_ev": float(energy_ev),
        "collision_coefficient_m-1": float(alpha_total),
        "targets": target_keys,
        "reactions": reaction_keys,
        "breakdown": breakdown,
    }


def mean_free_path(
    density_m3: Mapping[str, float],
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Compute local mean free path from target densities and cross sections.

    lambda(r, E) = 1 / [sum_j n_j(r) sum_k sigma_j,k(E)].
    """
    coefficient = collision_coefficient(
        density_m3=density_m3,
        projectile=projectile,
        energy_ev=energy_ev,
        targets=targets,
        reactions=reactions,
        cross_section_dir=cross_section_dir,
    )
    alpha = float(coefficient["collision_coefficient_m-1"])
    lambda_m = np.inf if alpha <= 0.0 else 1.0 / alpha
    return {
        **coefficient,
        "mean_free_path_m": float(lambda_m),
        "mean_free_path_km": float(lambda_m / 1000.0),
    }


def mean_free_path_at_position(
    lon_deg: float,
    lat_deg: float,
    altitude_km: float,
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
) -> dict[str, object]:
    """Compute mean free path using the packaged MGITM plus MAMPS atmosphere."""
    target_keys = tuple(normalize_target(target) for target in targets)
    density = neutral_density(
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        altitude_km=altitude_km,
        species=target_keys,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
    )
    out = mean_free_path(
        density_m3=density,
        projectile=projectile,
        energy_ev=energy_ev,
        targets=target_keys,
        reactions=reactions,
        cross_section_dir=cross_section_dir,
    )
    return {
        **out,
        "lon_deg": float(lon_deg),
        "lat_deg": float(lat_deg),
        "altitude_km": float(altitude_km),
        "solar": solar,
        "ls": int(str(ls).strip().lower().replace("ls", "")),
        "include_hot_o": bool(include_hot_o),
        "neutral_density_m-3": density,
    }


def mean_free_path_at_xyz(
    Pmso_xyz: object,
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
) -> dict[str, object]:
    """Compute mean free path at a Mars-centered Cartesian MSO position in m."""
    target_keys = tuple(normalize_target(target) for target in targets)
    position_m = np.asarray(Pmso_xyz, dtype=float)
    if position_m.shape != (3,):
        raise ValueError("Pmso_xyz must be a 3-element position vector in m.")
    density = neutral_density_xyz(
        position_m,
        species=target_keys,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
        position_unit="m",
    )
    out = mean_free_path(
        density_m3=density,
        projectile=projectile,
        energy_ev=energy_ev,
        targets=target_keys,
        reactions=reactions,
        cross_section_dir=cross_section_dir,
    )
    return {
        **out,
        "Pmso_xyz_m": position_m.copy(),
        "solar": solar,
        "ls": int(str(ls).strip().lower().replace("ls", "")),
        "include_hot_o": bool(include_hot_o),
        "neutral_density_m-3": density,
    }


def sample_collision_distance(mean_free_path_m: float, R: float) -> float:
    """Return sampled distance to collision, s = -lambda ln(R), in meters."""
    lambda_m = _as_positive_finite(mean_free_path_m, "mean_free_path_m")
    r = _validate_random_number(R)
    return float(-lambda_m * np.log(r))


def adaptive_step_length(
    mean_free_path_m: float,
    R: float,
    safety_factor: float = 1.0,
    min_step_m: float = 0.0,
    max_step_m: float | None = 1000.0,
) -> dict[str, float]:
    """Return an adaptive free-streaming step length.

    `s = -lambda ln(R)` is the sampled collision distance for a local
    exponential process. By default `dl = s`. For slowly varying atmosphere,
    use `safety_factor < 1` to take a smaller transport step before recomputing
    the local mean free path. `max_step_m` limits very long high-altitude steps.
    """
    collision_distance_m = sample_collision_distance(mean_free_path_m, R)
    factor = _as_positive_finite(safety_factor, "safety_factor")
    dl_m = factor * collision_distance_m
    if max_step_m is not None:
        dl_m = min(dl_m, _as_positive_finite(max_step_m, "max_step_m"))
    if min_step_m > 0.0:
        dl_m = max(dl_m, _as_positive_finite(min_step_m, "min_step_m"))
    return {
        "mean_free_path_m": float(mean_free_path_m),
        "collision_distance_m": collision_distance_m,
        "dl_m": float(dl_m),
        "safety_factor": float(factor),
        "min_step_m": float(min_step_m),
        "max_step_m": float(np.inf if max_step_m is None else max_step_m),
    }


def advance_particle_xyz_step(
    particle: Particle,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    safety_factor: float = 0.2,
    min_step_m: float = 0.0,
    max_step_m: float = 1000.0,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Advance one particle step using Cartesian MSO position in m.

    The cumulative collision variable is optical depth, tau = integral alpha ds.
    A collision is reached when tau >= -ln(R). No electromagnetic acceleration
    is included, so velocity and energy are unchanged during this free-streaming
    step.
    """
    energy_ev = particle_energy_ev(particle)
    mfp = mean_free_path_at_xyz(
        Pmso_xyz=particle.position,
        projectile=particle.projectile,
        energy_ev=energy_ev,
        targets=targets,
        reactions=reactions,
        cross_section_dir=cross_section_dir,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
    )
    alpha = float(mfp["collision_coefficient_m-1"])
    threshold_tau = -np.log(_validate_random_number(float(particle.R)))

    velocity = np.asarray(particle.velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed <= 0.0 or not np.isfinite(speed):
        raise ValueError("particle.velocity must have a positive finite speed.")
    direction = velocity / speed

    old_position = np.asarray(particle.position, dtype=float).copy()
    old_tau = float(particle.cumulative_collision_frequency)
    remaining_tau = threshold_tau - old_tau
    collided = remaining_tau <= 0.0

    if alpha <= 0.0 or not np.isfinite(alpha):
        dl_m = float(max_step_m)
        if min_step_m > 0.0:
            dl_m = max(dl_m, float(min_step_m))
        mean_free_path_m = np.inf
    elif collided:
        dl_m = 0.0
        mean_free_path_m = 1.0 / alpha
    else:
        mean_free_path_m = 1.0 / alpha
        candidate_step = float(safety_factor) * mean_free_path_m
        candidate_step = min(candidate_step, float(max_step_m))
        if min_step_m > 0.0:
            candidate_step = max(candidate_step, float(min_step_m))
        collision_step = remaining_tau / alpha
        dl_m = min(candidate_step, collision_step)
        collided = dl_m >= collision_step

    displacement = direction * dl_m
    dt_s = dl_m / speed
    particle.position = old_position + displacement
    particle.local_collision_frequency = speed * alpha
    particle.cumulative_collision_frequency = old_tau + alpha * dl_m
    particle.iteration += 1

    new_energy_ev = particle_energy_ev(particle)
    return {
        "particle": particle,
        "iteration": particle.iteration,
        "old_position_m": old_position,
        "new_position_m": particle.position.copy(),
        "displacement_m": displacement,
        "velocity_m_s": velocity.copy(),
        "dl_m": float(dl_m),
        "dt_s": float(dt_s),
        "speed_m_s": speed,
        "energy_before_ev": float(energy_ev),
        "energy_after_ev": float(new_energy_ev),
        "energy_loss_ev": float(energy_ev - new_energy_ev),
        "charge_state": int(particle.charge_state),
        "projectile": particle.projectile,
        "collision_coefficient_m-1": alpha,
        "mean_free_path_m": float(mean_free_path_m),
        "local_collision_frequency_s-1": particle.local_collision_frequency,
        "cumulative_tau_before": old_tau,
        "cumulative_tau_after": particle.cumulative_collision_frequency,
        "threshold_tau": float(threshold_tau),
        "collided": bool(collided),
        "mean_free_path": mfp,
    }


def advance_particle_until_collision_xyz(
    particle: Particle,
    targets: tuple[str, ...] = TARGETS,
    safety_factor: float = 0.2,
    min_step_m: float = 0.0,
    max_step_m: float = 1000.0,
    max_iterations: int = 100_000,
    rng: np.random.Generator | None = None,
    reset_after_collision: bool = False,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
    stop_at_domain_boundary: bool = False,
    min_altitude_km: float = 100.0,
    max_altitude_km: float = 1000.0,
) -> dict[str, object]:
    """Advance one Cartesian particle until the sampled collision occurs."""
    history = []
    for _ in range(int(max_iterations)):
        step = advance_particle_xyz_step(
            particle=particle,
            targets=targets,
            safety_factor=safety_factor,
            min_step_m=min_step_m,
            max_step_m=max_step_m,
            solar=solar,
            ls=ls,
            include_hot_o=include_hot_o,
            cross_section_dir=cross_section_dir,
        )
        history.append(step)
        if stop_at_domain_boundary:
            stop_check = should_stop_tracing(
                particle,
                min_energy_ev=-np.inf,
                min_altitude_km=min_altitude_km,
                max_altitude_km=max_altitude_km,
            )
            if stop_check["out_of_domain"]:
                return {
                    "particle": particle,
                    "collided": False,
                    "step_history": history,
                    "collision": None,
                    "stop_condition": stop_check,
                }
        if step["collided"]:
            density = step["mean_free_path"]["neutral_density_m-3"]
            energy_before = particle_energy_ev(particle)
            velocity_before = np.asarray(particle.velocity, dtype=float).copy()
            random_before_collision = float(particle.R)
            threshold_tau_before_collision = float(step["threshold_tau"])
            collision = apply_random_collision_to_particle(
                particle=particle,
                density_m3=density,
                rng=rng,
                targets=targets,
                cross_section_dir=cross_section_dir,
                collision_position=particle.position,
                increment_iteration=True,
            )
            energy_after = particle_energy_ev(particle)
            collision["energy_before_ev"] = float(energy_before)
            collision["energy_after_ev"] = float(energy_after)
            collision["energy_loss_ev"] = float(energy_before - energy_after)
            collision["velocity_before_m_s"] = velocity_before
            collision["velocity_after_m_s"] = np.asarray(particle.velocity, dtype=float).copy()
            collision["charge_state"] = int(particle.charge_state)
            collision["position_m"] = np.asarray(particle.position, dtype=float).copy()
            collision["random_before_collision"] = random_before_collision
            collision["threshold_tau_before_collision"] = threshold_tau_before_collision
            if reset_after_collision:
                collision["random_after_collision"] = reset_collision_random_number(
                    particle,
                    rng=rng,
                )
            else:
                collision["random_after_collision"] = float(particle.R)
            return {
                "particle": particle,
                "collided": True,
                "step_history": history,
                "collision": collision,
            }
    return {
        "particle": particle,
        "collided": False,
        "step_history": history,
        "collision": None,
        "stop_condition": should_stop_tracing(
            particle,
            min_energy_ev=-np.inf,
            min_altitude_km=min_altitude_km,
            max_altitude_km=max_altitude_km,
        ),
    }


def advance_particle_free_streaming(
    particle: Particle,
    energy_ev: float,
    lon_deg: float,
    lat_deg: float,
    altitude_km: float,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    safety_factor: float = 1.0,
    min_step_m: float = 0.0,
    max_step_m: float | None = 1000.0,
    position_unit: str = "m",
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Move one particle by `dl` along its current velocity direction.

    No electromagnetic acceleration is included. New code should use
    `advance_particle_xyz_step`, where particle position is always in m.
    """
    mfp = mean_free_path_at_position(
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        altitude_km=altitude_km,
        projectile=particle.projectile,
        energy_ev=energy_ev,
        targets=targets,
        reactions=reactions,
        cross_section_dir=cross_section_dir,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
    )
    step = adaptive_step_length(
        mean_free_path_m=float(mfp["mean_free_path_m"]),
        R=float(particle.R),
        safety_factor=safety_factor,
        min_step_m=min_step_m,
        max_step_m=max_step_m,
    )

    velocity = np.asarray(particle.velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed <= 0.0 or not np.isfinite(speed):
        speed = speed_from_energy(energy_ev)
        direction = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 0.0 else np.array([1.0, 0.0, 0.0])
    else:
        direction = velocity / speed
    dt_s = float(step["dl_m"] / speed)

    unit = position_unit.strip().lower()
    if unit == "m":
        displacement = direction * step["dl_m"]
    elif unit == "km":
        displacement = direction * (step["dl_m"] / 1000.0)
    else:
        raise ValueError("position_unit must be 'm' or 'km'.")

    old_position = np.asarray(particle.position, dtype=float).copy()
    particle.position = old_position + displacement
    particle.local_collision_frequency = speed * float(mfp["collision_coefficient_m-1"])
    particle.cumulative_collision_frequency += particle.local_collision_frequency * dt_s

    return {
        "particle": particle,
        "old_position": old_position,
        "new_position": particle.position.copy(),
        "displacement": displacement,
        "dt_s": dt_s,
        "speed_m_s": speed,
        "position_unit": unit,
        **step,
        "mean_free_path": mfp,
    }
