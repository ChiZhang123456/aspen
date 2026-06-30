from __future__ import annotations

"""Apply one sampled collision event to a `Particle`.

This is the high-level bridge between `particle_initialization.Particle` and
the cross-section reaction functions. It assumes a transport step has already
decided that a collision occurs at the particle's current location.
"""

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aspen_zc.constants import ELEMENTARY_CHARGE_C

from ._reaction_common import projectile_from_charge_state
from .charge_exchange import charge_exchange
from .collision_sampler import SAMPLE_REACTIONS, sample_collision_event
from .cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS
from .elastic_collision import MASS_KG, elastic_collision_after_scattering
from .electron_stripping import electron_stripping
from .ionization import ionization
from .lyman_alpha_emission import lyman_alpha_emission
from .scattering_angle import scattering_angle as scattering_angle_from_random


def _particle_energy_ev(particle: Any) -> float:
    projectile = projectile_from_charge_state(particle.charge_state)
    velocity = np.asarray(particle.velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    return float(0.5 * MASS_KG[projectile] * speed**2 / ELEMENTARY_CHARGE_C)


def _sampling_collision_coefficient(
    density_m3: Mapping[str, float],
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...],
    cross_section_dir: str | Path,
) -> dict[str, object]:
    from .cross_section import REACTION_INDEX, TARGET_INDEX, load_cross_sections, normalize_target

    target_keys = tuple(normalize_target(target) for target in targets)
    cs = load_cross_sections(str(Path(cross_section_dir)))
    projectile_index = 0 if projectile == "H+" else 1
    alpha_total = 0.0
    breakdown = []
    for target in target_keys:
        density = float(density_m3[target])
        target_index = TARGET_INDEX[target]
        sigma_sum = 0.0
        reaction_breakdown = []
        for reaction in SAMPLE_REACTIONS:
            reaction_index = REACTION_INDEX[reaction]
            sigma = float(np.interp(energy_ev, cs.energy_ev, cs.sigma_m2[projectile_index, target_index, reaction_index, :]))
            sigma_sum += sigma
            reaction_breakdown.append({"reaction": reaction, "sigma_m2": sigma})
        alpha = density * sigma_sum
        alpha_total += alpha
        breakdown.append(
            {
                "target": target,
                "density_m-3": density,
                "sigma_sum_m2": sigma_sum,
                "alpha_m-1": alpha,
                "reactions": reaction_breakdown,
            }
        )
    return {"collision_coefficient_m-1": alpha_total, "breakdown": breakdown}


def apply_random_collision_to_particle(
    particle: Any,
    density_m3: Mapping[str, float],
    random_target: float | None = None,
    random_reaction: float | None = None,
    random_scattering: float | None = None,
    rng: np.random.Generator | None = None,
    targets: tuple[str, ...] = TARGETS,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
    collision_position: object | None = None,
    elapsed_time_s: float | None = None,
    path_length_m: float | None = None,
    increment_iteration: bool = True,
) -> dict[str, object]:
    """Sample a target and reaction, then update one particle in place.

    Parameters
    ----------
    particle
        Particle from `aspen_zc.particle_initialization`.
    density_m3
        Local neutral density dictionary in m^-3.
    random_target, random_reaction, random_scattering
        Optional random numbers in [0, 1). Missing values are drawn from `rng`
        or the particle's own generator.
    collision_position
        Optional 3-vector. If supplied, the particle position is replaced by
        this collision position before returning.
    elapsed_time_s, path_length_m
        Optional accumulation controls. If `elapsed_time_s` is given, the
        cumulative collision frequency is incremented by `nu * dt`. If
        `path_length_m` is given, it is incremented by `alpha * ds`. If neither
        is given, the cumulative value is left unchanged.
    """
    generator = rng if rng is not None else particle.rng
    if generator is None:
        generator = np.random.default_rng()

    projectile = projectile_from_charge_state(particle.charge_state)
    energy_ev = _particle_energy_ev(particle)
    velocity = np.asarray(particle.velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))

    coefficient = _sampling_collision_coefficient(
        density_m3=density_m3,
        projectile=projectile,
        energy_ev=energy_ev,
        targets=targets,
        cross_section_dir=cross_section_dir,
    )
    alpha_total = float(coefficient["collision_coefficient_m-1"])
    local_frequency = speed * alpha_total
    particle.local_collision_frequency = local_frequency

    if elapsed_time_s is not None:
        particle.cumulative_collision_frequency += local_frequency * float(elapsed_time_s)
    elif path_length_m is not None:
        particle.cumulative_collision_frequency += alpha_total * float(path_length_m)

    event = sample_collision_event(
        density_m3=density_m3,
        projectile=projectile,
        energy_ev=energy_ev,
        random_target=random_target,
        random_reaction=random_reaction,
        rng=generator,
        targets=targets,
        cross_section_dir=cross_section_dir,
    )

    r_scatter = float(generator.random() if random_scattering is None else random_scattering)
    theta_deg = scattering_angle_from_random(r_scatter)
    target = str(event["target"])
    reaction = str(event["reaction"])
    density_target = float(density_m3[target])

    if reaction == "elastic":
        reaction_result = elastic_collision_after_scattering(
            projectile=projectile,
            target=target,
            projectile_velocity_m_s=particle.velocity,
            scattering_angle=theta_deg,
            charge_state=particle.charge_state,
            target_density_m3=density_target,
            cross_section_dir=str(cross_section_dir),
        )
    elif reaction == "ionization":
        reaction_result = ionization(
            projectile=projectile,
            target=target,
            projectile_velocity_m_s=particle.velocity,
            target_density_m3=density_target,
            scattering_angle=theta_deg,
            charge_state=particle.charge_state,
            cross_section_dir=cross_section_dir,
        )
    elif reaction == "lya":
        reaction_result = lyman_alpha_emission(
            projectile=projectile,
            target=target,
            projectile_velocity_m_s=particle.velocity,
            target_density_m3=density_target,
            scattering_angle=theta_deg,
            charge_state=particle.charge_state,
            cross_section_dir=cross_section_dir,
        )
    elif reaction == "state_change" and particle.charge_state == 0:
        reaction_result = electron_stripping(
            projectile=projectile,
            target=target,
            projectile_velocity_m_s=particle.velocity,
            target_density_m3=density_target,
            scattering_angle=theta_deg,
            charge_state=particle.charge_state,
            cross_section_dir=cross_section_dir,
        )
    elif reaction == "state_change" and particle.charge_state == 1:
        reaction_result = charge_exchange(
            projectile=projectile,
            target=target,
            projectile_velocity_m_s=particle.velocity,
            target_density_m3=density_target,
            scattering_angle=theta_deg,
            charge_state=particle.charge_state,
            cross_section_dir=cross_section_dir,
        )
    else:
        raise ValueError(f"Unsupported sampled reaction {reaction!r}.")

    particle.velocity = np.asarray(reaction_result["projectile_velocity_after_m_s"], dtype=float)
    particle.charge_state = 1 if reaction_result["charge_state_after"] == "ion" else 0
    particle.projectile = "H+" if particle.charge_state == 1 else "H"
    if collision_position is not None:
        particle.position = np.asarray(collision_position, dtype=float)
    if increment_iteration:
        particle.iteration += 1

    return {
        "particle": particle,
        "event": event,
        "reaction_result": reaction_result,
        "random_scattering": r_scatter,
        "local_collision_frequency_s-1": local_frequency,
        "cumulative_collision_frequency": particle.cumulative_collision_frequency,
        "collision_coefficient": coefficient,
        "position": np.asarray(particle.position, dtype=float).copy(),
        "iteration": particle.iteration,
    }
