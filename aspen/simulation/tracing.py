from __future__ import annotations

"""High-level particle tracing loops."""

from pathlib import Path

import numpy as np

from aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS
from aspen.particle_initialization import Particle

from .adaptive_step import advance_particle_until_collision_xyz
from .particle_state import should_stop_tracing


def trace_particle_xyz_until_stop(
    particle: Particle,
    targets: tuple[str, ...] = TARGETS,
    safety_factor: float = 0.2,
    min_step_m: float = 0.0,
    max_step_m: float = 1000.0,
    min_energy_ev: float = 10.0,
    min_altitude_km: float = 100.0,
    max_altitude_km: float = 1000.0,
    max_collisions: int = 10_000,
    max_steps_per_collision: int = 100_000,
    rng: np.random.Generator | None = None,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Trace one particle through repeated collisions until stop conditions.

    Charge-state changes do not stop tracing. After each collision, the
    particle receives a new random number and continues as `H+` or `H`,
    depending on the updated `charge_state`.
    """
    generator = rng if rng is not None else particle.rng
    if generator is None:
        generator = np.random.default_rng()

    segments = []
    collisions = []
    stop_check = should_stop_tracing(
        particle,
        min_energy_ev=min_energy_ev,
        min_altitude_km=min_altitude_km,
        max_altitude_km=max_altitude_km,
    )
    collision_count = 0
    while not stop_check["stop"] and collision_count < int(max_collisions):
        segment = advance_particle_until_collision_xyz(
            particle=particle,
            targets=targets,
            safety_factor=safety_factor,
            min_step_m=min_step_m,
            max_step_m=max_step_m,
            max_iterations=max_steps_per_collision,
            rng=generator,
            reset_after_collision=True,
            solar=solar,
            ls=ls,
            include_hot_o=include_hot_o,
            cross_section_dir=cross_section_dir,
            stop_at_domain_boundary=True,
            min_altitude_km=min_altitude_km,
            max_altitude_km=max_altitude_km,
        )
        segments.append(segment)
        if segment.get("collision") is not None:
            collisions.append(segment["collision"])
            collision_count += 1
        else:
            stop_check = segment.get("stop_condition") or should_stop_tracing(
                particle,
                min_energy_ev=min_energy_ev,
                min_altitude_km=min_altitude_km,
                max_altitude_km=max_altitude_km,
            )
            break
        stop_check = should_stop_tracing(
            particle,
            min_energy_ev=min_energy_ev,
            min_altitude_km=min_altitude_km,
            max_altitude_km=max_altitude_km,
        )

    if collision_count >= int(max_collisions) and not stop_check["stop"]:
        stop_check = {
            **stop_check,
            "stop": True,
            "reasons": tuple([*stop_check["reasons"], "max_collisions"]),
        }
    return {
        "particle": particle,
        "segments": segments,
        "collisions": collisions,
        "stop_condition": stop_check,
        "n_collisions": len(collisions),
        "n_steps": sum(len(segment.get("step_history", [])) for segment in segments),
    }
