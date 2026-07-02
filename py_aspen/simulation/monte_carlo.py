from __future__ import annotations

"""Monte Carlo particle ensembles for py_aspen transport."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from py_aspen.constants import MARS_RADIUS_KM
from py_aspen.collisions.collision_frequency import speed_from_energy
from py_aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS
from py_aspen.particle_initialization import Particle

from .particle_state import particle_altitude_km, particle_energy_ev
from .tracing import trace_particle_xyz_until_stop


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for one py_aspen Monte Carlo particle ensemble."""

    n_particles: int = 5000
    initial_projectile: str = "H-ENA"
    initial_altitude_km: float = 600.0
    initial_position_m: tuple[float, float, float] | None = None
    initial_velocity_m_s: tuple[float, float, float] = (-400_000.0, 0.0, 0.0)
    seed: int = 7
    targets: tuple[str, ...] = TARGETS
    safety_factor: float = 0.2
    max_step_m: float = 1000.0
    min_energy_ev: float = 10.0
    min_altitude_km: float = 100.0
    max_altitude_km: float = 1000.0
    max_collisions: int = 10_000
    max_steps_per_collision: int = 100_000
    solar: str | int = "solar_min"
    ls: int | str = 0
    include_hot_o: bool = True
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR

    def position_m(self) -> np.ndarray:
        """Return the initial Cartesian MSO position in m."""
        if self.initial_position_m is not None:
            position = np.asarray(self.initial_position_m, dtype=float)
        else:
            position = np.array([(MARS_RADIUS_KM + self.initial_altitude_km) * 1000.0, 0.0, 0.0])
        if position.shape != (3,):
            raise ValueError("initial_position_m must be a 3-element vector in m.")
        return position

    def velocity_m_s(self) -> np.ndarray:
        """Return the initial velocity in m/s."""
        velocity = np.asarray(self.initial_velocity_m_s, dtype=float)
        if velocity.shape != (3,):
            raise ValueError("initial_velocity_m_s must be a 3-element vector in m/s.")
        return velocity


def initialize_monte_carlo_particles(config: MonteCarloConfig) -> list[Particle]:
    """Initialize particles with identical position and velocity and unique R."""
    generator = np.random.default_rng(config.seed)
    position = config.position_m()
    velocity = config.velocity_m_s()
    particles = []
    for _ in range(int(config.n_particles)):
        particles.append(
            Particle(
                config.initial_projectile,
                velocity=velocity.copy(),
                position=position.copy(),
                R=float(generator.random()),
                rng=np.random.default_rng(int(generator.integers(0, np.iinfo(np.int32).max))),
            )
        )
    return particles


def _collision_counts(collisions: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {
        "n_elastic": 0,
        "n_ionization": 0,
        "n_lya": 0,
        "n_state_change": 0,
    }
    for collision in collisions:
        reaction = str(collision["event"]["reaction"])
        key = f"n_{reaction}"
        if key in counts:
            counts[key] += 1
    return counts


def run_one_monte_carlo_particle(
    particle_id: int,
    config: MonteCarloConfig,
) -> dict[str, object]:
    """Run one Monte Carlo particle and return a compact summary row."""
    seed_sequence = np.random.SeedSequence([config.seed, int(particle_id)])
    rng = np.random.default_rng(seed_sequence)
    particle = Particle(
        config.initial_projectile,
        velocity=config.velocity_m_s(),
        position=config.position_m(),
        R=float(rng.random()),
        rng=rng,
    )
    initial_energy_ev = particle_energy_ev(particle)
    initial_altitude_km = particle_altitude_km(particle)
    result = trace_particle_xyz_until_stop(
        particle,
        targets=config.targets,
        safety_factor=config.safety_factor,
        max_step_m=config.max_step_m,
        min_energy_ev=config.min_energy_ev,
        min_altitude_km=config.min_altitude_km,
        max_altitude_km=config.max_altitude_km,
        max_collisions=config.max_collisions,
        max_steps_per_collision=config.max_steps_per_collision,
        rng=rng,
        solar=config.solar,
        ls=config.ls,
        include_hot_o=config.include_hot_o,
        cross_section_dir=config.cross_section_dir,
    )
    final_position = np.asarray(particle.position, dtype=float)
    final_energy_ev = particle_energy_ev(particle)
    counts = _collision_counts(result["collisions"])
    stop = result["stop_condition"]
    return {
        "particle_id": int(particle_id),
        "initial_projectile": config.initial_projectile,
        "initial_charge_state": 0 if config.initial_projectile in ("H", "H-ENA", "ENA") else 1,
        "initial_energy_ev": initial_energy_ev,
        "initial_altitude_km": initial_altitude_km,
        "initial_x_m": float(config.position_m()[0]),
        "initial_y_m": float(config.position_m()[1]),
        "initial_z_m": float(config.position_m()[2]),
        "initial_vx_m_s": float(config.velocity_m_s()[0]),
        "initial_vy_m_s": float(config.velocity_m_s()[1]),
        "initial_vz_m_s": float(config.velocity_m_s()[2]),
        "final_projectile": particle.projectile,
        "final_charge_state": int(particle.charge_state),
        "final_energy_ev": final_energy_ev,
        "final_altitude_km": particle_altitude_km(particle),
        "final_x_m": float(final_position[0]),
        "final_y_m": float(final_position[1]),
        "final_z_m": float(final_position[2]),
        "n_steps": int(result["n_steps"]),
        "n_collisions": int(result["n_collisions"]),
        **counts,
        "stop_reasons": ";".join(stop["reasons"]),
    }


def summarize_monte_carlo_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return ensemble-level summary statistics from compact particle rows."""
    if not rows:
        return {}
    final_energy = np.asarray([float(row["final_energy_ev"]) for row in rows])
    final_altitude = np.asarray([float(row["final_altitude_km"]) for row in rows])
    n_collisions = np.asarray([int(row["n_collisions"]) for row in rows])
    stop_reasons = {}
    for row in rows:
        reason = str(row["stop_reasons"])
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "n_particles": len(rows),
        "final_energy_mean_ev": float(np.mean(final_energy)),
        "final_energy_median_ev": float(np.median(final_energy)),
        "final_altitude_mean_km": float(np.mean(final_altitude)),
        "n_collisions_mean": float(np.mean(n_collisions)),
        "n_collisions_median": float(np.median(n_collisions)),
        "stop_reasons": stop_reasons,
    }
