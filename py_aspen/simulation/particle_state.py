from __future__ import annotations

"""Particle state helpers and tracing stop conditions."""

import numpy as np

from py_aspen.constants import ELEMENTARY_CHARGE_C, MARS_RADIUS_KM
from py_aspen.collisions.elastic_collision import MASS_KG
from py_aspen.particle_initialization import Particle


def particle_energy_ev(particle: Particle) -> float:
    """Return particle kinetic energy from its velocity, in eV."""
    projectile = "H+" if particle.charge_state == 1 else "H"
    velocity = np.asarray(particle.velocity, dtype=float)
    speed2 = float(np.dot(velocity, velocity))
    return float(0.5 * MASS_KG[projectile] * speed2 / ELEMENTARY_CHARGE_C)


def particle_altitude_km(particle_or_position: Particle | object) -> float:
    """Return Mars-centered Cartesian particle altitude in km.

    Input position is assumed to be `Pmso_xyz` in m.
    """
    if isinstance(particle_or_position, Particle):
        position_m = np.asarray(particle_or_position.position, dtype=float)
    else:
        position_m = np.asarray(particle_or_position, dtype=float)
    if position_m.shape != (3,):
        raise ValueError("Pmso_xyz must be a 3-element position vector in m.")
    radius_km = float(np.linalg.norm(position_m) / 1000.0)
    return radius_km - MARS_RADIUS_KM


def should_stop_tracing(
    particle: Particle,
    min_energy_ev: float = 10.0,
    min_altitude_km: float = 100.0,
    max_altitude_km: float = 1000.0,
) -> dict[str, object]:
    """Return whether particle tracing should stop.

    Charge-state changes do not stop tracing. Tracing stops only when kinetic
    energy is below `min_energy_ev` or the particle is outside the altitude
    domain [`min_altitude_km`, `max_altitude_km`].
    """
    energy_ev = particle_energy_ev(particle)
    altitude_km = particle_altitude_km(particle)
    below_energy = energy_ev < float(min_energy_ev)
    below_domain = altitude_km < float(min_altitude_km)
    above_domain = altitude_km > float(max_altitude_km)
    reasons = []
    if below_energy:
        reasons.append("energy_below_min")
    if below_domain:
        reasons.append("altitude_below_min")
    if above_domain:
        reasons.append("altitude_above_max")
    return {
        "stop": bool(reasons),
        "reasons": tuple(reasons),
        "energy_ev": float(energy_ev),
        "min_energy_ev": float(min_energy_ev),
        "altitude_km": float(altitude_km),
        "min_altitude_km": float(min_altitude_km),
        "max_altitude_km": float(max_altitude_km),
        "out_of_domain": bool(below_domain or above_domain),
        "below_energy": bool(below_energy),
        "charge_state": int(particle.charge_state),
        "projectile": particle.projectile,
    }


def reset_collision_random_number(
    particle: Particle,
    rng: np.random.Generator | None = None,
) -> float:
    """Reset the particle collision random number after a collision.

    The cumulative optical depth is reset to zero so the next free-streaming
    path starts from a new threshold, tau = -ln(R).
    """
    generator = rng if rng is not None else particle.rng
    if generator is None:
        generator = np.random.default_rng()
    particle.R = float(generator.random())
    particle.cumulative_collision_frequency = 0.0
    return float(particle.R)
