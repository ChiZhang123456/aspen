from __future__ import annotations

"""Particle state containers used by py_aspen Monte Carlo transport."""

from dataclasses import dataclass, field

import numpy as np

from py_aspen.collisions.cross_section import normalize_projectile


def _as_vector3(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-element vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _random_unit_interval(rng: np.random.Generator | None = None) -> float:
    generator = rng if rng is not None else np.random.default_rng()
    return float(generator.random())


def _validate_random_number(value: float) -> float:
    random_number = float(value)
    if not 0.0 <= random_number < 1.0:
        raise ValueError("R must be in the interval [0, 1).")
    return random_number


@dataclass(slots=True)
class Particle:
    """State for one H or H+ Monte Carlo particle.

    Parameters
    ----------
    projectile
        Particle species, accepted values are `H`, `H-ENA`, `H+`, `Hplus`, or
        `proton`. The stored value is normalized to `H` or `H+`.
    velocity
        Particle velocity vector, usually in m/s.
    position
        Particle position vector in m. For py_aspen MSO transport this is
        Mars-centered Cartesian `Pmso_xyz`.
    R
        One random number in [0, 1) assigned at initialization.
    cumulative_collision_frequency
        Accumulated collision frequency or optical-depth-like integral.
    local_collision_frequency
        Collision frequency at the current particle location.
    iteration
        Number of collision or transport-update iterations already applied to
        this particle.
    """

    projectile: str
    velocity: object
    position: object
    R: float | None = None
    cumulative_collision_frequency: float = 0.0
    local_collision_frequency: float = 0.0
    iteration: int = 0
    rng: np.random.Generator | None = field(default=None, repr=False, compare=False)
    charge_state: int = field(init=False)

    def __post_init__(self) -> None:
        self.projectile = normalize_projectile(self.projectile)
        self.velocity = _as_vector3(self.velocity, "velocity")
        self.position = _as_vector3(self.position, "position")
        self.charge_state = 1 if self.projectile == "H+" else 0
        self.R = _validate_random_number(
            _random_unit_interval(self.rng) if self.R is None else self.R
        )
        self.cumulative_collision_frequency = float(self.cumulative_collision_frequency)
        self.local_collision_frequency = float(self.local_collision_frequency)
        self.iteration = int(self.iteration)

    @property
    def species(self) -> str:
        """Return the normalized particle species, `H` or `H+`."""
        return self.projectile


def initialize_particle(
    projectile: str,
    velocity: object,
    position: object,
    R: float | None = None,
    iteration: int = 0,
    rng: np.random.Generator | None = None,
) -> Particle:
    """Create one initialized H or H+ particle."""
    return Particle(
        projectile=projectile,
        velocity=velocity,
        position=position,
        R=R,
        iteration=iteration,
        rng=rng,
    )
