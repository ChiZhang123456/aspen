"""Cross-section and collision helper API for `aspen`.

This subpackage contains the packaged H/H+ collision cross-section tables and
the routines that use them: cross-section lookup, collision frequency,
Monte Carlo collision sampling, elastic collision kinematics, and scattering
angle lookup.
"""

from .collision_frequency import collision_frequency
from .collision_sampler import (
    collision_target_probabilities,
    reaction_probabilities,
    sample_collision_event,
)
from .charge_exchange import charge_exchange
from .cross_section import CrossSectionSet, cross_section, load_cross_sections
from .electron_stripping import electron_striping, electron_stripping
from .elastic_collision import elastic_collision_after_scattering
from .ionization import ionization
from .lyman_alpha_emission import lyman_alpha_emission
from .particle_collision import apply_random_collision_to_particle
from .scattering_angle import ScatteringAngleTable, scattering_angle

__all__ = [
    "CrossSectionSet",
    "apply_random_collision_to_particle",
    "charge_exchange",
    "collision_frequency",
    "collision_target_probabilities",
    "cross_section",
    "electron_striping",
    "electron_stripping",
    "elastic_collision_after_scattering",
    "ionization",
    "load_cross_sections",
    "lyman_alpha_emission",
    "reaction_probabilities",
    "sample_collision_event",
    "ScatteringAngleTable",
    "scattering_angle",
]
