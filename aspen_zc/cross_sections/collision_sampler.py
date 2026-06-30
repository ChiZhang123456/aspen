from __future__ import annotations

"""Monte Carlo sampling of collision targets and reaction channels.

This module is used after a transport step has already decided that a
collision occurs. It then samples:

1. Which neutral target was hit, using `n_j * sum_k sigma_j,k`.
2. Which reaction channel occurred for that target, using `sigma_k`.

The default reaction set is fixed to the four ASPEN core channels:
`elastic`, `ionization`, `state_change`, and `lya`. `halpha` is available in
the raw cross-section tables, but is intentionally not part of the default
Monte Carlo event sampler.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from .cross_section import (
    DEFAULT_CROSS_SECTION_DIR,
    REACTIONS,
    REACTION_INDEX,
    TARGETS,
    TARGET_INDEX,
    PROJECTILE_INDEX,
    load_cross_sections,
    normalize_projectile,
    normalize_target,
)


SAMPLE_REACTIONS = ("elastic", "ionization", "state_change", "lya")


def _normalize_probability_dict(probability: dict[str, float]) -> dict[str, float]:
    """Normalize positive weights and sort them from high to low probability."""
    total = float(sum(probability.values()))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Probability weights must have a positive finite sum.")
    normalized = {key: float(value / total) for key, value in probability.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def _choose_from_cumulative(
    probability: dict[str, float],
    random_number: float,
) -> tuple[str, dict[str, float]]:
    """Choose one key from a sorted probability dictionary.

    Returns both the selected key and the full cumulative probability table.
    For example, probabilities `CO2=0.7`, `O=0.2`, `N2=0.1` become cumulative
    values `0.7`, `0.9`, and `1.0`.
    """
    r = float(np.clip(random_number, 0.0, np.nextafter(1.0, 0.0)))
    cumulative = 0.0
    cumulative_probability: dict[str, float] = {}
    last_key = next(reversed(probability))
    selected_key: str | None = None
    for key, value in probability.items():
        cumulative += value
        cumulative_probability[key] = min(cumulative, 1.0)
        if selected_key is None and r < cumulative_probability[key]:
            selected_key = key
    return selected_key or last_key, cumulative_probability


def collision_target_probabilities(
    density_m3: Mapping[str, float],
    projectile: str,
    energy_ev: float,
    targets: tuple[str, ...] = ("CO2", "O"),
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, float]:
    """Return target probabilities after one collision has occurred.

    The probability for target j is

    P_j = n_j sum_k sigma_j,k / sum_j n_j sum_k sigma_j,k.

    `density_m3` is passed in directly so the sampler can be used with any
    atmospheric model or with a test dictionary.
    """
    projectile_key = normalize_projectile(projectile)
    target_keys = tuple(normalize_target(target) for target in targets)

    cs = load_cross_sections(str(Path(cross_section_dir)))
    ip = PROJECTILE_INDEX[projectile_key]
    weights: dict[str, float] = {}
    for target in target_keys:
        if target not in density_m3:
            raise KeyError(f"density_m3 is missing target {target!r}.")
        it = TARGET_INDEX[target]
        sigma_sum = 0.0
        for reaction in SAMPLE_REACTIONS:
            ir = REACTION_INDEX[reaction]
            sigma_sum += float(np.interp(energy_ev, cs.energy_ev, cs.sigma_m2[ip, it, ir, :]))
        weights[target] = float(density_m3[target]) * sigma_sum
    return _normalize_probability_dict(weights)


def reaction_probabilities(
    projectile: str,
    target: str,
    energy_ev: float,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, float]:
    """Return reaction probabilities for a fixed projectile, target, and energy.

    The probability for reaction k is

    P_k = sigma_k / sum_k sigma_k.

    The target density cancels in this second step because the target has
    already been selected.
    """
    projectile_key = normalize_projectile(projectile)
    target_key = normalize_target(target)

    cs = load_cross_sections(str(Path(cross_section_dir)))
    ip = PROJECTILE_INDEX[projectile_key]
    it = TARGET_INDEX[target_key]
    weights: dict[str, float] = {}
    for reaction in SAMPLE_REACTIONS:
        ir = REACTION_INDEX[reaction]
        weights[reaction] = float(np.interp(energy_ev, cs.energy_ev, cs.sigma_m2[ip, it, ir, :]))
    return _normalize_probability_dict(weights)


def sample_collision_event(
    density_m3: Mapping[str, float],
    projectile: str,
    energy_ev: float,
    random_target: float | None = None,
    random_reaction: float | None = None,
    rng: np.random.Generator | None = None,
    targets: tuple[str, ...] = ("CO2", "O"),
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Sample collision target and reaction after a collision has occurred.

    Parameters
    ----------
    density_m3
        Mapping such as {"CO2": nCO2, "O": nO}. Units are m^-3.
    projectile
        "H", "H-ENA", "H+", or "proton".
    energy_ev
        Projectile kinetic energy in eV.
    random_target, random_reaction
        Optional random numbers in [0, 1). If omitted, they are generated from
        `rng`.
    targets
        Candidate targets used in the normalization. The reaction channels are
        fixed to elastic, ionization, state_change, and lya.
    """
    generator = rng if rng is not None else np.random.default_rng()
    r_target = float(generator.random() if random_target is None else random_target)
    r_reaction = float(generator.random() if random_reaction is None else random_reaction)

    # Step 1: sample which neutral species participates in this collision.
    target_probability = collision_target_probabilities(
        density_m3=density_m3,
        projectile=projectile,
        energy_ev=energy_ev,
        targets=targets,
        cross_section_dir=cross_section_dir,
    )
    target, target_cumulative = _choose_from_cumulative(target_probability, r_target)

    # Step 2: sample the physical process for the selected neutral target.
    reaction_probability = reaction_probabilities(
        projectile=projectile,
        target=target,
        energy_ev=energy_ev,
        cross_section_dir=cross_section_dir,
    )
    reaction, reaction_cumulative = _choose_from_cumulative(
        reaction_probability,
        r_reaction,
    )

    return {
        "projectile": normalize_projectile(projectile),
        "energy_ev": float(energy_ev),
        "target": target,
        "reaction": reaction,
        "random_target": r_target,
        "random_reaction": r_reaction,
        "target_probability": target_probability,
        "target_cumulative_probability": target_cumulative,
        "reaction_probability": reaction_probability,
        "reaction_cumulative_probability": reaction_cumulative,
        "targets": tuple(target_probability.keys()),
        "reactions": tuple(reaction_probability.keys()),
        "reaction_set": SAMPLE_REACTIONS,
    }
