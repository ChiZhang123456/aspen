from __future__ import annotations

"""H Ly-alpha volume emission rate from Monte Carlo transport histories."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS, cross_section

from .common import density_at, iter_flux_crossings, mu_factor, rows_by_particle, validate_altitude_edges


def compute_lyman_alpha_emission_profile(
    rows: Iterable[Mapping[str, object]],
    *,
    altitude_edges_km: Sequence[float] | None = None,
    weight_m2_s: float,
    targets: tuple[str, ...] = TARGETS,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    mu_mode: str = "absolute",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return target-resolved H Ly-alpha emission rate in each altitude bin."""
    edges = validate_altitude_edges(altitude_edges_km)
    rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    crossing_counts = np.zeros(edges.size - 1, dtype=int)

    for particle_rows in rows_by_particle(rows).values():
        for crossing in iter_flux_crossings(particle_rows, edges):
            ibin = int(crossing["bin_index"])
            position_m = np.asarray(crossing["position_m"], dtype=float)
            velocity_m_s = np.asarray(crossing["velocity_m_s"], dtype=float)
            mu = mu_factor(position_m, velocity_m_s, mu_mode)
            if mu == 0.0:
                continue
            density = density_at(position_m, targets, solar, ls, include_hot_o)
            projectile = str(crossing["projectile"])
            energy = float(crossing["energy_ev"])
            for itarget, target in enumerate(targets):
                sigma = float(cross_section(projectile, target, "lya", energy, cross_section_dir)["sigma_m2"])
                rates[ibin, itarget] += float(weight_m2_s) * mu * float(density[target]) * sigma
            crossing_counts[ibin] += 1

    return {
        "altitude_edges_km": edges,
        "targets": tuple(targets),
        "emission_rate_m-3_s-1": rates,
        "total_emission_rate_m-3_s-1": rates.sum(axis=1),
        "n_flux_crossings": crossing_counts,
    }
