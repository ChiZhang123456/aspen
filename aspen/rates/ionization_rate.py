from __future__ import annotations

"""Ionization rate profile from density-weighted particle trajectories."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS, cross_section

from .common import (
    density_at,
    iter_flux_crossings,
    radial_velocity_component,
    rows_by_particle,
    validate_altitude_edges,
)


def _particle_weight(particle_id: int, particle_weight_m3: float | Mapping[int, float]) -> float:
    if isinstance(particle_weight_m3, Mapping):
        return float(particle_weight_m3[particle_id])
    return float(particle_weight_m3)


def compute_ionization_rate_profile(
    rows: Iterable[Mapping[str, object]],
    *,
    altitude_edges_km: Sequence[float] | None = None,
    particle_weight_m3: float | Mapping[int, float],
    targets: tuple[str, ...] = TARGETS,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
    radial_velocity_mode: str = "absolute",
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, object]:
    """Return target-resolved ionization rate from density-weighted particles.

    For each trajectory crossing in an altitude bin:

        q_j += n_j(r) * W_i * v_r * sigma_j(E)

    where `W_i` has units of m^-3 per model particle. H+ and H-ENA both
    contribute using their current projectile state, and charge-exchanged
    particles keep the same weight.
    """
    edges = validate_altitude_edges(altitude_edges_km)
    rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    hplus_rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    h_rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    crossing_counts = np.zeros(edges.size - 1, dtype=int)

    for particle_rows in rows_by_particle(rows).values():
        for crossing in iter_flux_crossings(particle_rows, edges):
            ibin = int(crossing["bin_index"])
            pid = int(crossing["particle_id"])
            position_m = np.asarray(crossing["position_m"], dtype=float)
            velocity_m_s = np.asarray(crossing["velocity_m_s"], dtype=float)
            vr_m_s = radial_velocity_component(position_m, velocity_m_s, radial_velocity_mode)
            if vr_m_s == 0.0:
                continue
            density = density_at(position_m, targets, solar, ls, include_hot_o)
            projectile = str(crossing["projectile"])
            energy = float(crossing["energy_ev"])
            weight = _particle_weight(pid, particle_weight_m3)
            for itarget, target in enumerate(targets):
                sigma = float(cross_section(projectile, target, "ionization", energy, cross_section_dir)["sigma_m2"])
                contribution = float(density[target]) * weight * vr_m_s * sigma
                rates[ibin, itarget] += contribution
                if projectile == "H+":
                    hplus_rates[ibin, itarget] += contribution
                else:
                    h_rates[ibin, itarget] += contribution
            crossing_counts[ibin] += 1

    return {
        "altitude_edges_km": edges,
        "targets": tuple(targets),
        "particle_weight_unit": "m-3",
        "radial_velocity_mode": radial_velocity_mode,
        "rate_m-3_s-1": rates,
        "hplus_rate_m-3_s-1": hplus_rates,
        "h_ena_rate_m-3_s-1": h_rates,
        "total_rate_m-3_s-1": rates.sum(axis=1),
        "n_flux_crossings": crossing_counts,
    }
