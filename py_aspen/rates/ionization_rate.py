from __future__ import annotations

"""Ionization rate profile from sampled ionization events."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from py_aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS, cross_section

from .common import (
    as_float,
    bin_index,
    density_at,
    radial_velocity_component,
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
    """Return target-resolved ionization rate from sampled ionization events.

    Only collision rows with `reaction == "ionization"` contribute. For each
    ionization event in an altitude bin:

        q_j += n_j(r) * W_i * v_r * sigma_j(E)

    where `W_i` has units of m^-3 per model particle. This follows the
    density-weight formula and does not divide by altitude-bin thickness.
    """
    edges = validate_altitude_edges(altitude_edges_km)
    rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    hplus_rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    h_rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
    event_counts = np.zeros((edges.size - 1, len(targets)), dtype=int)

    target_index = {target: i for i, target in enumerate(targets)}
    for row in rows:
        if str(row.get("event_type", "")) != "collision":
            continue
        if str(row.get("reaction", "")) != "ionization":
            continue
        target = str(row.get("target", ""))
        if target not in target_index:
            continue
        ibin = bin_index(as_float(row, "altitude_km"), edges)
        if ibin is None:
            continue
        position_m = np.array([as_float(row, "x_m"), as_float(row, "y_m"), as_float(row, "z_m")])
        velocity_m_s = np.array([as_float(row, "vx_m_s"), as_float(row, "vy_m_s"), as_float(row, "vz_m_s")])
        vr_m_s = radial_velocity_component(position_m, velocity_m_s, radial_velocity_mode)
        if vr_m_s == 0.0:
            continue
        pid = int(row["particle_id"])
        projectile = str(row["projectile"])
        energy_ev = as_float(row, "energy_before_ev")
        density = density_at(position_m, targets, solar, ls, include_hot_o)
        itarget = target_index[target]
        sigma_m2 = float(cross_section(projectile, target, "ionization", energy_ev, cross_section_dir)["sigma_m2"])
        contribution = float(density[target]) * _particle_weight(pid, particle_weight_m3) * vr_m_s * sigma_m2
        rates[ibin, itarget] += contribution
        if projectile == "H+":
            hplus_rates[ibin, itarget] += contribution
        else:
            h_rates[ibin, itarget] += contribution
        event_counts[ibin, itarget] += 1

    return {
        "altitude_edges_km": edges,
        "targets": tuple(targets),
        "particle_weight_unit": "m-3",
        "radial_velocity_mode": radial_velocity_mode,
        "rate_m-3_s-1": rates,
        "hplus_rate_m-3_s-1": hplus_rates,
        "h_ena_rate_m-3_s-1": h_rates,
        "total_rate_m-3_s-1": rates.sum(axis=1),
        "n_ionization_events": event_counts,
        "total_ionization_events": event_counts.sum(axis=1),
    }
