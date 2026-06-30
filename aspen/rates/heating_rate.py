from __future__ import annotations

"""Heating rate profile from Monte Carlo collision histories."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from aspen.constants import ELEMENTARY_CHARGE_C

from .common import as_float, bin_index, rows_by_particle, validate_altitude_edges


def compute_heating_rate_profile(
    rows: Iterable[Mapping[str, object]],
    *,
    altitude_edges_km: Sequence[float] | None = None,
    weight_m2_s: float,
) -> dict[str, object]:
    """Return chemical, thermalization, and total heating rates.

    Chemical heating is accumulated from sampled collision energy losses. If a
    particle stops because its energy falls below 10 eV, its remaining energy
    is counted as thermalized atmospheric heating in that altitude bin.
    """
    edges = validate_altitude_edges(altitude_edges_km)
    dz_m = np.diff(edges) * 1000.0
    chemical_ev_m2_s = np.zeros(edges.size - 1, dtype=float)
    thermal_ev_m2_s = np.zeros(edges.size - 1, dtype=float)
    collision_counts = np.zeros(edges.size - 1, dtype=int)
    thermal_counts = np.zeros(edges.size - 1, dtype=int)

    for particle_rows in rows_by_particle(rows).values():
        for row in particle_rows:
            if str(row["event_type"]) != "collision":
                continue
            ibin = bin_index(as_float(row, "altitude_km"), edges)
            if ibin is None:
                continue
            chemical_ev_m2_s[ibin] += float(weight_m2_s) * max(0.0, as_float(row, "energy_loss_ev", 0.0))
            collision_counts[ibin] += 1

        if particle_rows:
            last = particle_rows[-1]
            if "energy_below_min" in str(last.get("stop_reason", "")):
                ibin = bin_index(as_float(last, "altitude_km"), edges)
                if ibin is not None:
                    thermal_ev_m2_s[ibin] += float(weight_m2_s) * max(0.0, as_float(last, "energy_after_ev", 0.0))
                    thermal_counts[ibin] += 1

    chemical = chemical_ev_m2_s * ELEMENTARY_CHARGE_C / dz_m
    thermal = thermal_ev_m2_s * ELEMENTARY_CHARGE_C / dz_m
    return {
        "altitude_edges_km": edges,
        "chemical_heating_rate_J_m-3_s-1": chemical,
        "thermalization_heating_rate_J_m-3_s-1": thermal,
        "total_heating_rate_J_m-3_s-1": chemical + thermal,
        "n_collision_events": collision_counts,
        "n_thermalized_particles": thermal_counts,
    }
