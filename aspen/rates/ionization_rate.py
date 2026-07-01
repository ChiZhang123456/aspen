from __future__ import annotations

"""Ionization rate profile from sampled ionization events."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from aspen.collisions.cross_section import TARGETS

from .common import as_float, bin_index, event_rate_weight, validate_altitude_edges


def compute_ionization_rate_profile(
    rows: Iterable[Mapping[str, object]],
    *,
    altitude_edges_km: Sequence[float] | None = None,
    weight_m2_s: float,
    weight_unit: str = "m-2_s-1",
    targets: tuple[str, ...] = TARGETS,
) -> dict[str, object]:
    """Return target-resolved ionization rate from actual collision events.

    For a one-dimensional column model, each event contributes
    `weight_m2_s / dz_m`, giving units of m^-3 s^-1.
    """
    edges = validate_altitude_edges(altitude_edges_km)
    dz_m = np.diff(edges) * 1000.0
    rates = np.zeros((edges.size - 1, len(targets)), dtype=float)
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
        itarget = target_index[target]
        rates[ibin, itarget] += event_rate_weight(float(weight_m2_s), dz_m[ibin], weight_unit)
        event_counts[ibin, itarget] += 1

    return {
        "altitude_edges_km": edges,
        "targets": tuple(targets),
        "weight_unit": weight_unit,
        "rate_m-3_s-1": rates,
        "total_rate_m-3_s-1": rates.sum(axis=1),
        "n_ionization_events": event_counts,
        "total_ionization_events": event_counts.sum(axis=1),
    }
