from __future__ import annotations

"""Shared helpers for altitude rate diagnostics."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from py_aspen.neutral_density_model import neutral_density_xyz


def altitude_bin_edges(
    min_altitude_km: float = 100.0,
    max_altitude_km: float = 1000.0,
    bin_width_km: float = 10.0,
) -> np.ndarray:
    """Return regular altitude-bin edges in km."""
    return np.arange(
        float(min_altitude_km),
        float(max_altitude_km) + 0.5 * float(bin_width_km),
        float(bin_width_km),
    )


def flux_weight_per_particle(
    solar_wind_density_m3: float,
    solar_wind_speed_m_s: float,
    n_particles: int,
) -> float:
    """Return W = n_sw * V_sw / N in m^-2 s^-1 per model particle."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    return float(solar_wind_density_m3) * abs(float(solar_wind_speed_m_s)) / int(n_particles)


def validate_altitude_edges(altitude_edges_km: Sequence[float] | None) -> np.ndarray:
    """Return a validated 1D altitude edge array."""
    edges = np.asarray(altitude_edges_km if altitude_edges_km is not None else altitude_bin_edges(), dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("altitude_edges_km must be a 1D array with at least two edges.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("altitude_edges_km must be strictly increasing.")
    return edges


def altitude_centers(altitude_edges_km: Sequence[float]) -> np.ndarray:
    """Return altitude-bin centers in km."""
    edges = np.asarray(altitude_edges_km, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def as_float(row: Mapping[str, object], key: str, default: float = np.nan) -> float:
    """Read a row field as float while accepting empty CSV cells."""
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def bin_index(value: float, edges: np.ndarray) -> int | None:
    """Return the altitude-bin index for a value, or None outside the grid."""
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        return None
    return index


def event_rate_weight(weight: float, dz_m: float, weight_unit: str) -> float:
    """Convert a particle event weight to a volume event rate.

    Supported units:

    - `m-2_s-1`: column flux weight. The contribution is W / dz.
    - `m-3_s-1`: volume rate weight. The contribution is W directly.
    """
    key = weight_unit.strip().lower().replace("^", "")
    if key in {"m-2_s-1", "m^-2_s^-1", "m-2 s-1", "flux"}:
        return float(weight) / float(dz_m)
    if key in {"m-3_s-1", "m^-3_s^-1", "m-3 s-1", "volume"}:
        return float(weight)
    raise ValueError("weight_unit must be 'm-2_s-1' or 'm-3_s-1'.")


def rows_by_particle(rows: Iterable[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    """Group flattened history rows by particle id and sort by event number."""
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        pid = int(row["particle_id"])
        grouped.setdefault(pid, []).append(row)
    for particle_rows in grouped.values():
        particle_rows.sort(key=lambda row: int(row["event_number"]))
    return grouped


def mu_factor(position_m: np.ndarray, velocity_m_s: np.ndarray, mode: str) -> float:
    """Return the radial velocity factor for altitude-surface flux."""
    speed = float(np.linalg.norm(velocity_m_s))
    radius = float(np.linalg.norm(position_m))
    if speed <= 0.0 or radius <= 0.0:
        return 0.0
    radial_velocity = float(np.dot(velocity_m_s, position_m / radius))
    key = mode.strip().lower()
    if key == "absolute":
        return abs(radial_velocity) / speed
    if key == "inward":
        return max(0.0, -radial_velocity / speed)
    if key == "outward":
        return max(0.0, radial_velocity / speed)
    if key == "signed":
        return radial_velocity / speed
    raise ValueError("mu_mode must be 'absolute', 'inward', 'outward', or 'signed'.")


def radial_velocity_component(position_m: np.ndarray, velocity_m_s: np.ndarray, mode: str) -> float:
    """Return a nonnegative or signed radial velocity component in m/s."""
    radius = float(np.linalg.norm(position_m))
    if radius <= 0.0:
        return 0.0
    radial_velocity = float(np.dot(velocity_m_s, position_m / radius))
    key = mode.strip().lower()
    if key == "absolute":
        return abs(radial_velocity)
    if key == "inward":
        return max(0.0, -radial_velocity)
    if key == "outward":
        return max(0.0, radial_velocity)
    if key == "signed":
        return radial_velocity
    raise ValueError("radial_velocity_mode must be 'absolute', 'inward', 'outward', or 'signed'.")


def density_at(
    position_m: np.ndarray,
    targets: Sequence[str],
    solar: str | int,
    ls: int | str,
    include_hot_o: bool,
) -> dict[str, float]:
    """Return neutral density at a Cartesian MSO position in m."""
    return neutral_density_xyz(
        position_m,
        species=tuple(targets),
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
        position_unit="m",
    )


def iter_flux_crossings(
    particle_rows: Sequence[Mapping[str, object]],
    altitude_edges_km: Sequence[float],
) -> Iterable[dict[str, object]]:
    """Yield altitude-bin crossings from one particle history."""
    edges = np.asarray(altitude_edges_km, dtype=float)
    centers = altitude_centers(edges)
    previous = None
    for row in particle_rows:
        if str(row["event_type"]) == "transport" and previous is not None:
            alt0 = as_float(previous, "altitude_km")
            alt1 = as_float(row, "altitude_km")
            if np.isfinite(alt0) and np.isfinite(alt1) and alt0 != alt1:
                low = min(alt0, alt1)
                high = max(alt0, alt1)
                crossed = np.where((centers >= low) & (centers <= high))[0]
                p0 = np.array([as_float(previous, "x_m"), as_float(previous, "y_m"), as_float(previous, "z_m")])
                p1 = np.array([as_float(row, "x_m"), as_float(row, "y_m"), as_float(row, "z_m")])
                velocity = np.array([as_float(row, "vx_m_s"), as_float(row, "vy_m_s"), as_float(row, "vz_m_s")])
                for ibin in crossed:
                    frac = (centers[ibin] - alt0) / (alt1 - alt0)
                    yield {
                        "particle_id": int(row["particle_id"]),
                        "bin_index": int(ibin),
                        "position_m": p0 + frac * (p1 - p0),
                        "velocity_m_s": velocity,
                        "projectile": str(row["projectile"]),
                        "energy_ev": as_float(row, "energy_before_ev"),
                    }
        previous = row
