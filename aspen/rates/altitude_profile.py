from __future__ import annotations

"""Compute 1D altitude profiles from ASPEN Monte Carlo histories.

The ionization and H Ly-alpha profiles use a flux crossing estimator:

    q_j(z) = n_j(z) sum_i W_i mu_i sigma_j(E_i)

where W_i is the solar-wind flux weight per Monte Carlo particle in
m^-2 s^-1, and mu_i is the vertical or radial velocity factor. Heating from
sampled chemical reactions is accumulated from the collision history and
divided by altitude-bin thickness to give J m^-3 s^-1.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import csv

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import numpy as np

from aspen.constants import ELEMENTARY_CHARGE_C
from aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS, cross_section
from aspen.neutral_density_model import neutral_density_xyz


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


def _as_float(row: Mapping[str, object], key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _mu(position_m: np.ndarray, velocity_m_s: np.ndarray, mode: str) -> float:
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


def _bin_index(value: float, edges: np.ndarray) -> int | None:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        return None
    return index


def _rows_by_particle(rows: Iterable[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        pid = int(row["particle_id"])
        grouped.setdefault(pid, []).append(row)
    for particle_rows in grouped.values():
        particle_rows.sort(key=lambda row: int(row["event_number"]))
    return grouped


def _density_at(position_m: np.ndarray, targets: Sequence[str], solar: str | int, ls: int | str, include_hot_o: bool) -> dict[str, float]:
    return neutral_density_xyz(
        position_m,
        species=tuple(targets),
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
        position_unit="m",
    )


def compute_altitude_rate_profiles(
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
) -> list[dict[str, float]]:
    """Compute altitude profiles from flattened Monte Carlo history rows.

    Returns rows with:

    - `ionization_rate_m-3_s-1`
    - `lya_emission_rate_m-3_s-1`
    - `chemical_heating_rate_J_m-3_s-1`
    - `thermalization_heating_rate_J_m-3_s-1`
    - `total_heating_rate_J_m-3_s-1`
    """
    edges = np.asarray(
        altitude_edges_km if altitude_edges_km is not None else altitude_bin_edges(),
        dtype=float,
    )
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("altitude_edges_km must be a 1D array with at least two edges.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    dz_m = np.diff(edges) * 1000.0
    nbin = centers.size

    ion = np.zeros((nbin, len(targets)), dtype=float)
    lya = np.zeros((nbin, len(targets)), dtype=float)
    chem_heat_ev_m2_s = np.zeros(nbin, dtype=float)
    thermal_heat_ev_m2_s = np.zeros(nbin, dtype=float)
    crossing_counts = np.zeros(nbin, dtype=int)
    collision_counts = np.zeros(nbin, dtype=int)
    thermal_counts = np.zeros(nbin, dtype=int)

    grouped = _rows_by_particle(rows)
    for particle_rows in grouped.values():
        previous = None
        for row in particle_rows:
            event_type = str(row["event_type"])
            if event_type == "transport" and previous is not None:
                alt0 = _as_float(previous, "altitude_km")
                alt1 = _as_float(row, "altitude_km")
                if np.isfinite(alt0) and np.isfinite(alt1) and alt0 != alt1:
                    low = min(alt0, alt1)
                    high = max(alt0, alt1)
                    crossed = np.where((centers >= low) & (centers <= high))[0]
                    p0 = np.array([_as_float(previous, "x_m"), _as_float(previous, "y_m"), _as_float(previous, "z_m")])
                    p1 = np.array([_as_float(row, "x_m"), _as_float(row, "y_m"), _as_float(row, "z_m")])
                    vel = np.array([_as_float(row, "vx_m_s"), _as_float(row, "vy_m_s"), _as_float(row, "vz_m_s")])
                    projectile = str(row["projectile"])
                    energy = _as_float(row, "energy_before_ev")
                    for ibin in crossed:
                        frac = (centers[ibin] - alt0) / (alt1 - alt0)
                        pos = p0 + frac * (p1 - p0)
                        mu = _mu(pos, vel, mu_mode)
                        if mu == 0.0:
                            continue
                        density = _density_at(pos, targets, solar, ls, include_hot_o)
                        for itarget, target in enumerate(targets):
                            n_target = float(density[target])
                            sigma_i = float(
                                cross_section(projectile, target, "ionization", energy, cross_section_dir)[
                                    "sigma_m2"
                                ]
                            )
                            sigma_l = float(
                                cross_section(projectile, target, "lya", energy, cross_section_dir)[
                                    "sigma_m2"
                                ]
                            )
                            ion[ibin, itarget] += float(weight_m2_s) * mu * n_target * sigma_i
                            lya[ibin, itarget] += float(weight_m2_s) * mu * n_target * sigma_l
                        crossing_counts[ibin] += 1

            if event_type == "collision":
                ibin = _bin_index(_as_float(row, "altitude_km"), edges)
                if ibin is not None:
                    chem_heat_ev_m2_s[ibin] += float(weight_m2_s) * max(0.0, _as_float(row, "energy_loss_ev", 0.0))
                    collision_counts[ibin] += 1
            previous = row

        if particle_rows:
            last = particle_rows[-1]
            if "energy_below_min" in str(last.get("stop_reason", "")):
                ibin = _bin_index(_as_float(last, "altitude_km"), edges)
                if ibin is not None:
                    thermal_heat_ev_m2_s[ibin] += float(weight_m2_s) * max(0.0, _as_float(last, "energy_after_ev", 0.0))
                    thermal_counts[ibin] += 1

    out = []
    ion_total = ion.sum(axis=1)
    lya_total = lya.sum(axis=1)
    chemical_heat = chem_heat_ev_m2_s * ELEMENTARY_CHARGE_C / dz_m
    thermal_heat = thermal_heat_ev_m2_s * ELEMENTARY_CHARGE_C / dz_m
    for ibin, center in enumerate(centers):
        row = {
            "altitude_min_km": float(edges[ibin]),
            "altitude_max_km": float(edges[ibin + 1]),
            "altitude_center_km": float(center),
            "ionization_rate_m-3_s-1": float(ion_total[ibin]),
            "lya_emission_rate_m-3_s-1": float(lya_total[ibin]),
            "chemical_heating_rate_J_m-3_s-1": float(chemical_heat[ibin]),
            "thermalization_heating_rate_J_m-3_s-1": float(thermal_heat[ibin]),
            "total_heating_rate_J_m-3_s-1": float(chemical_heat[ibin] + thermal_heat[ibin]),
            "n_flux_crossings": int(crossing_counts[ibin]),
            "n_collision_events": int(collision_counts[ibin]),
            "n_thermalized_particles": int(thermal_counts[ibin]),
        }
        for itarget, target in enumerate(targets):
            row[f"ionization_rate_{target}_m-3_s-1"] = float(ion[ibin, itarget])
            row[f"lya_emission_rate_{target}_m-3_s-1"] = float(lya[ibin, itarget])
        out.append(row)
    return out


def write_rate_profile_csv(rows: list[dict[str, float]], filename: str | Path) -> Path:
    """Write rate profile rows to CSV."""
    if not rows:
        raise ValueError("No profile rows to write.")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_altitude_rate_profiles(rows: Sequence[Mapping[str, object]], filename: str | Path) -> Path:
    """Plot ionization, heating, and H Ly-alpha profiles versus altitude."""
    altitude = np.asarray([float(row["altitude_center_km"]) for row in rows])
    ion = np.asarray([float(row["ionization_rate_m-3_s-1"]) for row in rows])
    heat = np.asarray([float(row["total_heating_rate_J_m-3_s-1"]) for row in rows])
    chem = np.asarray([float(row["chemical_heating_rate_J_m-3_s-1"]) for row in rows])
    therm = np.asarray([float(row["thermalization_heating_rate_J_m-3_s-1"]) for row in rows])
    lya = np.asarray([float(row["lya_emission_rate_m-3_s-1"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0), constrained_layout=True, sharey=True)
    axes[0].plot(ion, altitude, color="tab:red", lw=2.0)
    axes[0].set_xlabel(r"Ionization rate (m$^{-3}$ s$^{-1}$)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_xscale("log")

    axes[1].plot(heat, altitude, color="black", lw=2.0, label="total")
    axes[1].plot(chem, altitude, color="tab:blue", lw=1.5, label="chemical")
    axes[1].plot(therm, altitude, color="tab:green", lw=1.5, label="thermalized")
    axes[1].set_xlabel(r"Heating rate (J m$^{-3}$ s$^{-1}$)")
    axes[1].set_xscale("log")
    axes[1].legend(frameon=False, fontsize=12)

    axes[2].plot(lya, altitude, color="tab:purple", lw=2.0)
    axes[2].set_xlabel(r"H Ly-alpha emission (m$^{-3}$ s$^{-1}$)")
    axes[2].set_xscale("log")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
