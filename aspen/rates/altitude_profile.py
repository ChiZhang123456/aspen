from __future__ import annotations

"""Combine altitude profiles for ASPEN rate diagnostics."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import csv

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import numpy as np

from aspen.collisions.cross_section import DEFAULT_CROSS_SECTION_DIR, TARGETS

from .common import altitude_bin_edges, altitude_centers, flux_weight_per_particle, validate_altitude_edges
from .heating_rate import compute_heating_rate_profile
from .ionization_rate import compute_ionization_rate_profile
from .lyman_alpha import compute_lyman_alpha_emission_profile


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
    """Compute ionization, heating, and H Ly-alpha altitude profiles."""
    history_rows = list(rows)
    edges = validate_altitude_edges(altitude_edges_km)
    centers = altitude_centers(edges)
    ion = compute_ionization_rate_profile(
        history_rows,
        altitude_edges_km=edges,
        weight_m2_s=weight_m2_s,
        targets=targets,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
        mu_mode=mu_mode,
        cross_section_dir=cross_section_dir,
    )
    heat = compute_heating_rate_profile(
        history_rows,
        altitude_edges_km=edges,
        weight_m2_s=weight_m2_s,
    )
    lya = compute_lyman_alpha_emission_profile(
        history_rows,
        altitude_edges_km=edges,
        weight_m2_s=weight_m2_s,
        targets=targets,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
        mu_mode=mu_mode,
        cross_section_dir=cross_section_dir,
    )

    ion_rate = np.asarray(ion["rate_m-3_s-1"], dtype=float)
    lya_rate = np.asarray(lya["emission_rate_m-3_s-1"], dtype=float)
    out = []
    for ibin, center in enumerate(centers):
        row = {
            "altitude_min_km": float(edges[ibin]),
            "altitude_max_km": float(edges[ibin + 1]),
            "altitude_center_km": float(center),
            "ionization_rate_m-3_s-1": float(ion["total_rate_m-3_s-1"][ibin]),
            "lya_emission_rate_m-3_s-1": float(lya["total_emission_rate_m-3_s-1"][ibin]),
            "chemical_heating_rate_J_m-3_s-1": float(heat["chemical_heating_rate_J_m-3_s-1"][ibin]),
            "thermalization_heating_rate_J_m-3_s-1": float(heat["thermalization_heating_rate_J_m-3_s-1"][ibin]),
            "total_heating_rate_J_m-3_s-1": float(heat["total_heating_rate_J_m-3_s-1"][ibin]),
            "n_flux_crossings": int(ion["n_flux_crossings"][ibin]),
            "n_collision_events": int(heat["n_collision_events"][ibin]),
            "n_thermalized_particles": int(heat["n_thermalized_particles"][ibin]),
        }
        for itarget, target in enumerate(targets):
            row[f"ionization_rate_{target}_m-3_s-1"] = float(ion_rate[ibin, itarget])
            row[f"lya_emission_rate_{target}_m-3_s-1"] = float(lya_rate[ibin, itarget])
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
