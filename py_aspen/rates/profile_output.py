from __future__ import annotations

"""CSV and plotting helpers for process-separated rate profiles."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import csv

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import numpy as np

from .common import altitude_centers


def combine_rate_profile_rows(
    ionization: Mapping[str, object],
    heating: Mapping[str, object],
    lyman_alpha: Mapping[str, object],
) -> list[dict[str, float]]:
    """Combine process-specific profile dictionaries into CSV rows."""
    edges = np.asarray(ionization["altitude_edges_km"], dtype=float)
    centers = altitude_centers(edges)
    targets = tuple(ionization["targets"])
    ion = np.asarray(ionization["rate_m-3_s-1"], dtype=float)
    ion_hplus = np.asarray(ionization["hplus_rate_m-3_s-1"], dtype=float)
    ion_hena = np.asarray(ionization["h_ena_rate_m-3_s-1"], dtype=float)
    ion_events = np.asarray(ionization["n_ionization_events"], dtype=int)
    lya = np.asarray(lyman_alpha["emission_rate_m-3_s-1"], dtype=float)
    lya_events = np.asarray(lyman_alpha["n_lya_events"], dtype=int)
    heat_by_target = np.asarray(heating["chemical_heating_rate_by_target_J_m-3_s-1"], dtype=float)
    heat_events = np.asarray(heating["n_collision_events_by_target"], dtype=int)

    out = []
    for ibin, center in enumerate(centers):
        row = {
            "altitude_min_km": float(edges[ibin]),
            "altitude_max_km": float(edges[ibin + 1]),
            "altitude_center_km": float(center),
            "ionization_rate_m-3_s-1": float(ionization["total_rate_m-3_s-1"][ibin]),
            "lya_emission_rate_m-3_s-1": float(lyman_alpha["total_emission_rate_m-3_s-1"][ibin]),
            "chemical_heating_rate_J_m-3_s-1": float(heating["chemical_heating_rate_J_m-3_s-1"][ibin]),
            "thermalization_heating_rate_J_m-3_s-1": float(heating["thermalization_heating_rate_J_m-3_s-1"][ibin]),
            "total_heating_rate_J_m-3_s-1": float(heating["total_heating_rate_J_m-3_s-1"][ibin]),
            "n_ionization_events": int(ionization["total_ionization_events"][ibin]),
            "n_lya_events": int(lyman_alpha["total_lya_events"][ibin]),
            "n_collision_events": int(heating["n_collision_events"][ibin]),
            "n_thermalized_particles": int(heating["n_thermalized_particles"][ibin]),
        }
        for itarget, target in enumerate(targets):
            row[f"ionization_rate_{target}_m-3_s-1"] = float(ion[ibin, itarget])
            row[f"ionization_rate_Hplus_{target}_m-3_s-1"] = float(ion_hplus[ibin, itarget])
            row[f"ionization_rate_HENA_{target}_m-3_s-1"] = float(ion_hena[ibin, itarget])
            row[f"n_ionization_events_{target}"] = int(ion_events[ibin, itarget])
            row[f"lya_emission_rate_{target}_m-3_s-1"] = float(lya[ibin, itarget])
            row[f"n_lya_events_{target}"] = int(lya_events[ibin, itarget])
            row[f"chemical_heating_rate_{target}_J_m-3_s-1"] = float(heat_by_target[ibin, itarget])
            row[f"n_collision_events_{target}"] = int(heat_events[ibin, itarget])
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


def plot_rate_profiles(rows: Sequence[Mapping[str, object]], filename: str | Path) -> Path:
    """Plot ionization, heating, and H Ly-alpha event-rate profiles."""
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
