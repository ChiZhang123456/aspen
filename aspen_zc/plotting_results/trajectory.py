from __future__ import annotations

"""Plot particle trajectory, reactions, energy loss, and collision probability."""

from pathlib import Path
import csv

import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import numpy as np


REACTION_COLORS = {
    "elastic": "0.25",
    "ionization": "tab:orange",
    "lya": "tab:green",
    "state_change": "tab:red",
}


def _read_rows(rows_or_csv: list[dict[str, object]] | str | Path) -> list[dict[str, object]]:
    if isinstance(rows_or_csv, (str, Path)):
        with Path(rows_or_csv).open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return rows_or_csv


def _num(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def plot_particle_trace_history(
    rows_or_csv: list[dict[str, object]] | str | Path,
    output_file: str | Path,
    particle_id: int | None = None,
    title: str | None = None,
    font_size: float = 15.0,
) -> Path:
    """Plot one particle history with reactions and cumulative probability.

    The input should be rows from `flatten_trace_history` or a CSV written by
    `write_history_csv`.
    """
    rows = _read_rows(rows_or_csv)
    if particle_id is not None:
        rows = [row for row in rows if int(row["particle_id"]) == int(particle_id)]
    if not rows:
        raise ValueError("No rows available for plotting.")

    old_rc = {
        "font.size": mpl.rcParams["font.size"],
        "axes.titlesize": mpl.rcParams["axes.titlesize"],
        "axes.labelsize": mpl.rcParams["axes.labelsize"],
        "xtick.labelsize": mpl.rcParams["xtick.labelsize"],
        "ytick.labelsize": mpl.rcParams["ytick.labelsize"],
        "legend.fontsize": mpl.rcParams["legend.fontsize"],
        "figure.titlesize": mpl.rcParams["figure.titlesize"],
    }
    mpl.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 3,
            "figure.titlesize": font_size + 4,
        }
    )

    collisions = [row for row in rows if row["event_type"] == "collision"]
    event = _num(rows, "event_number")
    altitude_km = _num(rows, "altitude_km")
    energy_ev = _num(rows, "energy_after_ev")
    charge_state = _num(rows, "charge_state")

    fig, axes = plt.subplots(3, 2, figsize=(16.0, 12.0), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(event, altitude_km, color="black", lw=1.4)
    axes[0].axhline(100.0, color="tab:red", ls="--", lw=1.1)
    axes[0].axhline(1000.0, color="tab:red", ls="--", lw=1.1)
    for reaction, color in REACTION_COLORS.items():
        subset = [row for row in collisions if row["reaction"] == reaction]
        if subset:
            axes[0].scatter(
                [float(row["event_number"]) for row in subset],
                [float(row["altitude_km"]) for row in subset],
                s=32,
                color=color,
                label=reaction,
                zorder=3,
            )
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Altitude and reaction locations")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(event, energy_ev, color="tab:red", lw=1.4)
    if collisions:
        axes[1].scatter(
            [float(row["event_number"]) for row in collisions],
            [float(row["energy_after_ev"]) for row in collisions],
            s=28,
            color="black",
            zorder=3,
        )
    axes[1].axhline(10.0, color="tab:red", ls="--", lw=1.1)
    axes[1].set_ylabel("Energy (eV)")
    axes[1].set_title("Energy history")

    axes[2].step(event, charge_state, where="post", color="tab:blue", lw=1.4)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("Charge state")
    axes[2].set_title("Charge state")

    if collisions:
        collision_number = np.asarray([float(row["collision_number"]) for row in collisions])
        energy_loss = np.asarray([float(row["energy_loss_ev"]) for row in collisions])
        collision_event = np.asarray([float(row["event_number"]) for row in collisions])
        colors = [REACTION_COLORS.get(row["reaction"], "black") for row in collisions]
        axes[3].scatter(collision_event, energy_loss, color=colors, s=36)
    axes[3].set_ylabel("Energy loss (eV)")
    axes[3].set_yscale("symlog", linthresh=0.01)
    axes[3].set_title("Energy loss at reactions")

    if collisions:
        scatter_rows = [
            row
            for row in collisions
            if "scattering_angle_deg" in row and str(row["scattering_angle_deg"]) not in ("", "nan")
        ]
        if scatter_rows:
            collision_event = np.asarray([float(row["event_number"]) for row in scatter_rows])
            scattering_angle = np.asarray([float(row["scattering_angle_deg"]) for row in scatter_rows])
            colors = [REACTION_COLORS.get(row["reaction"], "black") for row in scatter_rows]
            axes[4].scatter(collision_event, scattering_angle, color=colors, s=36)
    axes[4].set_xlabel("Event number")
    axes[4].set_ylabel("Scattering angle (deg)")
    axes[4].set_title("Scattering angle at reactions")

    if "cumulative_collision_probability" in rows[0]:
        collision_probability = _num(rows, "cumulative_collision_probability")
    elif "cumulative_tau" in rows[0]:
        collision_probability = 1.0 - np.exp(-_num(rows, "cumulative_tau"))
    else:
        collision_probability = np.full_like(event, np.nan)
    axes[5].plot(event, collision_probability, color="tab:purple", lw=1.4)
    if collisions:
        scatter_rows = [
            row
            for row in collisions
            if (
                "cumulative_collision_probability" in row
                and str(row["cumulative_collision_probability"]) not in ("", "nan")
            )
            or ("cumulative_tau" in row and str(row["cumulative_tau"]) not in ("", "nan"))
        ]
        if scatter_rows:
            scatter_y = []
            for row in scatter_rows:
                if "cumulative_collision_probability" in row and str(row["cumulative_collision_probability"]) not in ("", "nan"):
                    scatter_y.append(float(row["cumulative_collision_probability"]))
                else:
                    scatter_y.append(1.0 - np.exp(-float(row["cumulative_tau"])))
            axes[5].scatter(
                [float(row["event_number"]) for row in scatter_rows],
                scatter_y,
                s=32,
                color="black",
                zorder=3,
            )
    axes[5].set_ylim(-0.02, 1.02)
    axes[5].set_xlabel("Event number")
    axes[5].set_ylabel("Cumulative probability")
    axes[5].set_title("Cumulative collision probability")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    if title is None:
        pid = rows[0].get("particle_id", "")
        stop = rows[-1].get("stop_reason", "")
        title = f"Particle {pid} trace, stop: {stop}"
    fig.suptitle(title)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    mpl.rcParams.update(old_rc)
    return output_path
