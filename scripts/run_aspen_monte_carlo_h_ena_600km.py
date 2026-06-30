from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspen import MonteCarloConfig, run_one_monte_carlo_particle, summarize_monte_carlo_rows


def _run_particle(args: tuple[int, MonteCarloConfig]) -> dict[str, object]:
    particle_id, config = args
    return run_one_monte_carlo_particle(particle_id, config)


def run_ensemble(config: MonteCarloConfig, workers: int = 1) -> list[dict[str, object]]:
    """Run Monte Carlo particles and return compact summary rows."""
    jobs = [(i, config) for i in range(int(config.n_particles))]
    if workers <= 1:
        return [_run_particle(job) for job in jobs]

    rows = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(_run_particle, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: int(row["particle_id"]))


def write_csv(rows: list[dict[str, object]], filename: Path) -> None:
    """Write compact Monte Carlo rows to CSV."""
    if not rows:
        raise ValueError("No rows to write.")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, object]], filename: Path) -> None:
    """Plot basic ensemble diagnostics."""
    final_energy = np.asarray([float(row["final_energy_ev"]) for row in rows])
    final_altitude = np.asarray([float(row["final_altitude_km"]) for row in rows])
    n_collisions = np.asarray([int(row["n_collisions"]) for row in rows])
    n_state_change = np.asarray([int(row["n_state_change"]) for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    axes[0, 0].hist(final_energy, bins=40, color="tab:red", alpha=0.8)
    axes[0, 0].set_xlabel("Final energy (eV)")
    axes[0, 0].set_ylabel("Counts")

    axes[0, 1].hist(final_altitude, bins=40, color="black", alpha=0.8)
    axes[0, 1].axvline(100.0, color="tab:red", ls="--", lw=1.0)
    axes[0, 1].axvline(1000.0, color="tab:red", ls="--", lw=1.0)
    axes[0, 1].set_xlabel("Final altitude (km)")
    axes[0, 1].set_ylabel("Counts")

    axes[1, 0].hist(n_collisions, bins=40, color="tab:blue", alpha=0.8)
    axes[1, 0].set_xlabel("Number of collisions")
    axes[1, 0].set_ylabel("Counts")

    axes[1, 1].hist(n_state_change, bins=40, color="tab:green", alpha=0.8)
    axes[1, 1].set_xlabel("Number of state-change collisions")
    axes[1, 1].set_ylabel("Counts")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("ASPEN Monte Carlo, H-ENA, 600 km, V = [-400, 0, 0] km/s")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASPEN Monte Carlo H-ENA particles from 600 km."
    )
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-step-m", type=float, default=1000.0)
    parser.add_argument("--max-collisions", type=int, default=10_000)
    parser.add_argument("--max-steps-per-collision", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aspen_examples") / "monte_carlo_h_ena_600km",
    )
    args = parser.parse_args()

    config = MonteCarloConfig(
        n_particles=args.n_particles,
        initial_projectile="H-ENA",
        initial_altitude_km=600.0,
        initial_velocity_m_s=(-400_000.0, 0.0, 0.0),
        seed=args.seed,
        max_step_m=args.max_step_m,
        max_collisions=args.max_collisions,
        max_steps_per_collision=args.max_steps_per_collision,
    )

    rows = run_ensemble(config, workers=args.workers)
    summary = summarize_monte_carlo_rows(rows)

    csv_file = args.output_dir / "monte_carlo_h_ena_600km_summary.csv"
    figure_file = args.output_dir / "monte_carlo_h_ena_600km_summary.png"
    write_csv(rows, csv_file)
    plot_summary(rows, figure_file)

    print(f"n_particles={summary['n_particles']}")
    print(f"final_energy_mean_ev={summary['final_energy_mean_ev']:.3f}")
    print(f"final_energy_median_ev={summary['final_energy_median_ev']:.3f}")
    print(f"final_altitude_mean_km={summary['final_altitude_mean_km']:.3f}")
    print(f"n_collisions_mean={summary['n_collisions_mean']:.3f}")
    print(f"n_collisions_median={summary['n_collisions_median']:.3f}")
    print(f"stop_reasons={summary['stop_reasons']}")
    print(f"csv={csv_file.resolve()}")
    print(f"figure={figure_file.resolve()}")


if __name__ == "__main__":
    main()
