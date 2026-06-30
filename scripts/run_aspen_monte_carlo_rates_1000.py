from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspen import (  # noqa: E402
    MonteCarloConfig,
    Particle,
    flatten_trace_history,
    flux_weight_per_particle,
    plot_altitude_rate_profiles,
    summarize_monte_carlo_rows,
    trace_particle_xyz_until_stop,
    write_history_csv,
    write_rate_profile_csv,
)
from aspen.rates import altitude_bin_edges, compute_altitude_rate_profiles  # noqa: E402
from aspen.simulation.particle_state import particle_altitude_km, particle_energy_ev  # noqa: E402


def _collision_counts(collisions: list[dict[str, object]]) -> dict[str, int]:
    counts = {"n_elastic": 0, "n_ionization": 0, "n_lya": 0, "n_state_change": 0}
    for collision in collisions:
        reaction = str(collision["event"]["reaction"])
        key = f"n_{reaction}"
        if key in counts:
            counts[key] += 1
    return counts


def _run_particle_with_history(args: tuple[int, MonteCarloConfig]) -> tuple[dict[str, object], list[dict[str, object]]]:
    particle_id, config = args
    seed_sequence = np.random.SeedSequence([config.seed, int(particle_id)])
    rng = np.random.default_rng(seed_sequence)
    position = config.position_m()
    velocity = config.velocity_m_s()
    initial_R = float(rng.random())
    initial_particle = Particle(
        config.initial_projectile,
        velocity=velocity.copy(),
        position=position.copy(),
        R=initial_R,
        rng=np.random.default_rng(seed_sequence),
    )
    particle = Particle(
        config.initial_projectile,
        velocity=velocity.copy(),
        position=position.copy(),
        R=initial_R,
        rng=rng,
    )
    initial_energy_ev = particle_energy_ev(particle)
    result = trace_particle_xyz_until_stop(
        particle,
        targets=config.targets,
        safety_factor=config.safety_factor,
        max_step_m=config.max_step_m,
        min_energy_ev=config.min_energy_ev,
        min_altitude_km=config.min_altitude_km,
        max_altitude_km=config.max_altitude_km,
        max_collisions=config.max_collisions,
        max_steps_per_collision=config.max_steps_per_collision,
        rng=rng,
        solar=config.solar,
        ls=config.ls,
        include_hot_o=config.include_hot_o,
        cross_section_dir=config.cross_section_dir,
    )
    final_position = np.asarray(particle.position, dtype=float)
    stop = result["stop_condition"]
    summary = {
        "particle_id": int(particle_id),
        "initial_projectile": config.initial_projectile,
        "initial_charge_state": 0 if config.initial_projectile in ("H", "H-ENA", "ENA") else 1,
        "initial_energy_ev": initial_energy_ev,
        "initial_altitude_km": particle_altitude_km(initial_particle),
        "initial_x_m": float(position[0]),
        "initial_y_m": float(position[1]),
        "initial_z_m": float(position[2]),
        "initial_vx_m_s": float(velocity[0]),
        "initial_vy_m_s": float(velocity[1]),
        "initial_vz_m_s": float(velocity[2]),
        "final_projectile": particle.projectile,
        "final_charge_state": int(particle.charge_state),
        "final_energy_ev": particle_energy_ev(particle),
        "final_altitude_km": particle_altitude_km(particle),
        "final_x_m": float(final_position[0]),
        "final_y_m": float(final_position[1]),
        "final_z_m": float(final_position[2]),
        "n_steps": int(result["n_steps"]),
        "n_collisions": int(result["n_collisions"]),
        **_collision_counts(result["collisions"]),
        "stop_reasons": ";".join(stop["reasons"]),
    }
    history = flatten_trace_history(result, particle_id=particle_id, initial_particle=initial_particle)
    return summary, history


def run_detailed_ensemble(config: MonteCarloConfig, workers: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    jobs = [(i, config) for i in range(int(config.n_particles))]
    summaries = []
    histories = []
    if workers <= 1:
        for job in jobs:
            summary, history = _run_particle_with_history(job)
            summaries.append(summary)
            histories.extend(history)
        return summaries, histories

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(_run_particle_with_history, job) for job in jobs]
        for future in as_completed(futures):
            summary, history = future.result()
            summaries.append(summary)
            histories.extend(history)
    summaries.sort(key=lambda row: int(row["particle_id"]))
    histories.sort(key=lambda row: (int(row["particle_id"]), int(row["event_number"])))
    return summaries, histories


def write_csv(rows: list[dict[str, object]], filename: Path) -> Path:
    if not rows:
        raise ValueError("No rows to write.")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1000 ASPEN particles and compute altitude rate profiles."
    )
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sw-density-cm3", type=float, default=1.0)
    parser.add_argument("--sw-speed-km-s", type=float, default=400.0)
    parser.add_argument("--max-step-m", type=float, default=1000.0)
    parser.add_argument("--safety-factor", type=float, default=0.4)
    parser.add_argument("--max-collisions", type=int, default=2000)
    parser.add_argument("--max-steps-per-collision", type=int, default=100_000)
    parser.add_argument("--alt-bin-km", type=float, default=10.0)
    parser.add_argument("--mu-mode", choices=("absolute", "inward", "outward", "signed"), default="absolute")
    parser.add_argument("--output-dir", type=Path, default=Path("aspen_examples") / "monte_carlo_rates_1000")
    args = parser.parse_args()
    if args.max_step_m > 1000.0:
        raise ValueError("max_step_m must be <= 1000 m for ASPEN production runs.")

    config = MonteCarloConfig(
        n_particles=args.n_particles,
        initial_projectile="H-ENA",
        initial_altitude_km=600.0,
        initial_velocity_m_s=(-400_000.0, 0.0, 0.0),
        seed=args.seed,
        safety_factor=args.safety_factor,
        max_step_m=args.max_step_m,
        max_collisions=args.max_collisions,
        max_steps_per_collision=args.max_steps_per_collision,
    )
    sw_density_m3 = args.sw_density_cm3 * 1.0e6
    sw_speed_m_s = args.sw_speed_km_s * 1000.0
    flux_m2_s = sw_density_m3 * abs(sw_speed_m_s)
    weight_m2_s = flux_weight_per_particle(sw_density_m3, sw_speed_m_s, args.n_particles)

    summaries, histories = run_detailed_ensemble(config, workers=args.workers)
    rate_rows = compute_altitude_rate_profiles(
        histories,
        altitude_edges_km=altitude_bin_edges(100.0, 1000.0, args.alt_bin_km),
        weight_m2_s=weight_m2_s,
        mu_mode=args.mu_mode,
        solar=config.solar,
        ls=config.ls,
        include_hot_o=config.include_hot_o,
    )
    summary = summarize_monte_carlo_rows(summaries)

    out = args.output_dir
    summary_csv = write_csv(summaries, out / "particle_summary.csv")
    history_csv = write_history_csv(histories, out / "particle_history.csv")
    rate_csv = write_rate_profile_csv(rate_rows, out / "altitude_rate_profiles.csv")
    rate_png = plot_altitude_rate_profiles(rate_rows, out / "altitude_rate_profiles.png")

    print(f"n_particles={summary['n_particles']}")
    print(f"solar_wind_density_m3={sw_density_m3:.6e}")
    print(f"solar_wind_speed_m_s={sw_speed_m_s:.6e}")
    print(f"flux_m-2_s-1={flux_m2_s:.6e}")
    print(f"weight_m-2_s-1_per_particle={weight_m2_s:.6e}")
    print(f"final_energy_mean_ev={summary['final_energy_mean_ev']:.3f}")
    print(f"n_collisions_mean={summary['n_collisions_mean']:.3f}")
    print(f"stop_reasons={summary['stop_reasons']}")
    print(f"summary_csv={summary_csv.resolve()}")
    print(f"history_csv={history_csv.resolve()}")
    print(f"rate_csv={rate_csv.resolve()}")
    print(f"rate_png={rate_png.resolve()}")


if __name__ == "__main__":
    main()
