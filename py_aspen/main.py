from __future__ import annotations

"""Run one py_aspen particle example and plot the result.

Run from the project root with:

    python -m py_aspen
"""

from pathlib import Path

import numpy as np

from py_aspen.constants import MARS_RADIUS_KM
from py_aspen import (
    Particle,
    flatten_trace_history,
    particle_altitude_km,
    particle_energy_ev,
    plot_particle_trace_history,
    trace_particle_xyz_until_stop,
    write_history_csv,
)


def run_single_particle_example(
    output_dir: str | Path = "aspen_examples/single_particle_main",
    seed: int = 7,
    max_step_m: float = 1000.0,
) -> dict[str, object]:
    """Trace one H-ENA particle from 600 km and save CSV plus figure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    position_m = np.array([(MARS_RADIUS_KM + 600.0) * 1000.0, 0.0, 0.0])
    velocity_m_s = np.array([-400_000.0, 0.0, 0.0])
    initial_R = float(rng.random())

    initial_particle = Particle(
        "H-ENA",
        velocity=velocity_m_s.copy(),
        position=position_m.copy(),
        R=initial_R,
        rng=np.random.default_rng(seed),
    )
    particle = Particle(
        "H-ENA",
        velocity=velocity_m_s.copy(),
        position=position_m.copy(),
        R=initial_R,
        rng=rng,
    )

    result = trace_particle_xyz_until_stop(
        particle,
        max_step_m=max_step_m,
        rng=rng,
    )

    rows = flatten_trace_history(result, particle_id=0, initial_particle=initial_particle)
    history_csv = write_history_csv(rows, output_path / "single_particle_history.csv")
    figure_file = plot_particle_trace_history(
        rows,
        output_path / "single_particle_trace.png",
        particle_id=0,
        title="Single H-ENA particle, 600 km, V = [-400, 0, 0] km/s",
        font_size=17.0,
    )

    return {
        "particle": particle,
        "result": result,
        "history_csv": history_csv,
        "figure_file": figure_file,
        "final_energy_ev": particle_energy_ev(particle),
        "final_altitude_km": particle_altitude_km(particle),
    }


def main() -> None:
    out = run_single_particle_example()
    particle = out["particle"]
    result = out["result"]
    print(f"n_steps={result['n_steps']}")
    print(f"n_collisions={result['n_collisions']}")
    print(f"stop_reasons={result['stop_condition']['reasons']}")
    print(f"final_projectile={particle.projectile}")
    print(f"final_charge_state={particle.charge_state}")
    print(f"final_energy_ev={out['final_energy_ev']:.3f}")
    print(f"final_altitude_km={out['final_altitude_km']:.3f}")
    print(f"history_csv={out['history_csv']}")
    print(f"figure={out['figure_file']}")


if __name__ == "__main__":
    main()
