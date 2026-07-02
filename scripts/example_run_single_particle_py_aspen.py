from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_aspen import (  # noqa: E402
    Particle,
    flatten_trace_history,
    particle_altitude_km,
    particle_energy_ev,
    plot_particle_trace_history,
    trace_particle_xyz_until_stop,
    write_history_csv,
)
from py_aspen.constants import MARS_RADIUS_KM  # noqa: E402


def main() -> None:
    output_dir = PROJECT_ROOT / "aspen_examples" / "single_particle_quick_start"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    position_m = np.array([(MARS_RADIUS_KM + 600.0) * 1000.0, 0.0, 0.0])
    velocity_m_s = np.array([-400_000.0, 0.0, 0.0])

    initial_particle = Particle(
        "H-ENA",
        velocity=velocity_m_s.copy(),
        position=position_m.copy(),
        R=0.01,
        rng=np.random.default_rng(7),
    )
    particle = Particle(
        "H-ENA",
        velocity=velocity_m_s.copy(),
        position=position_m.copy(),
        R=0.01,
        rng=rng,
    )

    result = trace_particle_xyz_until_stop(
        particle,
        max_step_m=1000.0,
        rng=rng,
    )

    rows = flatten_trace_history(result, particle_id=0, initial_particle=initial_particle)
    history_csv = write_history_csv(rows, output_dir / "single_particle_history.csv")
    figure_file = plot_particle_trace_history(
        rows,
        output_dir / "single_particle_trace.png",
        particle_id=0,
        title="Single H-ENA particle, 600 km, V = [-400, 0, 0] km/s",
        font_size=17.0,
    )

    print(f"n_steps={result['n_steps']}")
    print(f"n_collisions={result['n_collisions']}")
    print(f"stop_reasons={result['stop_condition']['reasons']}")
    print(f"final_projectile={particle.projectile}")
    print(f"final_charge_state={particle.charge_state}")
    print(f"final_energy_ev={particle_energy_ev(particle):.3f}")
    print(f"final_altitude_km={particle_altitude_km(particle):.3f}")
    print(f"history_csv={history_csv}")
    print(f"figure={figure_file}")


if __name__ == "__main__":
    main()
