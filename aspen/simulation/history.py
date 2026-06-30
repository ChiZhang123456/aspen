from __future__ import annotations

"""Convert trace results into table-like history rows."""

from pathlib import Path
import csv

import numpy as np

from aspen.constants import MARS_RADIUS_KM
from aspen.particle_initialization import Particle

from .particle_state import particle_energy_ev, particle_altitude_km


def _altitude_km(position_m: np.ndarray) -> float:
    return float(np.linalg.norm(position_m) / 1000.0 - MARS_RADIUS_KM)


def _velocity_fields(position_m: np.ndarray, velocity_m_s: np.ndarray) -> dict[str, float]:
    velocity = np.asarray(velocity_m_s, dtype=float)
    position = np.asarray(position_m, dtype=float)
    speed = float(np.linalg.norm(velocity))
    radius = float(np.linalg.norm(position))
    radial_velocity = float(np.dot(velocity, position / radius)) if speed > 0.0 and radius > 0.0 else np.nan
    inward_mu = float(max(0.0, -radial_velocity / speed)) if speed > 0.0 and np.isfinite(radial_velocity) else np.nan
    return {
        "vx_m_s": float(velocity[0]),
        "vy_m_s": float(velocity[1]),
        "vz_m_s": float(velocity[2]),
        "speed_m_s": speed,
        "radial_velocity_m_s": radial_velocity,
        "inward_mu": inward_mu,
    }


def _collision_reaction_value(collision: dict[str, object], key: str, default: float = np.nan) -> float:
    reaction_result = collision.get("reaction_result", {})
    if isinstance(reaction_result, dict) and key in reaction_result:
        return float(reaction_result[key])
    return float(default)


def flatten_trace_history(
    result: dict[str, object],
    particle_id: int = 0,
    initial_particle: Particle | None = None,
) -> list[dict[str, object]]:
    """Flatten one trace result into rows for CSV output and plotting.

    Rows include transport steps and collision events. Collision rows include
    reaction, target, energy loss, scattering angle, and the random number
    reset used for the next path segment.
    """
    rows: list[dict[str, object]] = []
    event_number = 0
    collision_number = 0
    cumulative_time_s = 0.0
    cumulative_path_m = 0.0

    if initial_particle is not None:
        pos = np.asarray(initial_particle.position, dtype=float)
        vel = np.asarray(initial_particle.velocity, dtype=float)
        energy = particle_energy_ev(initial_particle)
        rows.append(
            {
                "particle_id": int(particle_id),
                "event_number": event_number,
                "segment_index": 0,
                "event_type": "initial",
                "collision_number": 0,
                "projectile": initial_particle.projectile,
                "charge_state": int(initial_particle.charge_state),
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "z_m": float(pos[2]),
                "x_km": float(pos[0] / 1000.0),
                "y_km": float(pos[1] / 1000.0),
                "z_km": float(pos[2] / 1000.0),
                **_velocity_fields(pos, vel),
                "altitude_km": particle_altitude_km(initial_particle),
                "dl_m": 0.0,
                "dt_s": 0.0,
                "cumulative_time_s": 0.0,
                "cumulative_path_km": 0.0,
                "energy_before_ev": energy,
                "energy_after_ev": energy,
                "energy_loss_ev": 0.0,
                "reaction": "",
                "target": "",
                "scattering_angle_deg": np.nan,
                "random_before_collision": float(initial_particle.R),
                "random_after_collision": float(initial_particle.R),
                "threshold_tau": -np.log(float(initial_particle.R)),
                "cumulative_tau": 0.0,
                "cumulative_collision_probability": 0.0,
                "stop_reason": "",
            }
        )

    for segment_index, segment in enumerate(result.get("segments", []), start=1):
        for step in segment.get("step_history", []):
            event_number += 1
            cumulative_time_s += float(step["dt_s"])
            cumulative_path_m += float(step["dl_m"])
            pos = np.asarray(step["new_position_m"], dtype=float)
            vel = np.asarray(step["velocity_m_s"], dtype=float)
            tau = float(step["cumulative_tau_after"])
            rows.append(
                {
                    "particle_id": int(particle_id),
                    "event_number": event_number,
                    "segment_index": segment_index,
                    "event_type": "transport",
                    "collision_number": collision_number,
                    "projectile": step["projectile"],
                    "charge_state": int(step["charge_state"]),
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "z_m": float(pos[2]),
                    "x_km": float(pos[0] / 1000.0),
                    "y_km": float(pos[1] / 1000.0),
                    "z_km": float(pos[2] / 1000.0),
                    **_velocity_fields(pos, vel),
                    "altitude_km": _altitude_km(pos),
                    "dl_m": float(step["dl_m"]),
                    "dt_s": float(step["dt_s"]),
                    "cumulative_time_s": cumulative_time_s,
                    "cumulative_path_km": float(cumulative_path_m / 1000.0),
                    "energy_before_ev": float(step["energy_before_ev"]),
                    "energy_after_ev": float(step["energy_after_ev"]),
                    "energy_loss_ev": float(step["energy_loss_ev"]),
                    "reaction": "",
                    "target": "",
                    "scattering_angle_deg": np.nan,
                    "random_before_collision": np.nan,
                    "random_after_collision": np.nan,
                    "threshold_tau": float(step["threshold_tau"]),
                    "cumulative_tau": tau,
                    "cumulative_collision_probability": float(1.0 - np.exp(-tau)),
                    "stop_reason": "",
                }
            )

        collision = segment.get("collision")
        if collision is not None:
            event_number += 1
            collision_number += 1
            pos = np.asarray(collision["position_m"], dtype=float)
            vel = np.asarray(collision.get("velocity_after_m_s", [np.nan, np.nan, np.nan]), dtype=float)
            tau = float(collision.get("threshold_tau_before_collision", np.nan))
            rows.append(
                {
                    "particle_id": int(particle_id),
                    "event_number": event_number,
                    "segment_index": segment_index,
                    "event_type": "collision",
                    "collision_number": collision_number,
                    "projectile": "H+" if int(collision["charge_state"]) == 1 else "H",
                    "charge_state": int(collision["charge_state"]),
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "z_m": float(pos[2]),
                    "x_km": float(pos[0] / 1000.0),
                    "y_km": float(pos[1] / 1000.0),
                    "z_km": float(pos[2] / 1000.0),
                    **_velocity_fields(pos, vel),
                    "altitude_km": _altitude_km(pos),
                    "dl_m": 0.0,
                    "dt_s": 0.0,
                    "cumulative_time_s": cumulative_time_s,
                    "cumulative_path_km": float(cumulative_path_m / 1000.0),
                    "energy_before_ev": float(collision["energy_before_ev"]),
                    "energy_after_ev": float(collision["energy_after_ev"]),
                    "energy_loss_ev": float(collision["energy_loss_ev"]),
                    "reaction": str(collision["event"]["reaction"]),
                    "target": str(collision["event"]["target"]),
                    "scattering_angle_deg": _collision_reaction_value(collision, "scattering_angle_deg"),
                    "random_before_collision": float(collision["random_before_collision"]),
                    "random_after_collision": float(collision["random_after_collision"]),
                    "threshold_tau": tau,
                    "cumulative_tau": tau,
                    "cumulative_collision_probability": float(1.0 - np.exp(-tau)),
                    "stop_reason": "",
                }
            )

    stop = result.get("stop_condition", {})
    if rows and isinstance(stop, dict):
        rows[-1]["stop_reason"] = ";".join(stop.get("reasons", ()))
    return rows


def write_history_csv(rows: list[dict[str, object]], filename: str | Path) -> Path:
    """Write flattened trace rows to CSV."""
    if not rows:
        raise ValueError("No history rows to write.")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
