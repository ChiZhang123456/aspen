from __future__ import annotations

"""Local collision coefficient, frequency, and mean-free-path calculations.

For a projectile of speed `v`, the local collision coefficient is

`alpha(r, E) = sum_j n_j(r) sum_k sigma_j,k(E)`

with units of `m^-1`. The collision frequency is `v * alpha` in `s^-1`, and
the mean free path is `1 / alpha`.

Neutral densities come from `aspen.neutral_density_model`, which combines
MGITM cold neutrals with MAMPS hot atomic oxygen when `include_hot_o=True`.
"""

from pathlib import Path

import numpy as np

from aspen.constants import ELEMENTARY_CHARGE_C, HYDROGEN_MASS_KG

from aspen.neutral_density_model import neutral_density
from .cross_section import (
    DEFAULT_CROSS_SECTION_DIR,
    REACTIONS,
    TARGETS,
    cross_section,
    normalize_projectile,
    normalize_reaction,
    normalize_target,
)


def speed_from_energy(energy_ev: float) -> float:
    """Convert H or H+ kinetic energy in eV to speed in m/s."""
    return float(np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE_C / HYDROGEN_MASS_KG))


def collision_frequency(
    altitude_km: float,
    lon_deg: float,
    lat_deg: float,
    projectile: str,
    energy_ev: float,
    neutral_mat: str | Path | None = None,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
    targets: tuple[str, ...] = TARGETS,
    reactions: tuple[str, ...] = REACTIONS,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    include_hot_o: bool = True,
) -> dict[str, object]:
    """Compute local collision coefficient and collision frequency.

    `collision_coefficient_m-1` is `alpha = sum n sigma`.
    `collision_frequency_s-1` is `v alpha`.

    Neutral densities are evaluated with the packaged MGITM + MAMPS model.
    `neutral_mat` is kept only for compatibility with older scripts and is
    not used.
    """
    projectile_key = normalize_projectile(projectile)
    target_keys = tuple(normalize_target(target) for target in targets)
    reaction_keys = tuple(normalize_reaction(reaction) for reaction in reactions)

    # Keep the function signature stable for older code, but use the packaged
    # MGITM + MAMPS model for all density values.
    density = neutral_density(
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        altitude_km=altitude_km,
        species=target_keys,
        solar=solar,
        ls=ls,
        include_hot_o=include_hot_o,
    )
    speed_m_s = speed_from_energy(energy_ev)

    breakdown = []
    alpha_total = 0.0
    for target_key in target_keys:
        n_m3 = float(density[target_key])
        for reaction_key in reaction_keys:
            cs = cross_section(
                projectile_key,
                target_key,
                reaction_key,
                energy_ev,
                cross_section_dir,
            )
            alpha = n_m3 * float(cs["sigma_m2"])
            alpha_total += alpha
            breakdown.append(
                {
                    "target": target_key,
                    "reaction": reaction_key,
                    "density_m-3": n_m3,
                    "sigma_m2": float(cs["sigma_m2"]),
                    "alpha_m-1": alpha,
                    "frequency_s-1": speed_m_s * alpha,
                    "energy_loss_ev": float(cs["energy_loss_ev"]),
                }
            )

    mean_free_path_m = np.inf if alpha_total <= 0.0 else 1.0 / alpha_total
    return {
        "altitude_km": float(altitude_km),
        "lon_deg": float(lon_deg),
        "lat_deg": float(lat_deg),
        "solar": solar,
        "ls": int(str(ls).strip().lower().replace("ls", "")),
        "include_hot_o": bool(include_hot_o),
        "projectile": projectile_key,
        "energy_ev": float(energy_ev),
        "speed_m_s": speed_m_s,
        "neutral_density_m-3": density,
        "collision_coefficient_m-1": alpha_total,
        "collision_frequency_s-1": speed_m_s * alpha_total,
        "mean_free_path_km": mean_free_path_m / 1000.0,
        "breakdown": breakdown,
    }
