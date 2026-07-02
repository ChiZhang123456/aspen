from __future__ import annotations

"""Read and interpolate ASPEN H/H+ collision cross sections.

This module is the single source of truth for the packaged collision
cross-section tables. The input text files are stored in the same directory as
this file and use `cm^2`; all public functions return SI cross sections in
`m^2`.

Supported projectiles are fast neutral hydrogen (`H`, also accepted as
`H-ENA`) and protons (`H+`, also accepted as `Hplus` or `proton`). Supported
targets are `CO2`, `O`, and `N2`. The canonical reaction names are:

`state_change`
    For H impact this is electron stripping, H -> H+. For H+ impact this is
    charge exchange, H+ -> H.
`ionization`
    Impact ionization of the neutral target.
`lya`
    H Ly-alpha related excitation channel.
`elastic`
    Elastic scattering.
`halpha`
    H Balmer-alpha channel from the reference table. This is available for
    cross-section lookup but is not used by the default Monte Carlo sampler.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


DEFAULT_CROSS_SECTION_DIR = Path(__file__).resolve().parent

TARGETS = ("CO2", "O", "N2")
TARGET_INDEX = {"CO2": 0, "O": 1, "N2": 2}
PROJECTILES = ("H+", "H")
REACTIONS = ("state_change", "ionization", "lya", "elastic", "halpha")

PROJECTILE_INDEX = {"H+": 0, "Hplus": 0, "proton": 0, "H": 1, "H-ENA": 1, "ENA": 1}
REACTION_INDEX = {
    "state_change": 0,
    "charge_exchange": 0,
    "electron_stripping": 0,
    "ionization": 1,
    "lya": 2,
    "ly-alpha": 2,
    "lyman_alpha": 2,
    "elastic": 3,
    "halpha": 4,
    "h-alpha": 4,
}

HPLUS_COLUMN_MAP = {
    "sigma_10": "state_change",
    "sigma_ip": "ionization",
    "sigma_La": "lya",
    "sigma_el": "elastic",
    "sigma_Ha": "halpha",
}

H_COLUMN_MAP = {
    "sigma_01": "state_change",
    "sigma_ia": "ionization",
    "sigma_La": "lya",
    "sigma_el": "elastic",
    "sigma_Ha": "halpha",
}


@dataclass(frozen=True)
class CrossSectionSet:
    """Cross-section tables interpolated onto one shared energy grid.

    Attributes
    ----------
    energy_ev
        Common energy grid in eV.
    sigma_m2
        Array with shape `(projectile, target, reaction, energy)`.
    energy_loss_ev
        Q-value magnitudes parsed from the table headers, in eV. Elastic
        channels have zero loss from the header.
    source_dir
        Directory containing the `.txt` cross-section tables.
    """

    energy_ev: np.ndarray
    sigma_m2: np.ndarray
    energy_loss_ev: np.ndarray
    source_dir: Path


def normalize_projectile(projectile: str) -> str:
    """Normalize user-facing projectile aliases to `H+` or `H`."""
    key = projectile.strip()
    if key not in PROJECTILE_INDEX:
        raise ValueError(f"Unknown projectile {projectile!r}. Use 'H' or 'H+'.")
    return PROJECTILES[PROJECTILE_INDEX[key]]


def normalize_target(target: str) -> str:
    """Normalize and validate a neutral target name."""
    key = target.strip()
    if key not in TARGET_INDEX:
        raise ValueError(f"Unknown target {target!r}. Use one of {TARGETS}.")
    return TARGETS[TARGET_INDEX[key]]


def normalize_reaction(reaction: str) -> str:
    """Normalize reaction aliases to the canonical reaction names."""
    key = reaction.strip()
    if key not in REACTION_INDEX:
        raise ValueError(f"Unknown reaction {reaction!r}. Use one of {REACTIONS}.")
    return REACTIONS[REACTION_INDEX[key]]


def cross_section_files(data_dir: Path) -> dict[tuple[str, str], Path]:
    """Return the packaged text file for every projectile-target pair."""
    return {
        ("H+", "CO2"): data_dir / "Hplus_CO2_cross_sections.txt",
        ("H+", "O"): data_dir / "Hplus_O_cross_sections.txt",
        ("H+", "N2"): data_dir / "Hplus_N2_cross_sections.txt",
        ("H", "CO2"): data_dir / "H_CO2_cross_sections.txt",
        ("H", "O"): data_dir / "H_O_cross_sections.txt",
        ("H", "N2"): data_dir / "H_N2_cross_sections.txt",
    }


def parse_q_values(filename: Path, projectile: str) -> dict[str, float]:
    """Parse reaction Q values from a cross-section file header.

    The reference headers list Q values in eV. The model stores the magnitude
    of the loss, so negative Q values become positive energy-loss values.
    """
    column_map = HPLUS_COLUMN_MAP if projectile == "H+" else H_COLUMN_MAP
    pattern = re.compile(r"#\s*(sigma_\w+):.*?Q\s*=\s*([+-]?\d+(?:\.\d+)?)\s*eV")
    q_values: dict[str, float] = {}
    with filename.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            match = pattern.search(line)
            if match and match.group(1) in column_map:
                q_values[column_map[match.group(1)]] = abs(float(match.group(2)))
    return q_values


@lru_cache(maxsize=8)
def load_cross_sections(cross_section_dir: str) -> CrossSectionSet:
    """Load all cross-section files and interpolate them onto one log grid.

    The raw tables have different column meanings for H and H+. `columns`
    below maps each file's table columns into the canonical reaction order:
    `state_change`, `ionization`, `lya`, `elastic`, `halpha`.
    """
    data_dir = Path(cross_section_dir)
    energy_grid = np.logspace(0.0, 7.0, 768)
    sigma = np.zeros((len(PROJECTILES), len(TARGETS), len(REACTIONS), energy_grid.size))
    energy_loss = np.zeros((len(PROJECTILES), len(TARGETS), len(REACTIONS)))
    for (projectile, target), filename in cross_section_files(data_dir).items():
        if not filename.exists():
            raise FileNotFoundError(filename)
        data = np.loadtxt(filename, comments="#", skiprows=9)
        source_energy = data[:, 0]
        columns = [1, 2, 3, 4, 5] if projectile == "H+" else [2, 4, 3, 1, 5]
        ip = PROJECTILE_INDEX[projectile]
        it = TARGET_INDEX[target]
        for ir, column in enumerate(columns):
            sigma[ip, it, ir, :] = (
                np.interp(energy_grid, source_energy, data[:, column], left=0.0, right=0.0)
                * 1.0e-4
            )
        for reaction, q_ev in parse_q_values(filename, projectile).items():
            energy_loss[ip, it, REACTION_INDEX[reaction]] = q_ev
    return CrossSectionSet(
        energy_ev=energy_grid,
        sigma_m2=sigma,
        energy_loss_ev=energy_loss,
        source_dir=data_dir,
    )


def cross_section(
    projectile: str,
    target: str,
    reaction: str,
    energy_ev: float,
    cross_section_dir: str | Path = DEFAULT_CROSS_SECTION_DIR,
) -> dict[str, float | str]:
    """Return one cross section and header-derived energy loss.

    Parameters
    ----------
    projectile
        `H`, `H-ENA`, `H+`, `Hplus`, or `proton`.
    target
        `CO2`, `O`, or `N2`.
    reaction
        Reaction name or accepted alias.
    energy_ev
        Projectile kinetic energy in eV.

    Returns
    -------
    dict
        Includes the normalized names, `sigma_m2`, `energy_loss_ev`, and
        source directory.
    """
    projectile_key = normalize_projectile(projectile)
    target_key = normalize_target(target)
    reaction_key = normalize_reaction(reaction)
    cs = load_cross_sections(str(Path(cross_section_dir)))
    ip = PROJECTILE_INDEX[projectile_key]
    it = TARGET_INDEX[target_key]
    ir = REACTION_INDEX[reaction_key]
    sigma_m2 = float(np.interp(energy_ev, cs.energy_ev, cs.sigma_m2[ip, it, ir, :]))
    return {
        "projectile": projectile_key,
        "target": target_key,
        "reaction": reaction_key,
        "energy_ev": float(energy_ev),
        "sigma_m2": sigma_m2,
        "energy_loss_ev": float(cs.energy_loss_ev[ip, it, ir]),
        "source_dir": str(cs.source_dir),
    }
