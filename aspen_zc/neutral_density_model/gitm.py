from __future__ import annotations

"""MGITM cold-neutral density model for ASPEN.

This module reads the packaged MGITM `.mat` files in
`aspen_zc/neutral_density_model/data`. These `.mat` files were converted from:

    D:/Work_Work/Mars/MAVEN/test_particle_jl/neutral/MGITM/MGITM_LS*_F*.dat

Main public functions
---------------------
`load_gitm_grid(solar, ls)`
    Load one MGITM case and return a `GITMGrid` object.

`gitm_density(lon_deg, lat_deg, altitude_km, solar, ls, species)`
    Return cold neutral densities and neutral temperature by 3D interpolation.

`gitm_density_xyz(x, y, z, solar, ls, species, position_unit)`
    Convert Mars-centered Sun-fixed Cartesian coordinates to model longitude,
    latitude, and altitude, then return MGITM densities.

Coordinate and unit conventions
-------------------------------
- Input longitude is model longitude in degrees. Longitudes are periodic.
- The model coordinate origin is Sun-fixed: `lon_deg = 0` and `lat_deg = 0`
  is the subsolar point. Use this convention when converting particle
  Cartesian coordinates into the neutral-atmosphere grid.
- For Cartesian input, the origin is Mars center, `+X` points to the subsolar
  point, `+Y` points toward `lon_deg = 90`, and `+Z` points north. The default
  coordinate unit is km.
- Input latitude is planetographic/geographic latitude in degrees.
- Input altitude is height above the surface in km.
- Returned densities are in m^-3.
- Returned `Tn` is in K.

Case selection
--------------
- `solar="solar_max"` or `200` loads F10.7 = 200.
- `solar="solar_moderate"` or `130` loads F10.7 = 130.
- `solar="solar_min"` or `70` loads F10.7 = 70.
- `ls` must be one of 0, 90, 180, or 270.

Interpolation and extrapolation
-------------------------------
Inside the MGITM altitude range, densities are interpolated in log-density
space on the 3D lon-lat-alt grid. `Tn` is interpolated linearly.

Above the MGITM top altitude, densities are hydrostatically extrapolated to
`GITM_TOP_KM = 10000 km`. The scale height is computed separately for each
species using the local top-layer `Tn` at the requested longitude and latitude.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from aspen_zc.constants import MARS_RADIUS_KM


DATA_DIR = Path(__file__).resolve().parent / "data"
GITM_TOP_KM = 10_000.0
GITM_SPECIES = ("CO2", "O", "O2", "N2", "CO")

KB_J_K = 1.380649e-23
MARS_G0_M_S2 = 3.71
AMU_KG = 1.66053906660e-27
MASS_KG = {
    "CO2": 44.01 * AMU_KG,
    "O": 15.999 * AMU_KG,
    "O2": 31.998 * AMU_KG,
    "N2": 28.014 * AMU_KG,
    "CO": 28.010 * AMU_KG,
}
FIELD_BY_SPECIES = {
    "CO2": "nCO2",
    "O": "nO",
    "O2": "nO2",
    "N2": "nN2",
    "CO": "nCO",
}


@dataclass(frozen=True)
class GITMGrid:
    """One packaged MGITM case.

    Arrays are stored as `(longitude, latitude, altitude)`.

    Attributes
    ----------
    lon_deg, lat_deg, alt_km
        Grid coordinates.
    fields
        Dictionary containing `Tn`, `Ti`, `Te`, `nCO2`, `nO`, `nO2`, `nN2`,
        and `nCO`.
    source_file
        Original MGITM `.dat` filename.
    filename
        Packaged `.mat` file used by this object.
    """

    lon_deg: np.ndarray
    lat_deg: np.ndarray
    alt_km: np.ndarray
    fields: dict[str, np.ndarray]
    source_file: str
    filename: Path


def _solar_to_f107(solar: str | int) -> int:
    """Normalize user-facing solar activity labels to F10.7 values."""
    key = str(solar).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in {"solarmax", "max", "f200", "200"}:
        return 200
    if key in {"solarmoderate", "moderate", "mod", "solarmean", "mean", "f130", "130"}:
        return 130
    if key in {"solarmin", "min", "f070", "f70", "70"}:
        return 70
    raise ValueError("solar must be solar_max, solar_moderate, solar_min, 200, 130, or 70")


def _normalize_ls(ls: int | str) -> int:
    """Normalize `ls` and require one of the packaged solar longitudes."""
    value = int(str(ls).strip().lower().replace("ls", ""))
    if value not in (0, 90, 180, 270):
        raise ValueError("ls must be one of 0, 90, 180, or 270")
    return value


def _mat_string(value: object, default: str = "unknown") -> str:
    """Extract a plain Python string from a MATLAB string-like value."""
    arr = np.ravel(value)
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(item, np.ndarray):
        item = np.ravel(item)[0]
    return str(item)


def cartesian_to_lon_lat_alt(
    x,
    y=None,
    z=None,
    *,
    position_unit: str = "km",
    mars_radius_km: float = MARS_RADIUS_KM,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Convert Sun-fixed Mars-centered Cartesian coordinates to model lon-lat-alt.

    The model coordinate system is Sun-fixed: `+X` is the subsolar direction,
    `+Y` is model east at `lon_deg = 90`, and `+Z` is north. Therefore a point
    at `[MARS_RADIUS_KM + h, 0, 0]` has `lon_deg = 0`, `lat_deg = 0`, and
    `altitude_km = h`.

    Parameters
    ----------
    x, y, z
        Cartesian coordinates. Either pass three separate arrays or pass `x` as
        an array whose last dimension is length 3.
    position_unit
        `km` by default. Use `m` if the input coordinates are in meters.
    mars_radius_km
        Reference Mars radius used to compute altitude.

    Returns
    -------
    lon_deg, lat_deg, altitude_km
        Model longitude, model latitude, and altitude above the surface.
    """
    if y is None and z is None:
        pos = np.asarray(x, dtype=np.float64)
        if pos.shape[-1] != 3:
            raise ValueError("Cartesian position array must have last dimension length 3.")
        x_arr = pos[..., 0]
        y_arr = pos[..., 1]
        z_arr = pos[..., 2]
    elif y is not None and z is not None:
        x_arr, y_arr, z_arr = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        )
    else:
        raise ValueError("Pass either x, y, z together or one (..., 3) position array.")

    unit = str(position_unit).strip().lower()
    if unit in {"m", "meter", "meters"}:
        x_arr = x_arr / 1000.0
        y_arr = y_arr / 1000.0
        z_arr = z_arr / 1000.0
    elif unit not in {"km", "kilometer", "kilometers"}:
        raise ValueError("position_unit must be 'km' or 'm'.")

    radius = np.sqrt(x_arr * x_arr + y_arr * y_arr + z_arr * z_arr)
    if np.any(radius <= 0.0):
        raise ValueError("Cartesian position must have nonzero radius.")

    lon_deg = np.degrees(np.arctan2(y_arr, x_arr)) % 360.0
    lat_deg = np.degrees(np.arcsin(np.clip(z_arr / radius, -1.0, 1.0)))
    altitude_km = radius - mars_radius_km

    if np.ndim(lon_deg) == 0:
        return float(lon_deg), float(lat_deg), float(altitude_km)
    return lon_deg, lat_deg, altitude_km


@lru_cache(maxsize=12)
def load_gitm_grid(solar: str | int = "solar_min", ls: int | str = 0) -> GITMGrid:
    """Load one packaged MGITM grid.

    Examples
    --------
    >>> grid = load_gitm_grid(solar="solar_max", ls=270)
    >>> grid.fields["nO"].shape
    (72, 36, 62)
    """
    f107 = _solar_to_f107(solar)
    ls_value = _normalize_ls(ls)
    filename = DATA_DIR / f"gitm_ls{ls_value:03d}_f{f107:03d}.mat"
    if not filename.exists():
        raise FileNotFoundError(filename)
    data = loadmat(filename)
    fields = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in ("Tn", "Ti", "Te", "nCO2", "nO", "nO2", "nN2", "nCO")
    }
    return GITMGrid(
        lon_deg=np.asarray(data["lon_deg"], dtype=np.float64).ravel(),
        lat_deg=np.asarray(data["lat_deg"], dtype=np.float64).ravel(),
        alt_km=np.asarray(data["alt_km"], dtype=np.float64).ravel(),
        fields=fields,
        source_file=_mat_string(data.get("source_file", np.array(["unknown"]))),
        filename=filename,
    )


def _bracket(grid: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper bracketing indices and linear interpolation weights."""
    idx2 = np.searchsorted(grid, value, side="right")
    idx2 = np.clip(idx2, 1, len(grid) - 1)
    idx1 = idx2 - 1
    below = value <= grid[0]
    above = value >= grid[-1]
    idx1 = np.where(below, 0, np.where(above, len(grid) - 1, idx1))
    idx2 = np.where(below, 0, np.where(above, len(grid) - 1, idx2))
    denom = np.where(idx1 == idx2, 1.0, grid[idx2] - grid[idx1])
    weight = np.where(idx1 == idx2, 0.0, (value - grid[idx1]) / denom)
    return idx1, idx2, weight


def _lon_bracket(grid: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longitude bracketing with periodic wrapping."""
    lon = np.mod(lon_deg - grid[0], 360.0) + grid[0]
    idx2 = np.searchsorted(grid, lon, side="right")
    idx1 = idx2 - 1
    idx1 = np.where(idx1 < 0, len(grid) - 1, idx1)
    idx2 = np.where(idx2 >= len(grid), 0, idx2)
    idx1 = np.clip(idx1, 0, len(grid) - 1)
    idx2 = np.clip(idx2, 0, len(grid) - 1)
    x1 = grid[idx1]
    x2 = np.where(idx2 == 0, grid[0] + 360.0, grid[idx2])
    x = np.where(lon < x1, lon + 360.0, lon)
    weight = np.where(idx1 == idx2, 0.0, (x - x1) / (x2 - x1))
    return idx1, idx2, weight


def _interp3(grid: GITMGrid, field: str, lon_deg, lat_deg, alt_km, *, log_density: bool) -> np.ndarray:
    """Trilinear interpolation on the packaged MGITM grid."""
    values = grid.fields[field]
    if log_density:
        values = np.log(np.maximum(values, 1.0e-300))

    i0, i1, wx = _lon_bracket(grid.lon_deg, lon_deg)
    j0, j1, wy = _bracket(grid.lat_deg, lat_deg)
    k0, k1, wz = _bracket(grid.alt_km, alt_km)

    c000 = values[i0, j0, k0]
    c100 = values[i1, j0, k0]
    c010 = values[i0, j1, k0]
    c110 = values[i1, j1, k0]
    c001 = values[i0, j0, k1]
    c101 = values[i1, j0, k1]
    c011 = values[i0, j1, k1]
    c111 = values[i1, j1, k1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    out = c0 * (1.0 - wz) + c1 * wz
    return np.exp(out) if log_density else out


def _scale_height_km(temperature_k: np.ndarray, species: str, altitude_km: float) -> np.ndarray:
    """Hydrostatic scale height in km for one neutral species."""
    radius_ratio = MARS_RADIUS_KM / (MARS_RADIUS_KM + altitude_km)
    g_alt = MARS_G0_M_S2 * radius_ratio**2
    return KB_J_K * temperature_k / (MASS_KG[species] * g_alt) / 1000.0


def _density_one_species(grid: GITMGrid, species: str, lon, lat, alt) -> np.ndarray:
    """Interpolate or extrapolate one MGITM density field."""
    field = FIELD_BY_SPECIES[species]
    model_top = grid.alt_km[-1]
    direct_alt = np.minimum(alt, model_top)
    density = _interp3(grid, field, lon, lat, direct_alt, log_density=True)

    high = alt > model_top
    if np.any(high):
        top_alt = np.full_like(alt, model_top)
        top_density = _interp3(grid, field, lon, lat, top_alt, log_density=True)
        top_temperature = _interp3(grid, "Tn", lon, lat, top_alt, log_density=False)
        scale_height = _scale_height_km(top_temperature, species, model_top)
        extrapolated = top_density * np.exp(-(alt - model_top) / scale_height)
        density = np.where(high, extrapolated, density)
    return density


def gitm_density(
    lon_deg,
    lat_deg,
    altitude_km,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    species: Iterable[str] = GITM_SPECIES,
) -> dict[str, float | np.ndarray]:
    """Return MGITM neutral density in m^-3 by 3D interpolation.

    Above the MGITM model top, densities are hydrostatically extrapolated to
    10000 km with the local top-layer neutral temperature.

    Examples
    --------
    >>> rho = gitm_density(27.5, 0.0, 200.0, solar="solar_max", ls=270)
    >>> rho["O"]  # cold MGITM O density, m^-3
    >>> rho["Tn"] # neutral temperature, K

    Vector altitude input is also supported:

    >>> rho = gitm_density(27.5, 0.0, [150.0, 250.0, 500.0], solar=70, ls=0)
    >>> rho["CO2"].shape
    (3,)
    """
    alt, lon, lat = np.broadcast_arrays(
        np.asarray(altitude_km, dtype=np.float64),
        np.asarray(lon_deg, dtype=np.float64),
        np.asarray(lat_deg, dtype=np.float64),
    )
    if np.any(alt > GITM_TOP_KM):
        raise ValueError(f"altitude_km must be <= {GITM_TOP_KM:g} km")

    grid = load_gitm_grid(solar=solar, ls=ls)
    out: dict[str, np.ndarray] = {}
    for raw_name in species:
        name = str(raw_name).strip()
        if name not in GITM_SPECIES:
            raise ValueError(f"Unknown GITM species {raw_name!r}. Use {GITM_SPECIES}.")
        out[name] = _density_one_species(grid, name, lon, lat, alt)

    tn_alt = np.minimum(alt, grid.alt_km[-1])
    out["Tn"] = _interp3(grid, "Tn", lon, lat, tn_alt, log_density=False)

    scalar = np.ndim(altitude_km) == 0 and np.ndim(lon_deg) == 0 and np.ndim(lat_deg) == 0
    if scalar:
        return {key: float(value) for key, value in out.items()}
    return out


def gitm_density_xyz(
    x,
    y=None,
    z=None,
    *,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    species: Iterable[str] = GITM_SPECIES,
    position_unit: str = "km",
    mars_radius_km: float = MARS_RADIUS_KM,
) -> dict[str, float | np.ndarray]:
    """Return MGITM density from Sun-fixed Mars-centered Cartesian coordinates.

    This is a convenience wrapper around `gitm_density`. The Cartesian
    convention is:

    - origin at Mars center
    - `+X` points to the subsolar point, `lon_deg = 0`, `lat_deg = 0`
    - `+Y` points toward `lon_deg = 90`
    - `+Z` points north

    Examples
    --------
    >>> rho = gitm_density_xyz(MARS_RADIUS_KM + 200.0, 0.0, 0.0)
    >>> rho["CO2"]  # density at lon=0, lat=0, alt=200 km

    A vector position is also accepted:

    >>> rho = gitm_density_xyz([MARS_RADIUS_KM + 200.0, 0.0, 0.0])
    """
    lon_deg, lat_deg, altitude_km = cartesian_to_lon_lat_alt(
        x,
        y,
        z,
        position_unit=position_unit,
        mars_radius_km=mars_radius_km,
    )
    return gitm_density(
        lon_deg,
        lat_deg,
        altitude_km,
        solar=solar,
        ls=ls,
        species=species,
    )
