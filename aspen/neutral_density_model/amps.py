from __future__ import annotations

"""MAMPS hot-oxygen density model for ASPEN.

This module reads the packaged MAMPS `.mat` files in
`aspen/neutral_density_model/data`. These `.mat` files were converted from:

    D:/Work_Work/Mars/MAVEN/test_particle_jl/neutral/MAMPS/MAMPS_LS*_F*.dat

Despite the historical function name `amps_density`, the data source here is
MAMPS. The returned density is the hot atomic oxygen component, `O_hot`, in
m^-3.

Main public functions
---------------------
`load_amps_grid(solar, ls)`
    Load one packaged MAMPS hot-O case and return a `MAMPSGrid` object.

`amps_density(lon_deg, lat_deg, altitude_km, solar, ls)`
    Return MAMPS hot O density by 3D interpolation.

`amps_density_xyz(x, y, z, solar, ls, position_unit)`
    Convert Mars-centered Sun-fixed Cartesian coordinates to model longitude,
    latitude, and altitude, then return MAMPS hot O density.

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
- Returned `O_hot` density is in m^-3.

Case selection
--------------
- `solar="solar_max"` or `200` loads F10.7 = 200.
- `solar="solar_moderate"` or `130` loads F10.7 = 130.
- `solar="solar_min"` or `70` loads F10.7 = 70.
- `ls` must be one of 0, 90, 180, or 270.

Interpolation
-------------
Inside the MAMPS altitude range, `O_hot` is interpolated in log-density space
on the 3D lon-lat-alt grid. Outside the MAMPS altitude range, this module
returns 0.0, because no MAMPS hot-O values are available there.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .gitm import (
    _bracket,
    _lon_bracket,
    _mat_string,
    _normalize_ls,
    _solar_to_f107,
    cartesian_to_lon_lat_alt,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class MAMPSGrid:
    """One packaged MAMPS hot-O case.

    Arrays are stored as `(longitude, latitude, altitude)`.

    Attributes
    ----------
    lon_deg, lat_deg, alt_km
        Grid coordinates.
    nO_hot
        Hot atomic oxygen density in m^-3.
    source_file
        Original MAMPS `.dat` filename.
    filename
        Packaged `.mat` file used by this object.
    """

    lon_deg: np.ndarray
    lat_deg: np.ndarray
    alt_km: np.ndarray
    nO_hot: np.ndarray
    source_file: str
    filename: Path


@lru_cache(maxsize=12)
def load_amps_grid(solar: str | int = "solar_min", ls: int | str = 0) -> MAMPSGrid:
    """Load one packaged MAMPS hot-O grid.

    Examples
    --------
    >>> grid = load_amps_grid(solar="solar_max", ls=270)
    >>> grid.nO_hot.shape
    (72, 36, 668)
    """
    f107 = _solar_to_f107(solar)
    ls_value = _normalize_ls(ls)
    filename = DATA_DIR / f"mamps_ls{ls_value:03d}_f{f107:03d}.mat"
    if not filename.exists():
        raise FileNotFoundError(filename)
    data = loadmat(filename)
    return MAMPSGrid(
        lon_deg=np.asarray(data["lon_deg"], dtype=np.float64).ravel(),
        lat_deg=np.asarray(data["lat_deg"], dtype=np.float64).ravel(),
        alt_km=np.asarray(data["alt_km"], dtype=np.float64).ravel(),
        nO_hot=np.asarray(data["nO_hot"], dtype=np.float64),
        source_file=_mat_string(data.get("source_file", np.array(["unknown"]))),
        filename=filename,
    )


def _interp3_hot_o(grid: MAMPSGrid, lon_deg, lat_deg, alt_km) -> np.ndarray:
    """Trilinear interpolation of MAMPS hot O in log-density space."""
    values = np.log(np.maximum(grid.nO_hot, 1.0e-300))
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
    return np.exp(c0 * (1.0 - wz) + c1 * wz)


def amps_density(
    lon_deg,
    lat_deg,
    altitude_km,
    solar: str | int = "solar_min",
    ls: int | str = 0,
) -> dict[str, float | np.ndarray]:
    """Return MAMPS hot O density in m^-3 by 3D interpolation.

    Values outside the MAMPS altitude domain are returned as 0.

    Examples
    --------
    >>> rho = amps_density(27.5, 0.0, 1000.0, solar="solar_max", ls=270)
    >>> rho["O_hot"]  # m^-3

    Vector altitude input is also supported:

    >>> rho = amps_density(27.5, 0.0, [500.0, 1000.0, 3000.0], solar=130, ls=90)
    >>> rho["O_hot"].shape
    (3,)
    """
    alt, lon, lat = np.broadcast_arrays(
        np.asarray(altitude_km, dtype=np.float64),
        np.asarray(lon_deg, dtype=np.float64),
        np.asarray(lat_deg, dtype=np.float64),
    )
    grid = load_amps_grid(solar=solar, ls=ls)
    valid = (alt >= grid.alt_km[0]) & (alt <= grid.alt_km[-1])
    density = np.zeros_like(alt, dtype=np.float64)
    if np.any(valid):
        density[valid] = _interp3_hot_o(grid, lon[valid], lat[valid], alt[valid])

    scalar = np.ndim(altitude_km) == 0 and np.ndim(lon_deg) == 0 and np.ndim(lat_deg) == 0
    if scalar:
        return {"O_hot": float(density)}
    return {"O_hot": density}


def amps_density_xyz(
    x,
    y=None,
    z=None,
    *,
    solar: str | int = "solar_min",
    ls: int | str = 0,
    position_unit: str = "km",
    mars_radius_km: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Return MAMPS hot O density from Sun-fixed Cartesian coordinates.

    This is a convenience wrapper around `amps_density`. The Cartesian
    convention is:

    - origin at Mars center
    - `+X` points to the subsolar point, `lon_deg = 0`, `lat_deg = 0`
    - `+Y` points toward `lon_deg = 90`
    - `+Z` points north

    Examples
    --------
    >>> from aspen.constants import MARS_RADIUS_KM
    >>> rho = amps_density_xyz(MARS_RADIUS_KM + 1000.0, 0.0, 0.0)
    >>> rho["O_hot"]  # density at lon=0, lat=0, alt=1000 km
    """
    kwargs = {"position_unit": position_unit}
    if mars_radius_km is not None:
        kwargs["mars_radius_km"] = mars_radius_km
    lon_deg, lat_deg, altitude_km = cartesian_to_lon_lat_alt(x, y, z, **kwargs)
    return amps_density(
        lon_deg,
        lat_deg,
        altitude_km,
        solar=solar,
        ls=ls,
    )
