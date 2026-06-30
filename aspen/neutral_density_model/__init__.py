"""Neutral atmosphere model used by ASPEN.

This subpackage combines:

- MGITM cold neutral atmosphere from `gitm.py`
- MAMPS hot atomic oxygen from `amps.py`

All packaged data are stored in `aspen/neutral_density_model/data` as `.mat`
files, converted from the large source `.dat` files under:

- `D:/Work_Work/Mars/MAVEN/test_particle_jl/neutral/MGITM`
- `D:/Work_Work/Mars/MAVEN/test_particle_jl/neutral/MAMPS`

Recommended public calls
------------------------
For separate components:

```python
from aspen.neutral_density_model.gitm import gitm_density
from aspen.neutral_density_model.amps import amps_density

cold = gitm_density(27.5, 0.0, 200.0, solar="solar_max", ls=270)
hot = amps_density(27.5, 0.0, 1000.0, solar="solar_max", ls=270)
```

For the combined neutral model:

```python
from aspen.neutral_density_model import neutral_density

rho = neutral_density(27.5, 0.0, 1000.0, solar="solar_max", ls=270)
```

For Sun-fixed Cartesian particle positions:

```python
from aspen.constants import MARS_RADIUS_KM
from aspen.neutral_density_model import neutral_density_xyz

rho = neutral_density_xyz(MARS_RADIUS_KM + 1000.0, 0.0, 0.0)
```

The combined dictionary contains `CO2`, `O`, `O2`, `N2`, `CO`, `Tn`,
`O_cold`, and `O_hot` when `O` is requested. Here:

```text
O = O_cold(MGITM) + O_hot(MAMPS)
```

Longitude is model longitude. The model coordinate origin is Sun-fixed:
`lon = 0`, `lat = 0` is the subsolar point. Cartesian wrappers use Mars center
as the origin, `+X` toward the subsolar point, `+Y` toward `lon = 90`, and `+Z`
north.
All densities are returned in m^-3.
"""

from .amps import MAMPSGrid, amps_density, amps_density_xyz, load_amps_grid
from .gitm import (
    GITMGrid,
    GITM_SPECIES,
    GITM_TOP_KM,
    cartesian_to_lon_lat_alt,
    gitm_density,
    gitm_density_xyz,
    load_gitm_grid,
)


NEUTRAL_SPECIES = ("CO2", "O", "O2", "N2", "CO")


def neutral_density(
    lon_deg,
    lat_deg,
    altitude_km,
    solar="solar_min",
    ls=0,
    species=NEUTRAL_SPECIES,
    include_hot_o=True,
):
    """Return combined MGITM + MAMPS neutral densities in m^-3.

    `O` is total atomic oxygen, `O_cold` from MGITM plus `O_hot` from MAMPS.

    Parameters
    ----------
    lon_deg, lat_deg
        Longitude east and latitude in degrees.
    altitude_km
        Altitude above the surface in km. Scalars and NumPy-broadcastable
        arrays are accepted.
    solar
        Solar activity selector. Use `solar_max`, `solar_moderate`,
        `solar_min`, or equivalently 200, 130, 70.
    ls
        Solar longitude. Must be 0, 90, 180, or 270.
    species
        MGITM cold neutral species to return. Supported values are `CO2`, `O`,
        `O2`, `N2`, and `CO`.
    include_hot_o
        If true and `O` is requested, add MAMPS hot O to MGITM cold O.

    Returns
    -------
    dict
        Densities in m^-3 plus `Tn` in K.
    """
    gitm = gitm_density(lon_deg, lat_deg, altitude_km, solar=solar, ls=ls, species=species)
    out = dict(gitm)
    if "O" in out:
        out["O_cold"] = out["O"]
        hot = amps_density(lon_deg, lat_deg, altitude_km, solar=solar, ls=ls)["O_hot"] if include_hot_o else 0.0
        out["O_hot"] = hot
        out["O"] = out["O_cold"] + out["O_hot"]
    return out


def neutral_density_xyz(
    x,
    y=None,
    z=None,
    *,
    solar="solar_min",
    ls=0,
    species=NEUTRAL_SPECIES,
    include_hot_o=True,
    position_unit="km",
    mars_radius_km=None,
):
    """Return combined MGITM + MAMPS density from Sun-fixed Cartesian position.

    The Cartesian convention is Mars-centered with `+X` at the subsolar point,
    `+Y` toward model longitude 90 degrees, and `+Z` north. The default input
    coordinate unit is km. Use `position_unit="m"` for meters.
    """
    kwargs = {"position_unit": position_unit}
    if mars_radius_km is not None:
        kwargs["mars_radius_km"] = mars_radius_km
    lon_deg, lat_deg, altitude_km = cartesian_to_lon_lat_alt(x, y, z, **kwargs)
    return neutral_density(
        lon_deg,
        lat_deg,
        altitude_km,
        solar=solar,
        ls=ls,
        species=species,
        include_hot_o=include_hot_o,
    )

__all__ = [
    "cartesian_to_lon_lat_alt",
    "GITMGrid",
    "GITM_SPECIES",
    "GITM_TOP_KM",
    "MAMPSGrid",
    "NEUTRAL_SPECIES",
    "amps_density",
    "amps_density_xyz",
    "gitm_density",
    "gitm_density_xyz",
    "load_amps_grid",
    "load_gitm_grid",
    "neutral_density",
    "neutral_density_xyz",
]
