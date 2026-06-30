# aspen_zc

`aspen_zc` is a lightweight Python package for ASPEN-style Mars H/H+ collision calculations.

The package is organized around atmosphere and cross-section based collision tools:

```text
aspen_zc.neutral_density_model
aspen_zc.cross_sections
```

It is split into review-friendly modules:

```text
aspen_zc/neutral_density_model/gitm.py
aspen_zc/neutral_density_model/amps.py
aspen_zc/neutral_density_model/__init__.py
aspen_zc/neutral_density_model/data/*.mat
aspen_zc/cross_sections/cross_section.py
aspen_zc/cross_sections/collision_frequency.py
aspen_zc/cross_sections/collision_sampler.py
aspen_zc/cross_sections/elastic_collision.py
aspen_zc/cross_sections/scattering_angle.py
aspen_zc/cross_sections/*.txt
```

The subpackage provides three main functions:

```python
from aspen_zc import neutral_density, cross_section, collision_frequency
```

## 1. Local neutral density

```python
rho = neutral_density(
    lon_deg=26.4,
    lat_deg=0.0,
    altitude_km=200,
    solar="solar_max",
    ls=270,
)
```

Returns a dictionary in `m^-3`:

```python
{"CO2": ..., "O": ..., "O_cold": ..., "O_hot": ..., "N2": ..., "O2": ..., "CO": ..., "Tn": ...}
```

The neutral density model uses MGITM cold neutral densities and MAMPS hot O.
The total atomic oxygen density is:

```text
O = O_cold(MGITM) + O_hot(MAMPS)
```

Supported solar activity choices are `solar_max`, `solar_moderate`, and `solar_min`.
These map to `F200`, `F130`, and `F070`, respectively. Supported `ls` values
are `0`, `90`, `180`, and `270`.

MGITM provides `CO2`, `O`, `O2`, `N2`, `CO`, and `Tn`. Above the MGITM top
altitude, each density is hydrostatically extrapolated to 10000 km using the
local top-layer neutral temperature. MAMPS provides hot O on a 3D grid from
100 to 6770 km.

## 2. Cross section for one reaction

```python
cs = cross_section(
    projectile="H",
    target="CO2",
    reaction="ionization",
    energy_ev=1000,
)
```

Returns:

```python
{
    "sigma_m2": ...,
    "energy_loss_ev": ...,
}
```

By default, cross sections are read from the package data folder:

```text
aspen_zc/cross_sections
```

The `.txt` files were copied from the extracted reference tables under
`JGR_Figures/JGR_Figures`.

The `energy_loss_ev` values are parsed from `Q = ... eV` in the `.txt` headers.

Supported projectiles:

```text
H
H+
```

Supported targets:

```text
CO2
O
N2
```

Supported cross-section reactions:

```text
state_change
ionization
lya
elastic
halpha
```

For `H`, `state_change` means electron stripping, `H -> H+`.

For `H+`, `state_change` means charge exchange, `H+ -> H`.

The default Monte Carlo reaction sampler uses the four core ASPEN channels:

```text
elastic
ionization
state_change
lya
```

It does not include `halpha` unless explicitly requested.

For fast neutral H impact on O, the default channels correspond to:

```text
H + O -> H + O                 elastic scattering
H + O -> H + O+ + e-           impact ionization of target
H + O -> H+ + O + e-           electron stripping, H becomes proton
H + O -> H + O + Ly-alpha      excitation of fast H followed by Ly-alpha emission
```

For H+ impact on O, the default channels correspond to:

```text
H+ + O -> H+ + O               elastic collision
H+ + O -> H + O+               charge exchange, H+ becomes H-ENA
H+ + O -> H+ + O+ + e-         impact ionization of target
H+ + O -> H + O + Ly-alpha     H Ly-alpha related excitation channel
```

## 3. Collision frequency

```python
out = collision_frequency(
    altitude_km=200,
    lon_deg=26.4,
    lat_deg=0.0,
    projectile="H",
    energy_ev=1000,
    solar="solar_min",
    ls=0,
)
```

Returns:

```text
collision_coefficient_m-1 = sum n sigma
collision_frequency_s-1  = v sum n sigma
mean_free_path_km        = 1 / (sum n sigma)
breakdown               = per target and reaction contribution
```

Here `v` is computed from particle energy using the hydrogen mass. Neutral
densities are evaluated with the packaged MGITM + MAMPS model in
`aspen_zc/neutral_density_model`; `O = O_cold + O_hot` when `include_hot_o=True`.

## 4. Scattering angle

```python
from aspen_zc import scattering_angle

theta_deg = scattering_angle(0.99)
```

This reads:

```text
aspen_zc/cross_sections/scattering_angle_distribution.txt
```

The table is treated as an inverse-CDF style mapping from random number to
LAB-frame scattering angle.

## 5. Elastic collision energy loss

```python
from aspen_zc import elastic_collision_after_scattering

out = elastic_collision_after_scattering(
    projectile="H",
    target="CO2",
    projectile_velocity_m_s=[400e3, 0, 0],
    scattering_angle=10.0,
    charge_state="neutral",
    target_density_m3=1e12,
)
```

The target atmosphere particle is assumed initially at rest. The calculation
uses two-body elastic collision kinematics with conservation of momentum and
kinetic energy. The returned dictionary includes:

```text
projectile_velocity_after_m_s
target_velocity_after_m_s
projectile_energy_after_eV
projectile_energy_loss_eV
target_recoil_energy_eV
charge_state_after
scattering_angle_deg
```

The scattering angle only fixes the deflection from the initial projectile
velocity. The optional `azimuth_angle` selects the scattering plane.

## 6. Chemical reaction outcomes

```python
from aspen_zc import electron_stripping, ionization, lyman_alpha_emission

out = electron_stripping(
    projectile="H",
    target="O",
    projectile_velocity_m_s=[400e3, 0, 0],
    target_density_m3=1e12,
    scattering_angle=10.0,
    charge_state="neutral",
)
```

The three non-elastic reaction helpers are:

```text
electron_stripping     H + X -> H+ + X + e-
ionization             H/H+ + X -> H/H+ + X+ + e-
lyman_alpha_emission   H/H+ + X -> H + X + Ly-alpha
```

All reaction helpers take `target`, projectile velocity, target density,
scattering angle, and `charge_state`. They return the post-reaction charge
state, post-collision projectile velocity, cross section, local coefficient
`n sigma`, scattering angle in degrees, and the tabulated energy loss.

## 7. Sample collision target and reaction

```python
from aspen_zc import sample_collision_event

event = sample_collision_event(
    density_m3={"CO2": 2.9e13, "O": 3.3e13},
    projectile="H",
    energy_ev=1000,
    random_target=0.5,
    random_reaction=0.2,
    targets=("CO2", "O"),
)
```

This samples the collision target first:

```text
P_j = n_j sum_k sigma_j,k / sum_j n_j sum_k sigma_j,k
```

Then it samples the reaction channel for that target:

```text
P_k = sigma_k / sum_k sigma_k
```

The returned dictionary includes the selected `target`, selected `reaction`,
the two random numbers, the target probabilities, and the reaction
probabilities.

The sampler does not take a `reactions` input. It always uses the four core
channels:

```text
elastic
ionization
state_change
lya
```

## 8. Apply one sampled collision to a particle

```python
from aspen_zc import Particle, apply_random_collision_to_particle

p = Particle(
    projectile="H",
    velocity=[400e3, 0, 0],
    position=[0, 0, 1000],
)

out = apply_random_collision_to_particle(
    particle=p,
    density_m3={"CO2": 2.9e13, "O": 3.3e13},
    targets=("CO2", "O"),
)
```

This function first samples the collision target and reaction, then applies
the corresponding reaction function. It updates the particle in place:

```text
particle.projectile
particle.charge_state
particle.velocity
particle.position, if collision_position is supplied
particle.local_collision_frequency
particle.cumulative_collision_frequency, if elapsed_time_s or path_length_m is supplied
particle.iteration
```

For `state_change`, neutral H uses electron stripping and H+ uses charge
exchange.

## 9. Numba adaptive transport loop

```python
from aspen_zc import Particle, advance_particle_until_collision_from_position_numba
from aspen_zc.constants import MARS_RADIUS_KM

p = Particle(
    projectile="H",
    velocity=[0, 0, -400e3],
    position=[MARS_RADIUS_KM + 1000, 0, 0],
    R=0.02,
)

out = advance_particle_until_collision_from_position_numba(
    p,
    targets=("CO2", "O"),
    solar="solar_min",
    ls=0,
    safety_factor=0.5,
    max_step_m=5000,
)
```

This loop uses `particle.position` to compute local neutral density, then uses
numba for the adaptive transport step:

```text
E from particle velocity and charge_state
alpha = sum_j n_j sum_k sigma_j,k
local_collision_frequency = speed * alpha
cumulative_collision_frequency += alpha * ds
collision occurs when cumulative_collision_frequency > particle.R
```

When a collision occurs, it samples the target and reaction using the current
charge state and energy, applies the corresponding reaction function, updates
the particle velocity, charge state, projectile label, position, and iteration
count. By default it resets the cumulative collision frequency and draws a new
`R` after the collision.
