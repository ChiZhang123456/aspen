# aspen quick start

This note shows the shortest workflow for running one ASPEN test particle.

## 1. Run from the project root

Open PowerShell in:

```powershell
D:\Work_Work\Mars\MAVEN\iuvs_data_kp
```

Use the Mars Python environment:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B -m aspen
```

The same example can also be run from the script:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\example_run_single_particle_aspen.py
```

The example creates one particle:

```text
species: H-ENA
initial altitude: 600 km
initial position: [(Mars radius + 600 km), 0, 0] m
initial velocity: [-400000, 0, 0] m/s
max step: 1000 m
```

The tracing stops only when:

```text
energy < 10 eV
altitude < 100 km
altitude > 1000 km
```

Charge-state changes do not stop tracing. If H-ENA becomes H+, or H+ becomes H-ENA, tracing continues with the new charge state.

## 2. Main code pattern

```python
import numpy as np

from aspen import (
    Particle,
    flatten_trace_history,
    plot_particle_trace_history,
    trace_particle_xyz_until_stop,
    write_history_csv,
)
from aspen.constants import MARS_RADIUS_KM

rng = np.random.default_rng(7)

position_m = np.array([(MARS_RADIUS_KM + 600.0) * 1000.0, 0.0, 0.0])
velocity_m_s = np.array([-400_000.0, 0.0, 0.0])

particle = Particle(
    "H-ENA",
    velocity=velocity_m_s,
    position=position_m,
    R=float(rng.random()),
    rng=rng,
)

result = trace_particle_xyz_until_stop(
    particle,
    max_step_m=1000.0,
    rng=rng,
)

rows = flatten_trace_history(result, particle_id=0)
write_history_csv(rows, "history.csv")
plot_particle_trace_history(rows, "trace.png", particle_id=0)
```

## 3. Important functions

Use these most often:

```python
Particle
trace_particle_xyz_until_stop
flatten_trace_history
write_history_csv
plot_particle_trace_history
```

For Monte Carlo summary runs:

```python
MonteCarloConfig
run_one_monte_carlo_particle
summarize_monte_carlo_rows
```

## 4. Output columns

The detailed history CSV includes:

```text
event_number
event_type
projectile
charge_state
x_m, y_m, z_m
altitude_km
energy_before_ev
energy_after_ev
energy_loss_ev
reaction
target
scattering_angle_deg
cumulative_tau
cumulative_collision_probability
random_before_collision
random_after_collision
```

`cumulative_tau` is the integrated collision optical depth.

The cumulative collision probability is:

```text
1 - exp(-tau)
```

It is always between 0 and 1.

## 5. Run two particles in parallel

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\run_aspen_monte_carlo_h_ena_600km.py --n-particles 2 --workers 2 --max-step-m 1000 --output-dir aspen_examples\monte_carlo_h_ena_600km_2p_step1km
```

For a larger run, increase `--n-particles` and `--workers`.

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\run_aspen_monte_carlo_h_ena_600km.py --n-particles 5000 --workers 8 --max-step-m 1000 --output-dir aspen_examples\monte_carlo_h_ena_600km_5000p
```
