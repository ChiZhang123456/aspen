# aspen

ASPEN-style Monte Carlo transport tools for Mars proton aurora test particles.

The package traces H-ENA and H+ particles through a neutral Mars atmosphere and
samples collisions with CO2, N2, and O. It tracks position, altitude, energy,
charge state, reaction type, scattering angle, energy loss, and cumulative
collision probability.

## Quick start

Run from the repository root:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B -m aspen
```

The same example can also be run from the script:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\example_run_single_particle_aspen.py
```

The example starts one H-ENA particle at 600 km:

```text
position = [(Mars radius + 600 km), 0, 0] m
velocity = [-400000, 0, 0] m/s
max step = 5000 m
```

Outputs are written to:

```text
aspen_examples/single_particle_main
```

## Main single-particle API

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
    max_step_m=5000.0,
    rng=rng,
)

rows = flatten_trace_history(result, particle_id=0)
write_history_csv(rows, "history.csv")
plot_particle_trace_history(rows, "trace.png", particle_id=0)
```

## Stop conditions

Tracing stops only when:

```text
energy < 10 eV
altitude < 100 km
altitude > 1000 km
```

Charge-state changes do not stop tracing. After each collision, a new random
number `R` is drawn and the cumulative optical depth is reset.

## Monte Carlo example

Run two particles in parallel:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\run_aspen_monte_carlo_h_ena_600km.py --n-particles 2 --workers 2 --max-step-m 5000
```

Run 5000 particles:

```powershell
& C:\Users\Win\.conda\envs\mars\python.exe -B scripts\run_aspen_monte_carlo_h_ena_600km.py --n-particles 5000 --workers 8 --max-step-m 5000
```

## Data included

The package includes compact neutral atmosphere `.mat` files and collision
cross-section tables needed by the examples.

## More notes

See:

```text
docs/aspen_quick_start_zh.md
```
