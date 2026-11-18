# Running the cases

## Environment

Reference: Windows 11, Python 3.13.2, NVIDIA RTX 5080 (16 GB). Python 3.12+ is required by lbmpy 2.1.1; Python 3.13 is the pinned reference target. Paths are portable, but Linux execution has not been validated in this packaging pass.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/doctor.py
```

Linux/macOS shell syntax for environment creation (the solver still requires a supported NVIDIA CUDA environment):

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

`cupy-cuda12x[ctk]` supplies CUDA component wheels; a compatible NVIDIA driver is still needed. Do not install multiple CuPy variants into one environment. Exact reference component versions are listed in `environment/cuda12-reference.txt`; install those only when matching that toolchain. See [CuPy installation](https://docs.cupy.dev/en/stable/install.html).

Optional PNG postprocessing: `python -m pip install -r requirements-visualization.txt`. The bundled MP4s play without these packages. FFmpeg is not needed for solving or viewing the supplied media.

## First checks

```sh
python tools/verify_package.py
python tools/doctor.py
python tools/run_case.py --case natural --stage setup
python tools/run_case.py --case natural --stage smoke
```

The smoke test uses a small closed box and checks GPU startup and conservation. It does not establish full-case convergence or accuracy. `setup` voxelizes the supplied STLs, checks material presence and symmetry, and writes `runs/<case>/setup.json`; it does not run flow.

## Explicit runs

```sh
python tools/run_case.py --case natural --stage run
python tools/run_case.py --case ventilation --stage run
python tools/run_case.py --case cpu_fan --stage run
python tools/run_case.py --case cold_plate --with-case-air --stage run
```

Inspect the corresponding `--stage plan` first. Run one case at a time on a single GPU. In a batch, retain a failed case's diagnostics and proceed to independent cases; do not silently accept a failed precursor or erase its files.

- **Natural:** coupled LBM + heat, 0.2 ms flow/thermal step, initially 25 °C and at rest.
- **Ventilation / CPU fan:** new bounded isothermal flow preparation, then heat in the fixed velocity/eddy-viscosity fields. Defaults: 4 s preparation, final 1 s average, inlet ramp over 0.5 s. This is a runnable alternative to the omitted warm-start states, not an exact replay. A preparation that fails its Mach/finite-field/volume checks must be reviewed before continuing.
- **Cold plate:** converge constant-property laminar water flow and solve its local steady thermal problem once to supply the optional case-air model; then compute physical thermal startup from 30 °C. `--with-case-air` additionally runs the original one-way air workflow. Without it, only the local water/package model is computed. The supplied water video includes that separate case-air calculation.

Intermediate stages can be invoked separately: `--stage flow`, `--stage heat`, and, with `--with-case-air`, `--stage case-air-flow` / `case-air-heat`. Natural convection has no separate fixed-flow stage.

Generated results default to `runs/<case>/`. Use `--output-root /path/to/new/run` for a new run. Existing thermal histories are protected from overwrite. The scripts use `COOLING_OUTPUT_ROOT` and `COOLING_CACHE_ROOT` internally; caches and compiler temporary files stay under `.cache/` by default.

`case.json` separates a **model summary** from editable **run controls**. Flow timestep, preparation duration/window, maximum thermal time, thermal step cap and snapshot interval are passed by the runner. Geometry, properties, power splits, fan law and boundaries are defined in the solver modules and must be changed there consistently; editing the model summary alone does not reconfigure physics. A non-200 W model summary is rejected by the wrapper.

## Output and visualization

Each thermal run records metadata, a scalar history, saved fields and a final field. These generated files are ignored by Git.

```sh
python tools/render_result.py runs/natural/heat/last_fields.npz --output runs/natural/temperature.png
python tools/plot_history.py runs/natural/heat/history.json --output runs/natural/history.png
```

The lightweight postprocessor regenerates an inspection figure from a newly calculated field. The edited showcase movies are reference artifacts; rebuilding their exact cinematography requires the omitted snapshot sequence and original editing session, not just the final endpoint.

## Adjusting a run

Do not fit parameters to force a desired CPU temperature. Record changed geometry, sources, material properties, contacts, inlet/pressure boundaries, flow initialization and timestep. Preserve the distinction between physical heat time and flow-preparation iterations. Stop at the 95 °C comparison event or the documented settling criterion; no hardware throttle controller is implemented.

`src/solver/solve_air.py` and `prepare_forced_flow.py` retain the historical expert workflows. They are not the default public runner: the former can perform lengthy auxiliary equilibrium iterations, and the latter requires an explicitly supplied predecessor field.
