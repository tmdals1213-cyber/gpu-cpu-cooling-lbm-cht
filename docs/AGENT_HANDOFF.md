# Handoff to another local agent

## Pasteable starting task

> Open this repository, read AGENTS.md and docs/RUNNING.md, and inspect docs/NUMERICS.md before making claims about the results. Verify the portable package and installed GPU environment. Run a small smoke test and a setup-only check. Show the planned commands and estimated resource needs before starting the full case requested by the user. Keep raw output under runs/ and preserve the archived reference summaries. Do not assume the omitted predecessor flow fields are available.

## What is already complete

Four Revision C thermal histories, endpoint tables, temperature plots, individual landscape clips and edited landscape/portrait comparisons. The videos use calculated fields and CFD-advected markers; their thermal and tracer clocks can differ and are labelled. Reference data and media are immutable evidence of that calculation, not expected byte-for-byte test fixtures for a cold restart.

## Source map

| Module | Responsibility |
|---|---|
| `case_setup.py`, `model.py` | Air-case geometry, boundaries, materials, fan/source reference |
| `transient.py`, `gpu_step.py` | GPU LBM and conservative explicit heat transport |
| `pressure_boundary.py` | Pressure-boundary helpers |
| `heatpipes.py` | Geometry and conservative equivalent-conductance heatpipes |
| `thermal_base.py`, `thermal.py` | GPU thermal operator and implicit stepping |
| `run_transient_buffered.py` | Coupled natural heat/flow history and checkpoints |
| `prepare_flow.py` | New portable cold-start forced-air preparation |
| `fixed_flow_heat.py` | Physical thermal startup in a fixed CFD field |
| `cold_plate.py` | Fine sealed copper/water geometry and laminar-flow preparation |
| `water_case_air.py`, `water_air_heat_history.py` | Optional one-way case-air companion |
| `thermal_convergence.py`, `checkpoint_io.py` | Thermal stopping criterion and atomic output |

`solve_air.py` and `prepare_forced_flow.py` preserve the historical expert workflows. They are intentionally outside the default portable route.

## Missing by design

No original NPZ fields, PDF populations, checkpoints, CUDA caches, environment folders or full snapshot movies. A new solve regenerates those locally. There are no hidden references to the original user's filesystem. The original four-case run was not repeated when preparing this repository.

## Presentation conventions

Black background; short English labels; separate solid/air/water temperature scales; CPU source powers 90/90/20 W; visible case boundaries and inlet/outlet; translucent fan context. Water stays inside the cold plate. No heatpipes in the water case. End with a 2 × 2 comparison. Always distinguish fixed 200 W from real CPU throttling.

## Editing geometry or physics

Read `case.json` as a model summary plus driver controls, not a general-purpose CFD input format. Geometry/properties are Python definitions. Keep source partition, interface resistance, water containment, x symmetry and physical units consistent. The `.stl` files are generated surrogate parts; no proprietary CPU CAD is embedded. The legacy geometry JSON's descriptive material table is not the active property database; `model.py` / `case_setup.py` / `cold_plate.py` are authoritative.
