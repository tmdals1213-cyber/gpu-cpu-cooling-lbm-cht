# Cpu And Case Fans

[Video summary](summary.mp4) · [Setup](case.json) · [Common runbook](../docs/RUNNING.md)

![Endpoint](endpoint.png)

- Reference CPU temperature: **95.0 °C** at **339.0 s**.
- Endpoint: `temperature_limit_reached`. Constant 200 W; no hardware throttling controller.
- Method: GPU LBM flow + finite-volume CHT; voxel bounce-back, not IBM.

From this directory: `python run.py --stage plan`, then `python run.py --stage setup`.
The source is shared in `../src/solver`; generated calculation files go to `../runs/cpu_fan`.
See the common runbook before starting `--stage run` or changing parameters.
