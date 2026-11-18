# Local agent entry point

Read `README.md`, `docs/RUNNING.md`, `docs/NUMERICS.md` and `docs/AGENT_HANDOFF.md` before modifying a case. User instructions take precedence over this workflow guidance.

- The four case folders share `src/solver/`; avoid creating divergent copies of the solver.
- Run `python tools/verify_package.py`, then `doctor.py`, one `--stage setup` and the small GPU smoke test before a long calculation.
- Default to `--stage plan`; start hours-long GPU runs when that is the user's requested task.
- Save generated data in `runs/` or a user-selected output directory; never commit fields, checkpoints, caches, environments or credentials.
- Preserve existing run histories. Diagnose a failed case, retain its diagnostics and continue independent queued cases when the user requested a batch.
- Do not change physics or tune properties to obtain an attractive temperature. Record changes and keep reference results separate from newly computed results.
- A fresh forced-air preparation is not the exact archived warm-start flow. Maintain this distinction in reports and videos.
- Retain upstream LICENSE/NOTICE, source attribution and modification history. Do not claim authorship of lbmpy/pystencils or remove their notices.
- Public uploads should contain only the reviewed allowlisted source/setup/summary artifacts. Use `tools/verify_package.py` before staging files.
