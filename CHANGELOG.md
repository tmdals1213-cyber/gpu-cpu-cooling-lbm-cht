# Modification history

## 2026-11-18 — portable public package

- Collected the active Revision C solver and its local dependencies into a shared source tree.
- Preserved upstream GPL LICENSE/NOTICE and documented the lbmpy/pystencils basis.
- Replaced workstation-relative imports, cache locations and output paths with repository-relative paths and explicit output roots.
- Added case-specific entry points, setup export, a GPU smoke test, package validation and a local-agent handoff.
- Added bounded isothermal cold-start forced-air preparation so omitted predecessor fields are not required. This is explicitly distinguished from the archived warm-start reference history.
- Standardized the optional water-history folder name for fresh runs; equations and one-way coupling remain unchanged.
- Included only small geometry, code, setup, reference summaries and edited clips. Excluded full fields, checkpoints, runtime environments and raw archives.
- Reframed the showcase into a 1920 × 1080, 29 s landscape edit; retained the 1080 × 1920 portrait edit.

## Revision C — numerical/model changes retained

- Added six finite equivalent-conductance heatpipes to the three air coolers, with copper-level interface conductance and conservative local diffusion.
- Kept the 200 W package, contact resistances, fin envelope, case boundaries and 1200 rpm CPU-fan scenario. No heatpipes were added to the cold plate.
- Completed natural coupled heating and forced-flow thermal histories to a 95 °C event or the stated temperature-settling criterion.
- Preserved the unresolved macroscopic mass-flux diagnostic in the natural case and the limitations of the mixed coupled/fixed-flow comparison.
