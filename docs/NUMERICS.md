# Numerical and physical scope

## Shared model

- Constant 200 W: CCD 1 = 90 W, CCD 2 = 90 W, I/O = 20 W.
- Simplified, symmetric AM5-inspired substrate, dies and copper-equivalent IHS; no manufacturer internal CAD. Effective package material properties are modelling assumptions.
- Physical case: 240 × 200 × 320 mm (15.36 L). Uniform air grid: 1 mm. Half-domain symmetry in x; mirrored full-domain displays.
- 32 mm exterior flow-only ducts move pressure planes away from the case openings. The thermal domain remains the physical case. The outer 16 mm of each duct has a viscosity sponge; it is not an enlarged heated case.
- External case-wall loss: 25 °C ambient, h = 5 W/(m² K), zero explicit wall resistance/capacity. Internal half-cell conduction is retained. External air CFD and radiation are not solved.
- Die/IHS areal contact resistance: 2e-6 m² K/W. IHS/cooler TIM: 2e-5 m² K/W. Cooler base/fin bonds are ideal.
- Flow: lbmpy D3Q27 cumulant LBM with voxel NoSlip / bounce-back. The low-Mach problem is not literally meshless: the Cartesian lattice is a mesh. No IBM forcing is used.
- Heat: conservative finite-volume conduction/advection with physical heat capacities. A CPU temperature limiter does not reduce the prescribed heat source.

## Cooling modes

Natural convection has low and high open ports at equal reduced/gauge reference pressure. Boussinesq buoyancy supplies the circulation. The reported natural heat/flow history is coupled in physical time.

Case ventilation imposes 40 CFM at the inlet with a pressure-reference outlet. The CPU fan is an actuator pressure-rise model at 1200 rpm based on a parabolic interpolation of published fan endpoints; it is not a resolved rotating-blade computation or a measured complete fan curve. Translucent fan blades in the videos are visual context.

The air cooler has a 120 × 120 mm copper base and aluminium fins. Six 6 mm heatpipes use **assumed** 10,000 W/(m K) internal equivalent conductivity, copper-level interface conduction (390 W/(m K)) and 1.8 MJ/(m³ K) volumetric heat capacity. Internal boiling, wick transport, capillary limits, dryout and orientation are not resolved. These parameters were not fitted to a target CPU temperature.

The local water model is 64 × 80 × 24 mm at 0.25 mm spacing, with a sealed copper cold plate, water-like constant properties, 30 °C inlet and nominal 1 L/min. It has no heatpipes. The radiator/pump loop is represented by prescribed inlet temperature/flow. Electronics remain dry. Separate coarse case-air heating is one-way: heat removed by that air is **not** added to the closed local 200 W water budget or fed back into the fine water model.

## Reference history versus portable restart

The archived CPU-fan auxiliary mean-temperature iteration stopped before thermal-equilibrium convergence. Only its last valid velocity/eddy-viscosity field was reused; heating reset to 25 °C. That velocity includes buoyancy from the warm precursor. Ventilation reused an earlier flow field, adjusted isothermally for 1.6 s and averaged over the last 0.8 s. Neither forced-air heat history includes evolving buoyancy feedback or flow startup.

The public cold-start helper instead prepares both forced-air cases isothermally from rest for a bounded interval. Full-domain cold-start outcomes have **not** been recalculated in this packaging pass. They may differ from the reference movies. Original solver routines remain available for an explicitly supplied historical field, but such fields are excluded from the repository.

## Stop and stability checks

Stop at CPU peak ≥95 °C, or |dT_CPU/dt| <0.01 °C/s over a continuous 10 s interval using 1 s fitted windows. This is a numerical reporting criterion; it is not a simulated hardware shutdown or a guarantee of global flow equilibrium.

Natural flow/thermal step: 0.2 ms, with conservative local heatpipe diffusion subcycles. Forced-air heat uses implicit steps up to 2 s after smaller startup steps, reduced to at most 1 s near the temperature limit or a plateau. Water heat uses at most 0.5 s. LBM Mach, density, finite fields, thermal stability and energy residuals are monitored. No grid/time-step independence study is claimed.

## Known diagnostics that remain unresolved

In the archived natural case:

- Maximum recorded lattice Mach number: 0.225926 (guard 0.25).
- Maximum sampled thermal energy residual: 0.003373% of supplied heat; final residual about 0.001285%.
- Integrated macroscopic mass-flux closure residual: −0.002845356 kg, equal to **15.93% of initial case-air mass**, or **0.780% of cumulative inlet mass**.
- Final instantaneous port-volume imbalance: 5.16%; last 10 s mean: 1.06%.

This mass diagnostic samples boundary-cell density × velocity every 0.2 s and uses trapezoidal integration. It is not an exact population-link flux audit. Exact mass closure has not been established; the residual is retained rather than hidden or tuned away. Treat all temperature results as demonstration outputs, not validated CPU product predictions.

The GPU label in the reference videos states the manufacturer's FP32 peak specification. It is not a measured solver FLOP/s benchmark.
