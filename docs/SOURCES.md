# Open-source sources and references

## Software

- [lbmpy 2.1.1](https://pypi.org/project/lbmpy/2.1.1/) — GPL-3.0-or-later. GPU collision/streaming and boundaries. Upstream contributor credits remain with the project.
- [pystencils 2.1](https://pypi.org/project/pystencils/2.1/) — GPL-3.0-or-later. Stencil code generation and data handling.
- [pycodegen project](https://pycodegen.walberla.net/) — upstream project/tutorials.
- [CuPy](https://cupy.dev/) and [installation documentation](https://docs.cupy.dev/en/stable/install.html) — GPU arrays, CUDA runtime and sparse solvers.
- [Trimesh](https://trimesh.org/) — STL loading and voxel material tests.
- [PyVista](https://docs.pyvista.org/), VTK, Matplotlib and Pillow — postprocessing and media layout.

Dependencies are installed from their upstream distributions, not bundled here. The `GpuStep` adapter adapts the initialization/output pattern of `lbmpy.lbstep.LatticeBoltzmannStep`. LICENSE and NOTICE preserve the existing GPL terms; source attribution does not replace the license.

## Physical reference inputs

- [AMD Ryzen 9 9950X3D2 Dual Edition](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d2-dual-edition.html) — the original 200 W design-reference choice. Internal geometry and die powers here are assumptions, not vendor CAD or measured package-power distribution.
- [Noctua NF-A12x25 G2 PWM information sheet](https://cdn.noctua.at/media/noctua_nf_a12x25_g2_pwm_infosheet_en_web.pdf) — endpoints used for the approximate fan law. The model uses 1200 rpm; a full measured fan curve is not supplied.
- [ACT heat-pipe design guide](https://www.1-act.com/thermal-solutions/passive/heat-pipes/heat-pipe-design-guide/) — background only. It does not calibrate the chosen equivalent conductivity.

The 200 W fixed source, 90/90/20 W split, contact resistances, heatpipe equivalent properties and simplified geometry are declared model inputs. These references do not constitute validation of the calculated CPU temperatures.

## Distribution

[GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) was used to keep individual committed artifacts below its regular Git file limit. Raw calculation arrays are intentionally excluded.
