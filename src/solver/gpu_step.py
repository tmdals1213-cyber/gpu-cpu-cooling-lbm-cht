# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU-only lbmpy 2.1.1 adapter (GPL-3.0-or-later).

The upstream LatticeBoltzmannStep compiles CPU initialization/output kernels even
for GPU simulations. This small subclass keeps those kernels on CUDA too, so no
host C++ compiler is required. Flow collision and boundary algorithms are lbmpy.
"""
from pathlib import Path
import os

from runtime_paths import CACHE
for variable, folder in [('CUPY_CACHE_DIR', 'cupy'), ('PYSTENCILS_CACHE_DIR', 'pystencils'),
                         ('TEMP', 'tmp'), ('TMP', 'tmp')]:
    path = CACHE / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)

import pystencils as ps
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments


class GpuStep(LatticeBoltzmannStep):
    def __init__(self, **kwargs):
        kwargs.update(compute_velocity_in_every_step=True, compute_density_in_every_step=True)
        super().__init__(**kwargs)

    def _compile_macroscopic_setter_and_getter(self):
        fields = self.data_handling.fields
        pdf = fields[self._pdf_arr_name]
        rho = fields[self.density_data_name].center
        vel = fields[self.velocity_data_name].center_vector
        config = ps.CreateKernelConfig(target=ps.Target.GPU)
        config.default_dtype = self._config.get_option('default_dtype')
        getter = self.method.conserved_quantity_computation.output_equations_from_pdfs(
            pdf.center_vector, {'density': rho, 'velocity': vel})
        setter = pdf_initialization_assignments(self.method, rho, vel, pdf.center_vector)
        return ps.create_kernel(getter, config=config).compile(), ps.create_kernel(setter, config=config).compile()

    def set_pdf_fields_from_macroscopic_values(self):
        self.data_handling.to_gpu(self.density_data_name)
        self.data_handling.to_gpu(self.velocity_data_name)
        self.data_handling.run_kernel(self._setterKernel, **self.kernel_params)

    def pre_run(self):
        pass  # GPU fields remain resident between run() calls.

    def post_run(self):
        self.data_handling.run_kernel(self._getterKernel, **self.kernel_params)
        self.data_handling.to_cpu(self.velocity_data_name)
        self.data_handling.to_cpu(self.density_data_name)
