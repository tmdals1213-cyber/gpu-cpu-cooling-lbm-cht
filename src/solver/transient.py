# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path


import os, sys, json, time, argparse


from runtime_paths import OUTPUT as ROOT


from runtime_paths import ASSETS as BASE


from gpu_step import GpuStep


import numpy as np


import cupy as cp


import sympy as sp


import pystencils as ps


from lbmpy import LBMConfig,LBMOptimisation


from lbmpy.enums import Method, Stencil, SubgridScaleModel,ForceModel


from lbmpy.boundaries import NoSlip, FixedDensity, FreeSlip, UBB, ExtrapolationOutflow, SimpleExtrapolationOutflow


from lbmpy.geometry import add_box_boundary


from pystencils.slicing import slice_from_direction


from pressure_boundary import NonEquilibriumPressure,EquilibriumPressure


from entropic_stabilization import equilibrium_safe_simplification


class TransientCHT:
    def __init__(self, ids, k, capacity, source_w, dx, dt, nu, fluid_rhocp,
                 buoyancy, contacts=None, wall_temperatures=None, advection=True, open_faces=None,
                 flow_dtype='float64', wall_h=None, symmetry_faces=None,
                 ports=None, fluid_density=1.184, sgs_constant=None, turbulent_prandtl=.9,
                 external_drive=False, cumulant_correction=False, entropic=False, viscosity_field=None):
        self.ids, self.dx, self.dt = ids, dx, dt
        self.shape, self.dim = ids.shape, ids.ndim
        self.inner = (slice(1, -1),) * self.dim
        self.capacity = np.asarray(capacity) * dx**self.dim
        self.source_w = source_w
        self.time = 0.0
        self.open_faces = tuple(open_faces or ())
        self.flow_dtype = flow_dtype
        self.fluid_rhocp=fluid_rhocp;self.fluid_density=fluid_density
        self.ports=list(ports or [])
        self.sgs_constant=sgs_constant;self.turbulent_prandtl=turbulent_prandtl
        self.dh = ps.create_data_handling(self.shape, default_target=ps.Target.GPU,
                                          default_ghost_layers=1)
        self.fields = {}
        names = ['T', 'Tnext', 'fluid', 'invC', 'Q', 'wallG', 'wallGT']
        names += [f'G{a}' for a in range(self.dim)]
        for name in names:
            self.fields[name] = self.dh.add_array(name, dtype=np.float64)
            self.put(name, np.zeros(self.shape))
        self.put('fluid', (ids == 0).astype(float))
        self.put('invC', 1 / self.capacity)
        self.put('Q', source_w)
        self.port_masks=[]
        for i,port in enumerate(self.ports):
            coords=np.meshgrid(*[np.arange(n)+.5 for n in self.shape],indexing='ij',sparse=True)
            mask=np.ones(self.shape,dtype=bool)
            for a,(lo,hi) in enumerate(port['ranges']):mask&=(coords[a]>=lo)&(coords[a]<hi)
            if 'circle' in port:
                circle=port['circle'];axes=[a for a in range(self.dim) if a!=port['axis']]
                mask&=sum((coords[a]-c)**2 for a,c in zip(axes,circle['center']))<circle['radius']**2
            face=[slice(None)]*self.dim;face[port['axis']]=0 if port['side']==0 else -1
            face_mask=np.zeros(self.shape,dtype=bool);face_mask[tuple(face)]=True;mask&=face_mask
            self.port_masks.append(mask)
            self.fields[f'port{i}']=self.dh.add_array(f'port{i}',dtype=np.float64)
            self.put(f'port{i}',mask.astype(float))
        if any('pressure_Pa' in port for port in self.ports):
            self.fields['boundaryDensity']=self.dh.add_array('boundaryDensity',dtype=flow_dtype)
            self.dh.cpu_arrays['boundaryDensity'].fill(1.)
            self.dh.to_gpu('boundaryDensity')
        if any(port.get('ramp_velocity') for port in self.ports):
            self.fields['boundaryVelocityScale']=self.dh.add_array('boundaryVelocityScale',dtype=flow_dtype)
            self.dh.cpu_arrays['boundaryVelocityScale'].fill(0.)
            self.dh.to_gpu('boundaryVelocityScale')
            profile=np.ones(tuple(n+2 for n in self.shape),dtype=flow_dtype)
            for port,mask in zip(self.ports,self.port_masks):
                if port.get('sine_rect'):
                    coords=np.meshgrid(*[np.arange(-1,n+1)+.5 for n in self.shape],indexing='ij',sparse=True)
                    smooth=np.ones(profile.shape,dtype=flow_dtype)
                    for a in range(self.dim):
                        if a==port['axis']:continue
                        lo,hi=port['ranges'][a]
                        smooth*=np.sin(np.pi*np.clip((coords[a]-lo)/(hi-lo),0,1))
                    smooth/=smooth[self.inner][mask].mean();profile*=smooth
                if port.get('parabolic'):
                    circle=port['circle'];axes=[a for a in range(self.dim) if a!=port['axis']]
                    coords=np.meshgrid(*[np.arange(-1,n+1)+.5 for n in self.shape],indexing='ij',sparse=True)
                    radius2=sum((coords[a]-c)**2 for a,c in zip(axes,circle['center']))
                    radial=np.broadcast_to(np.maximum(0.,1-radius2/circle['radius']**2),profile.shape).copy()
                    radial/=radial[self.inner][mask].mean()
                    profile*=radial
            self.port_velocity_profile=cp.asarray(profile)
        self.wallG, self.wallGT = np.zeros(self.shape), np.zeros(self.shape)
        # Geometry is cell centred. Temperature boundary is at the outer face.
        for (axis, side), value in (wall_temperatures or {}).items():
            face = [slice(None)] * self.dim
            face[axis] = 0 if side == 0 else -1
            face = tuple(face)
            if wall_h and (axis,side) in wall_h:
                g = dx**(self.dim-1)/(dx/(2*k[face+(axis,)])+1/wall_h[(axis,side)])
            else:
                g = 2 * k[face + (axis,)] * dx**(self.dim - 2)
            self.wallG[face] += g
            self.wallGT[face] += g * value
        for mask in self.port_masks:
            self.wallG[mask]=0.;self.wallGT[mask]=0.
        self.put('wallG', self.wallG)
        self.put('wallGT', self.wallGT)
        conductance_sum = self.wallG.copy()
        for a in range(self.dim):
            left, right = [slice(None)] * self.dim, [slice(None)] * self.dim
            left[a], right[a] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            resistance = dx / (2 * k[left + (a,)]) + dx / (2 * k[right + (a,)])
            for pair, rc in (contacts or {}).items():
                ia, ib = pair
                match = ((ids[left] == ia) & (ids[right] == ib)) | ((ids[left] == ib) & (ids[right] == ia))
                resistance += match * rc
            g = dx**(self.dim - 1) / resistance
            gg = np.zeros(self.shape); gg[left] = g
            self.put(f'G{a}', gg)
            conductance_sum[left] += g; conductance_sum[right] += g
        self.diffusive_cfl = float(np.max(dt * conductance_sum / self.capacity))
        if self.diffusive_cfl >= 0.8:
            raise ValueError(f'Thermal diffusion CFL too large: {self.diffusive_cfl}')
        force = [sp.Integer(0)] * self.dim
        if flow_dtype == 'float32':
            self.fields['Tforce'] = self.dh.add_array('Tforce',dtype=np.float32)
            self.put('Tforce',np.zeros(self.shape))
            force[-1] = buoyancy * dt**2 / dx * self.fields['Tforce'].center
        else:
            force[-1] = buoyancy * dt**2 / dx * self.fields['T'].center * self.fields['fluid'].center
        extra_config={}
        if sgs_constant is not None:
            self.fields['eddyNu']=self.dh.add_array('eddyNu',dtype=flow_dtype)
            self.put('eddyNu',np.zeros(self.shape))
            extra_config=dict(subgrid_scale_model=(SubgridScaleModel.SMAGORINSKY,sgs_constant),
                eddy_viscosity_field=self.fields['eddyNu'])
        if external_drive:
            for a in range(self.dim):
                name=f'drive{a}';self.fields[name]=self.dh.add_array(name,dtype=flow_dtype)
                self.put(name,np.zeros(self.shape));force[a]+=self.fields[name].center
        cfg = ps.CreateKernelConfig(target=ps.Target.GPU)
        cfg.default_dtype = flow_dtype
        if cumulant_correction:
            extra_config.update(galilean_correction=True, fourth_order_correction=.01)
        if entropic:
            assert not sgs_constant and not cumulant_correction
            extra_config.update(entropic=True,zero_centered=False,force_model=ForceModel.EDM)
        self.tau = 0.5 + 3 * nu * dt / dx**2
        shear_rate=1/self.tau
        if viscosity_field is not None:
            self.fields['flowOmega']=self.dh.add_array('flowOmega',dtype=flow_dtype)
            self.put('flowOmega',1/(.5+3*np.asarray(viscosity_field)*dt/dx**2))
            shear_rate=self.fields['flowOmega'].center
        # Entropy determines the second rate; a numeric TRT default would
        # assign it twice in lbmpy 2.1.1's symbolic collision construction.
        if entropic:extra_config['relaxation_rates']=[1/self.tau,sp.Symbol('omega_free')]
        self.hydro = GpuStep(data_handling=self.dh, name='hydro', config=cfg,
                            lbm_optimisation=LBMOptimisation(simplification=equilibrium_safe_simplification) if entropic else None,
                            lbm_config=LBMConfig(stencil=Stencil.D2Q9 if self.dim == 2 else Stencil.D3Q27,
                                                method=Method.TRT_KBC_N4 if entropic else Method.CUMULANT,
                                                relaxation_rate=None if entropic else shear_rate,
                                                compressible=True, force=sp.Matrix(force),**extra_config))
        add_box_boundary(self.hydro.boundary_handling)
        directions = {(0,0):'W',(0,1):'E',(1,0):'S',(1,1):'N',(2,0):'B',(2,1):'T'}
        for face in self.open_faces:
            self.hydro.boundary_handling.set_boundary(FixedDensity(1.0),
                slice_from_direction(directions[face], self.dim))
        for i,port in enumerate(self.ports):
            def port_boundary_mask(*coords,port=port):
                coords=np.broadcast_arrays(*coords);valid=np.ones(coords[0].shape,dtype=bool)
                for a,c in enumerate(coords):
                    if a==port['axis']:valid&=(c<0) if port['side']==0 else (c>=self.shape[a])
                    else:
                        # A port continues into a tangential symmetry ghost layer.
                        # Otherwise diagonal inlet links acquire spurious no-slip corners.
                        physical=(c>=0)&(c<self.shape[a])
                        if (a,0) in (symmetry_faces or ()):physical|=(c>=-1)&(c<0)
                        if (a,1) in (symmetry_faces or ()):physical|=(c>=self.shape[a])&(c<self.shape[a]+1)
                        cc=np.clip(c,.5,self.shape[a]-.5)
                        lo,hi=port['ranges'][a];valid&=(cc>=lo)&(cc<hi)&physical
                if 'circle' in port:
                    circle=port['circle'];axes=[a for a in range(self.dim) if a!=port['axis']]
                    valid&=sum((np.clip(coords[a],.5,self.shape[a]-.5)-c)**2
                        for a,c in zip(axes,circle['center']))<circle['radius']**2
                return valid
            if 'velocity_m_s' in port:
                velocity=[sp.Float(v*dt/dx) for v in port['velocity_m_s']]
                if port.get('ramp_velocity'):
                    velocity=[v*self.fields['boundaryVelocityScale'].center for v in velocity]
                bc=UBB(velocity,adapt_velocity_to_force=True,name=f'Port velocity {i}',data_type=flow_dtype)
            elif port.get('equilibrium_pressure'):
                bc=EquilibriumPressure(1.,name=f'Reservoir pressure {i}')
            elif port.get('non_equilibrium_pressure'):
                bc=NonEquilibriumPressure(1.,name=f'Extrapolated pressure {i}')
            elif port.get('simple_outflow'):
                normal=[0]*self.dim;normal[port['axis']]=-1 if port['side']==0 else 1
                bc=SimpleExtrapolationOutflow(normal,self.hydro.method.stencil,name=f'Simple outflow {i}')
            elif port.get('outflow'):
                normal=[0]*self.dim;normal[port['axis']]=-1 if port['side']==0 else 1
                bc=ExtrapolationOutflow(normal,self.hydro.method,initial_density=1.,
                    initial_velocity=tuple(0. for _ in range(self.dim)),data_type=flow_dtype,name=f'Outflow {i}')
            elif 'pressure_Pa' in port:
                bc=FixedDensity(self.fields['boundaryDensity'].center,name=f'Port pressure {i}')
                self.set_port_pressure(port['pressure_Pa'])
            else:bc=FixedDensity(1.,name=f'Port ambient {i}')
            self.hydro.boundary_handling.set_boundary(bc,mask_callback=port_boundary_mask)
        for axis,side in (symmetry_faces or ()):
            # FreeSlip expects the normal pointing from the boundary into the fluid.
            normal=[0]*self.dim;normal[axis]=1 if side==0 else -1
            def symmetry_face_mask(*coords,axis=axis,side=side):
                coords=np.broadcast_arrays(*coords);valid=np.ones(coords[0].shape,dtype=bool)
                for a,c in enumerate(coords):
                    if a==axis:
                        valid &= (c<0) if side==0 else (c>=self.shape[a])
                    else:valid &= (c>=0)&(c<self.shape[a])
                return valid
            self.hydro.boundary_handling.set_boundary(FreeSlip(self.hydro.method.stencil,normal_direction=normal),
                mask_callback=symmetry_face_mask)
        if np.any(ids):
            def obstacle(*coords):
                coords = np.broadcast_arrays(*coords)
                valid = np.ones(coords[0].shape, dtype=bool); index = []
                for axis, c in enumerate(coords):
                    ii = np.floor(c).astype(int)
                    axis_valid=(ii >= 0) & (ii < self.shape[axis])
                    # Solids intersecting a symmetry plane continue into its
                    # mirrored ghost layer. Diagonal links there must bounce
                    # off the solid rather than reflect as fluid free-slip.
                    if (axis,0) in (symmetry_faces or ()):axis_valid|=(ii==-1)
                    if (axis,1) in (symmetry_faces or ()):axis_valid|=(ii==self.shape[axis])
                    valid &= axis_valid
                    index.append(np.clip(ii, 0, self.shape[axis]-1))
                return valid & (ids[tuple(index)] != 0)
            self.hydro.boundary_handling.set_boundary(NoSlip(), mask_callback=obstacle)
        T, F = self.fields['T'], self.fields['fluid']
        U = self.dh.fields[self.hydro.velocity_data_name]
        castU=lambda expr:ps.tcast(expr,'float64') if flow_dtype=='float32' else expr
        rate = self.fields['Q'].center + self.fields['wallGT'].center - self.fields['wallG'].center*T.center
        for a in range(self.dim):
            pos, neg = [0]*self.dim, [0]*self.dim
            pos[a], neg[a] = 1, -1
            pos, neg = tuple(pos), tuple(neg)
            G = self.fields[f'G{a}']
            gp,gm=G.center,G[neg]
            if sgs_constant is not None:
                nu=self.fields['eddyNu']
                scale_t=fluid_rhocp*dx**self.dim/dt/turbulent_prandtl
                gp+=sp.Piecewise((scale_t*.5*(castU(nu.center)+castU(nu[pos])),sp.And(F.center>0,F[pos]>0)),(0,True))
                gm+=sp.Piecewise((scale_t*.5*(castU(nu.center)+castU(nu[neg])),sp.And(F.center>0,F[neg]>0)),(0,True))
            rate += gp*(T[pos]-T.center) + gm*(T[neg]-T.center)
            if advection:
                # Flux in W: volumetric heat capacity * face area * velocity * theta_upwind.
                scale = fluid_rhocp * dx**self.dim / dt
                up = sp.Piecewise(((castU(U.center(a))+castU(U[pos](a)))*sp.Rational(1,2),sp.And(F.center>0,F[pos]>0)),(0,True))
                um = sp.Piecewise(((castU(U.center(a))+castU(U[neg](a)))*sp.Rational(1,2),sp.And(F.center>0,F[neg]>0)),(0,True))
                flux_p = sp.Max(up,0)*T.center + sp.Min(up,0)*T[pos]
                flux_m = sp.Max(um,0)*T[neg] + sp.Min(um,0)*T.center
                rate -= scale*(flux_p-flux_m)
        for i,port in enumerate(self.ports):
            axis=port['axis'];sign=-1 if port['side']==0 else 1
            un=sign*castU(U.center(axis))
            # Inflow theta=0 (ambient); outflow zero normal gradient.
            # Retain half-cell conduction only on incoming links.
            conductivity=float(np.mean(k[self.port_masks[i],axis]))
            inlet_conduction=2*conductivity*dx**(self.dim-2)*sp.Piecewise((1,un<0),(0,True))
            rate-=self.fields[f'port{i}'].center*(fluid_rhocp*dx**self.dim/dt*sp.Max(un,0)+inlet_conduction)*T.center
        updated=sp.Symbol('updated_temperature')
        assignments=[ps.Assignment(updated,T.center+dt*self.fields['invC'].center*rate),
                     ps.Assignment(self.fields['Tnext'].center,updated)]
        if flow_dtype=='float32':
            assignments.append(ps.Assignment(self.fields['Tforce'].center,ps.tcast(updated*F.center,'float32')))
        thermal_cfg=ps.CreateKernelConfig(target=ps.Target.GPU)
        thermal_cfg.default_dtype='float64'
        self.thermal = ps.create_kernel(assignments, config=thermal_cfg).compile()
        self.force_update=None
        if any(port.get('outflow') for port in self.ports):
            # Outflow history initialization reads CPU PDFs; the GPU adapter
            # normally keeps these resident and does not maintain a CPU mirror.
            self.dh.to_cpu(self.hydro._pdf_arr_name)
        # Compile/upload all kernels and boundary lists before CUDA graph capture.
        self.hydro.time_step(); self.dh.run_kernel(self.thermal)
        cp.cuda.get_current_stream().synchronize()
        self.reset()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.batch = 20
        with self.stream:
            self.stream.begin_capture()
            for _ in range(self.batch):
                if self.force_update:self.dh.run_kernel(self.force_update)
                self.hydro.time_step()
                self.dh.run_kernel(self.thermal)
                self.dh.swap('T', 'Tnext', gpu=True)
            self.graph = self.stream.end_capture()
            self.stream.begin_capture()
            for _ in range(self.batch):self.hydro.time_step()
            self.flow_graph = self.stream.end_capture()
            self.stream.begin_capture()
            for _ in range(self.batch):
                self.dh.run_kernel(self.thermal);self.dh.swap('T','Tnext',gpu=True)
            self.thermal_graph=self.stream.end_capture()

    def set_port_pressure(self,pressure_Pa):
        rho=1+3*pressure_Pa/self.fluid_density*(self.dt/self.dx)**2
        self.dh.gpu_arrays['boundaryDensity'].fill(rho)
        cp.cuda.get_current_stream().synchronize()

    def set_port_velocity_scale(self,value):
        self.dh.gpu_arrays['boundaryVelocityScale'][...]=self.port_velocity_profile*value
        cp.cuda.get_current_stream().synchronize()

    def advance_thermal(self,steps):
        """Physical heat transient with an explicitly frozen/pre-established velocity field."""
        assert steps%self.batch==0
        for _ in range(steps//self.batch):self.thermal_graph.launch(self.stream)
        self.stream.synchronize();self.time+=steps*self.dt

    def put(self, name, values):
        arr = self.dh.cpu_arrays[name]
        arr.fill(0); arr[self.inner] = values
        self.dh.to_gpu(name)
        if name=='T' and 'Tforce' in self.fields:
            forceT=self.dh.cpu_arrays['Tforce'];forceT.fill(0)
            forceT[self.inner]=np.where(self.ids==0,values,0)
            self.dh.to_gpu('Tforce')
        # advance()/advance_flow() use a non-blocking CUDA stream. Complete
        # host uploads before that stream reads a changed temperature/drive.
        cp.cuda.get_current_stream().synchronize()

    def reset(self, temperature=None):
        self.put('T', np.zeros(self.shape) if temperature is None else temperature)
        self.put('Tnext', np.zeros(self.shape))
        self.dh.cpu_arrays[self.hydro.velocity_data_name].fill(0)
        self.dh.cpu_arrays[self.hydro.density_data_name].fill(1)
        self.hydro.set_pdf_fields_from_macroscopic_values()
        cp.cuda.get_current_stream().synchronize()
        self.time = 0.0

    def advance(self, steps):
        if self.open_faces:
            raise ValueError('Open faces require the steady thermal operator; use advance_flow().')
        if steps % self.batch:
            raise ValueError('steps must be a multiple of graph batch')
        for _ in range(steps // self.batch):
            self.graph.launch(self.stream)
        self.stream.synchronize()
        self.time += steps*self.dt

    def advance_flow(self, steps):
        """Hydrodynamic relaxation at fixed T; NOT coupled physical elapsed time."""
        assert steps % self.batch == 0
        for _ in range(steps // self.batch):self.flow_graph.launch(self.stream)
        self.stream.synchronize()

    def arrays(self):
        temperature = cp.asnumpy(self.dh.gpu_arrays['T'][self.inner])
        velocity = cp.asnumpy(self.dh.gpu_arrays[self.hydro.velocity_data_name][self.inner])
        density = cp.asnumpy(self.dh.gpu_arrays[self.hydro.density_data_name][self.inner])
        velocity[self.ids != 0] = 0
        if not np.isfinite(temperature).all() or not np.isfinite(velocity).all():
            tb=np.argwhere(~np.isfinite(temperature));ub=np.argwhere(~np.isfinite(velocity))
            raise RuntimeError(f'Non-finite solution: temperature {len(tb)}, velocity {len(ub)}; '
                               f'first T {tb[:1].tolist()}, first U {ub[:1].tolist()}')
        max_lattice_speed = float(np.linalg.norm(velocity, axis=-1).max())
        if max_lattice_speed > 0.15:
            where=np.unravel_index(np.argmax(np.linalg.norm(velocity,axis=-1)),self.shape)
            raise RuntimeError(f'Lattice velocity exceeds conservative limit: {max_lattice_speed}; '
                f'cell {where}, density range {density[self.ids==0].min()} .. {density[self.ids==0].max()}')
        return temperature, velocity*self.dx/self.dt, density


def package_geometry(dx_mm=1., lengths_mm=(96.,80.,112.), lift_mm=0.):
    import trimesh
    spec=json.loads((BASE/'geometry/am5_surrogate/model.json').read_text())
    lengths=np.array(lengths_mm); offset=np.array([(lengths[0]-96)/2,(lengths[1]-64)/2,lift_mm])
    shape=tuple(np.rint(lengths/dx_mm).astype(int));ids=np.zeros(shape,dtype=np.uint8)
    for part in spec['parts']:
        mesh=trimesh.load_mesh(BASE/'geometry/am5_surrogate'/part['file'])
        mesh.apply_translation(offset)
        ranges=[]
        for axis,n in enumerate(shape):
            centers=(np.arange(n)+.5)*dx_mm
            ranges.append(np.flatnonzero((centers>=mesh.bounds[0,axis])&(centers<mesh.bounds[1,axis])))
        index=np.ix_(*ranges)
        points=np.stack(np.meshgrid(*[(r+.5)*dx_mm for r in ranges],indexing='ij'),axis=-1)
        mask=mesh.contains(points.reshape(-1,3)).reshape(points.shape[:-1])
        old=ids[index];old[mask]=part['material'];ids[index]=old
    return ids,spec,offset
