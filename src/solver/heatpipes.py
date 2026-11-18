# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finite-conductance heatpipe surrogate, shared by all Revision C air cases.

Six sealed 6 mm copper/water pipes have a central evaporator in the base and
two curved condenser arms through the fins. The fluid inside is not resolved.
The *assumed* internal equivalent conductivity is 10,000 W/m/K; interface
conduction uses copper (390 W/m/K). This is not a calibrated commercial pipe
or a prediction of capillary/dryout limits. Nothing is fitted to a CPU target.
"""
import numpy as np

PIPE_ID = 10
SHELL_K = 390.
INTERNAL_K = 10000.
VOLUMETRIC_CAPACITY = 1.8e6
CENTRES_Y = [60., 76., 92., 108., 124., 140.]


def add_geometry(ids):
    x, y, z = np.meshgrid(*[np.arange(n)+.5 for n in ids.shape], indexing='ij', sparse=True)
    # A slotted racetrack in x-z: bottom evaporator, two R21 mm bends,
    # separated upper condenser tips. All six circuits are independent.
    def segment(a, b, height):
        return np.sqrt((x-np.clip(x,a,b))**2+(z-height)**2)
    distance = np.minimum(segment(87,153,33),
                          np.minimum(segment(87,116,75),segment(124,153,75)))
    for xc, side in [(87,-1),(153,1)]:
        arc = np.abs(np.sqrt((x-xc)**2+(z-54)**2)-21)
        arc = np.where((x-xc)*side>=0,arc,np.inf)
        distance = np.minimum(distance,arc)
    for yc in CENTRES_Y:
        ids[distance**2+(y-yc)**2<=3**2] = PIPE_ID
    return ids


def specification():
    return dict(count=6,outer_diameter_mm=6.,bend_radius_mm=21.,
        y_centres_mm=CENTRES_Y,evaporator_z_mm=33.,condenser_top_z_mm=75.,
        equivalent_internal_conductivity_W_mK=INTERNAL_K,
        interface_conductivity_W_mK=SHELL_K,
        volumetric_heat_capacity_J_m3K=VOLUMETRIC_CAPACITY,
        pipe_to_base_and_fin_contact='Ideal bonded contact; unchanged CPU TIM resistance',
        geometry='Six independent slotted-racetrack pipes, symmetric about x=120 mm',
        model='Finite equivalent internal conductivity with copper interface conduction',
        parameter_status='Declared engineering assumption, not product-calibrated or fitted to CPU temperature',
        limits='Internal two-phase flow, startup, capillary limit, dryout and orientation-dependent performance are not resolved',
        sources=[
            'https://www.1-act.com/thermal-solutions/passive/heat-pipes/heat-pipe-design-guide/',
            'https://info.boydcorp.com/hubfs/Thermal/Two-Phase-Cooling/Boyd-Copper-Water-Heat-Pipes.pdf',
            'https://www.qats.com/cms/wp-content/uploads/Heat-Pipe-Engineering-eBook.pdf'])


def apply_implicit_conductance(op):
    """Only internal pipe-pipe faces get enhanced conduction; conserve each face."""
    import cupy as cp
    if getattr(op,'_heatpipe_enriched',False):raise ValueError('Heatpipe conductance already applied to this operator')
    mask=op.sim.ids==PIPE_ID
    for axis in range(3):
        # The base operator may alias contiguous sim.G arrays. Keep those at
        # copper conductivity so constructing another operator cannot enrich twice.
        op.G[axis]=op.G[axis].copy()
        left=[slice(None)]*3;right=left.copy()
        left[axis]=slice(None,-1);right[axis]=slice(1,None)
        left,right=tuple(left),tuple(right)
        both=mask[left]&mask[right]
        # Baseline already contains k_Cu*dx on these same-material faces.
        op.G[axis][left]+=cp.asarray(both)*(INTERNAL_K-SHELL_K)*op.sim.dx
    op._heatpipe_enriched=True


class LocalPipeDiffusion:
    """Conservative explicit subcycles on pipe nodes only (not the air grid).

    The main CHT step handles copper-level conduction on all faces. This adds
    only the excess internal heatpipe conductance. Each subcycle has a positive
    diagonal weight with CFL <= 0.60; equal/opposite face fluxes conserve energy.
    """
    def __init__(self,flow):
        import cupy as cp
        self.cp=cp;self.flow=flow
        xyz=np.argwhere(flow.ids==PIPE_ID)
        self.n=len(xyz)
        lookup=np.full(flow.shape,-1,dtype='int32')
        lookup[tuple(xyz.T)]=np.arange(self.n,dtype='int32')
        neighbours=np.full((self.n,6),-1,dtype='int32')
        for axis in range(3):
            for side,sgn in enumerate([-1,1]):
                q=xyz.copy();q[:,axis]+=sgn
                valid=np.all((q>=0)&(q<np.array(flow.shape)),axis=1)
                neighbours[valid,2*axis+side]=lookup[tuple(q[valid].T)]
        # Pystencils arrays are not guaranteed C contiguous. Index by strides.
        array=flow.dh.gpu_arrays['T']
        stride=np.array(array.strides)//array.itemsize
        self.index=cp.asarray(((xyz+1)*stride).sum(axis=1).astype('int64'))
        self.neighbours=cp.asarray(neighbours)
        self.a=cp.empty(self.n,dtype='float64');self.b=cp.empty_like(self.a)
        capacity=VOLUMETRIC_CAPACITY*flow.dx**3
        self.alpha=flow.dt*(INTERNAL_K-SHELL_K)*flow.dx/capacity
        degree=(neighbours>=0).sum(axis=1)
        self.substeps=max(1,int(np.ceil(self.alpha*degree.max()/.60)))
        self.factor=self.alpha/self.substeps
        source=r'''
        extern "C" __global__ void gather(const double* t,const long long* ix,double* a,int n){
            int i=blockDim.x*blockIdx.x+threadIdx.x;if(i<n)a[i]=t[ix[i]];
        }
        extern "C" __global__ void diffuse(const double* a,double* b,const int* nb,double f,int n){
            int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
            double delta=0.;for(int j=0;j<6;j++){int k=nb[6*i+j];if(k>=0)delta+=a[k]-a[i];}
            b[i]=a[i]+f*delta;
        }
        extern "C" __global__ void scatter(double* t,const long long* ix,const double* a,int n){
            int i=blockDim.x*blockIdx.x+threadIdx.x;if(i<n)t[ix[i]]=a[i];
        }'''
        module=cp.RawModule(code=source)
        self.gather=module.get_function('gather');self.diffuse=module.get_function('diffuse');self.scatter=module.get_function('scatter')
        self.launch=((self.n+255)//256,);self.block=(256,)
        self.diagonal_cfl=float(self.factor*degree.max())

    def step(self):
        cp=self.cp;t=self.flow.dh.gpu_arrays['T'];a,b=self.a,self.b;n=np.int32(self.n)
        self.gather(self.launch,self.block,(t,self.index,a,n))
        for _ in range(self.substeps):
            self.diffuse(self.launch,self.block,(a,b,self.neighbours,np.float64(self.factor),n));a,b=b,a
        self.scatter(self.launch,self.block,(t,self.index,a,n))


def install_transient_graph(flow):
    import cupy as cp
    local=LocalPipeDiffusion(flow)
    local.step();cp.cuda.get_current_stream().synchronize()
    with flow.stream:
        flow.stream.begin_capture()
        for _ in range(flow.batch):
            if flow.force_update:flow.dh.run_kernel(flow.force_update)
            flow.hydro.time_step()
            flow.dh.run_kernel(flow.thermal)
            flow.dh.swap('T','Tnext',gpu=True)
            local.step()
        flow.graph=flow.stream.end_capture()
    flow.heatpipe_diffusion=local
    return dict(method='Conservative local pipe diffusion subcycles after each physical CHT step',
        pipe_nodes=local.n,substeps_per_flow_step=local.substeps,
        local_diagonal_CFL=local.diagonal_cfl,
        extra_conductivity_W_mK=INTERNAL_K-SHELL_K)
