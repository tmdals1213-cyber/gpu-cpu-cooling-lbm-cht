# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Case-air CFD around the solved water block, initially one-way thermal coupling.

Water-block/package temperatures come from the fine water CHT solution. The
air heat flux is measured explicitly to assess feedback. It must not simply
be added to the original water-only heat budget and called a closed balance.
"""
from pathlib import Path
import argparse,json,time
from collections import deque
import numpy as np
from scipy.ndimage import map_coordinates
from case_setup import ROOT,build_buffered
import cupy as cp
from thermal import ThermalOperator
from checkpoint_io import save_fields,save_json

class PrescribedBodyOperator(ThermalOperator):
    def __init__(self,sim,k,fixed_temperature):
        super().__init__(sim,k)
        self.fixed=cp.asarray(sim.ids!=0);self.Tfixed=cp.asarray(fixed_temperature)
    def update(self,u,eddy_nu=None):
        super().update(u,eddy_nu)
        rhs=cp.zeros(self.shape,dtype='float64')
        for axis in range(3):
            left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
            left,right=tuple(left),tuple(right)
            a=self.fixed[left];b=self.fixed[right]
            forward=self.coeff[2*axis+2][left];backward=self.coeff[2*axis+1][right]
            rhs[left]+=cp.where(~a&b,-forward*self.Tfixed[right],0.)
            rhs[right]+=cp.where(a&~b,-backward*self.Tfixed[left],0.)
            forward[~a&b]=0;backward[a&~b]=0
        self.coeff[:,self.fixed]=0.;self.coeff[0][self.fixed]=1.
        rhs[self.fixed]=0.;self.rhs=rhs.ravel();self.normb=float(cp.linalg.norm(self.rhs));self.diag=self.coeff[0].ravel()
    def body_heat(self,unknown):
        total=cp.asarray(0.,dtype='float64')
        for axis in range(3):
            left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
            left,right=tuple(left),tuple(right);g=self.G[axis][left]
            a=self.fixed[left];b=self.fixed[right]
            total+=cp.sum(cp.where(a&~b,g*(self.Tfixed[left]-unknown[right]),0.))
            total+=cp.sum(cp.where(~a&b,g*(self.Tfixed[right]-unknown[left]),0.))
        return 2*float(total)

def prescribed_temperature(ids):
    water=np.load(ROOT/'cold_plate_run/steady_fields.npz')['T_half']
    fixed=ids!=0;coords=np.argwhere(fixed).astype(float)+.5
    local=(coords-np.array([88.,60.,20.]))/.25-.5
    assert np.all(local>=0) and np.all(local<np.array(water.shape))
    value=np.zeros(ids.shape);value[fixed]=map_coordinates(water,local.T,order=1,mode='nearest')-25.
    return value

def seed_flow(sim,path,seed_dt):
    if not path:return
    saved=np.load(path)
    U=saved['instantaneous_velocity_half'] if 'instantaneous_velocity_half' in saved else saved['velocity_half']
    R=saved['density_half'] if 'density_half' in saved else np.ones(sim.shape)
    U=U.copy();U[sim.ids!=0]=0
    vel=sim.dh.cpu_arrays[sim.hydro.velocity_data_name];vel.fill(0);vel[sim.inner]=U*sim.dt/sim.dx
    rho=sim.dh.cpu_arrays[sim.hydro.density_data_name];rho.fill(1);rho[sim.inner]=1+(R-1)*(sim.dt/seed_dt)**2
    backend=sim.flow_backend;fv=backend.dh.cpu_arrays[sim.hydro.velocity_data_name][backend.inner]
    fr=backend.dh.cpu_arrays[sim.hydro.density_data_name][backend.inner]
    fv[:,:32,:320]=U[:,0,None,:,:]*sim.dt/sim.dx;fv[:,32:232,320:]=U[:,:,None,-1,:]*sim.dt/sim.dx
    fv[backend.ids!=0]=0;scale=(sim.dt/seed_dt)**2
    fr[:,:32,:320]=1+(R[:,0,None,:]-1)*scale*np.linspace(0,1,32)[None,:,None]
    fr[:,32:232,320:]=1+(R[:,:,-1,None]-1)*scale*np.linspace(1,0,32)[None,None,:]
    backend.dh.to_gpu(sim.hydro.velocity_data_name);backend.dh.to_gpu(sim.hydro.density_data_name)
    sim.hydro.set_pdf_fields_from_macroscopic_values();cp.cuda.get_current_stream().synchronize()

def run(args):
    folder=ROOT/args.name;folder.mkdir(exist_ok=True);start=time.perf_counter()
    assert not (folder/'history.json').exists(),'Use a new --name to preserve previous results'
    sim,full,offset,meta=build_buffered('water_case_air',0.,args.dt)
    fixed_temperature=prescribed_temperature(sim.ids);fixed=sim.ids!=0
    op=PrescribedBodyOperator(sim,sim.case_conductivity,fixed_temperature)
    T=fixed_temperature.copy();sim.put('T',T);seed_flow(sim,args.seed,args.seed_dt)
    sim.set_port_velocity_scale(1.)
    water=json.loads((ROOT/'cold_plate_run/result.json').read_text())
    meta.update(status='running',case_flow_CFM=40.,power_W=200.,air_volumetric_source_W=0.,
        cooler=dict(description='Sealed 48 x 80 x 12 mm water-block envelope; fine internal water CFD is separate'),
        thermal_coupling='One-way prescribed surface temperatures from the 0.25 mm water/package solution; feedback not yet applied',
        CPU_C=water['CPU_C'],water_result='cold_plate_run/result.json',physical_time=False,
        air_case_boundary='25 C supply, 40 CFM prescribed intake, pressure outlet, h=5 W/m2K external walls',
        screening_criteria=dict(heat_change_W=.05 if args.pr_screening else .01,
            energy_residual_fraction=.01 if args.pr_screening else .001,
            volume_imbalance_fraction=.02 if args.pr_screening else .01,
            consecutive_passes=3 if args.pr_screening else 5))
    save_json(folder/'metadata.json',meta)
    history=[];means=deque(maxlen=4);previous=None;stable=0;unknown=np.zeros(sim.shape)
    for it in range(args.max_iter):
        meanU=np.zeros(sim.shape+(3,),dtype='float32');meanE=np.zeros(sim.shape,dtype='float32')
        for _ in range(10):
            sim.advance_flow(1000);_,U,R=sim.arrays();meanU+=U/10
            E=cp.asnumpy(sim.dh.gpu_arrays['eddyNu'][sim.inner]);E[fixed]=0;meanE+=E/10
        means.append((meanU,meanE));Umean=np.mean([x[0] for x in means],axis=0);Emean=np.mean([x[1] for x in means],axis=0)
        op.update(cp.asarray(Umean,dtype='float64'),cp.asarray(Emean,dtype='float64'))
        solved,info=op.solve(unknown,rtol=1e-9);unknown=cp.asnumpy(solved)
        T=unknown.copy();T[fixed]=fixed_temperature[fixed];sim.put('T',T)
        balance=op.balances(solved);heat=op.body_heat(solved);qout=2*balance['boundary_heat_W']
        air_peak=float(T[~fixed].max()+25);m=sim.port_masks[1];weights=np.maximum(Umean[...,2][m],0)
        outlet=float(np.sum(weights*T[m])/np.sum(weights)+25)
        row=dict(iteration=it+1,heat_to_air_W=heat,heat_out_W=qout,
            energy_residual_fraction=abs(qout-heat)/max(abs(heat),1e-20),volume_imbalance_fraction=balance['volume_imbalance'],
            air_peak_C=air_peak,outlet_bulk_C=outlet,CPU_C=water['CPU_C'],
            heat_change_W=None if previous is None else abs(heat-previous),
            flow_relaxation_s=(it+1)*10000*sim.dt,max_speed_m_s=float(np.linalg.norm(U,axis=-1).max()),
            thermal_solver=info,wallclock_s=time.perf_counter()-start)
        criteria=meta['screening_criteria']
        good=(it>=12 and previous is not None and abs(heat-previous)<criteria['heat_change_W']
            and row['energy_residual_fraction']<criteria['energy_residual_fraction']
            and balance['volume_imbalance']<criteria['volume_imbalance_fraction'])
        stable=stable+1 if good else 0;previous=heat;row['consecutive_passes']=stable;history.append(row)
        save_json(folder/'history.json',history);print(json.dumps(row),flush=True)
        save_fields(folder/'latest_fields.npz',T_half=(T+25).astype('float32'),velocity_half=Umean,
            instantaneous_velocity_half=U,density_half=R,eddy_nu_half=Emean,materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
        if stable>=criteria['consecutive_passes']:
            meta.update(status='one_way_case_air_screening_converged',final=row,heat_to_air_W=heat,
                air_peak_C=air_peak,air_outlet_bulk_C=outlet,heat_fraction_of_CPU_reference=heat/200,
                original_local_air_heat_W=.4533152354,additional_air_heat_W=heat-.4533152354,
                feedback_assessment='Compare measured air heat loss with original local-air loss before claiming a closed water+air 200 W energy balance')
            save_fields(folder/'steady_fields.npz',T_half=(T+25).astype('float32'),velocity_half=Umean,
                eddy_nu_half=Emean,materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
            save_json(folder/'metadata.json',meta);print('RESULT',json.dumps(meta),flush=True);return
    meta.update(status='not_converged',last=history[-1]);save_json(folder/'metadata.json',meta)
    raise RuntimeError('Case-air heat-flux convergence not established')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.00002);p.add_argument('--seed')
    p.add_argument('--seed-dt',type=float,default=.00002);p.add_argument('--max-iter',type=int,default=80)
    p.add_argument('--name',default='water_case_air_steady');p.add_argument('--pr-screening',action='store_true');run(p.parse_args())
