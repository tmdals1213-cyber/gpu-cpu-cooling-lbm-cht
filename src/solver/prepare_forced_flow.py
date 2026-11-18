# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded isothermal ventilation adjustment; no thermal-equilibrium solves."""
import argparse,json,time
from pathlib import Path
import numpy as np
from case_setup import ROOT,build_buffered
from checkpoint_io import save_fields,save_json
import cupy as cp


def port_rates(sim,velocity):
    incoming=outgoing=0.
    for port,mask in zip(sim.ports,sim.port_masks):
        outward=velocity[...,port['axis']][mask]*(1 if port['side'] else -1)
        incoming+=2*float(np.maximum(-outward,0).sum())*sim.dx**2
        outgoing+=2*float(np.maximum(outward,0).sum())*sim.dx**2
    return dict(volume_in_CFM=incoming/.00047194745,volume_out_CFM=outgoing/.00047194745,
        volume_imbalance_fraction=abs(incoming-outgoing)/max(incoming,outgoing,1e-20))


def run(args):
    folder=ROOT/args.name;folder.mkdir(exist_ok=True)
    assert not (folder/'history.json').exists(),'Use a new output name'
    assert 0<args.average_window<=args.duration<=2.
    start=time.perf_counter();sim,full,offset,meta=build_buffered('ventilation',200.,args.dt)
    fluid=sim.ids==0;backend=sim.flow_backend
    with np.load(args.seed) as seed:
        U=seed['instantaneous_velocity_half'].copy() if 'instantaneous_velocity_half' in seed else seed['velocity_half'].copy()
        R=seed['density_half'].copy() if 'density_half' in seed else np.ones(sim.shape)
    U[~fluid]=0
    sim.put('T',np.zeros(sim.shape));sim.set_port_velocity_scale(1.)
    velocity=sim.dh.cpu_arrays[sim.hydro.velocity_data_name];velocity.fill(0);velocity[sim.inner]=U*sim.dt/sim.dx
    density=sim.dh.cpu_arrays[sim.hydro.density_data_name];density.fill(1);density[sim.inner]=R
    fv=backend.dh.cpu_arrays[sim.hydro.velocity_data_name][backend.inner]
    fr=backend.dh.cpu_arrays[sim.hydro.density_data_name][backend.inner]
    fv[:,:32,:320]=U[:,0,None,:,:]*sim.dt/sim.dx
    fv[:,32:232,320:]=U[:,:,None,-1,:]*sim.dt/sim.dx
    fv[backend.ids!=0]=0
    fr[:,:32,:320]=1+(R[:,0,None,:]-1)*np.linspace(0,1,32)[None,:,None]
    fr[:,32:232,320:]=1+(R[:,:,-1,None]-1)*np.linspace(1,0,32)[None,None,:]
    backend.dh.to_gpu(sim.hydro.velocity_data_name);backend.dh.to_gpu(sim.hydro.density_data_name)
    sim.hydro.set_pdf_fields_from_macroscopic_values();cp.cuda.get_current_stream().synchronize()
    meta.update(status='running',method='Bounded isothermal LBM ventilation preparation, without a thermal equilibrium iteration',
        time_axis='Flow preparation seconds only; not CPU heating time',seed_from=str(args.seed),
        seed_method='Velocity/pressure from earlier ventilated case; new heatpipe cells zeroed, equilibrium PDFs reinitialised',
        seed_dt_s=args.dt,duration_s=args.duration,average_window_s=args.average_window,
        temperature_during_preparation_C=25.,thermal_equilibrium_converged=False,
        approximation='Forced-flow preparation at 25 C. Evolving buoyancy and flow startup are omitted from the subsequent fixed-flow heat history.',
        acceptance='Finite velocity/density; positive density; Ma <=0.25; final mean port-volume imbalance <=2 percent. No CPU equilibrium criterion.')
    save_json(folder/'metadata.json',meta)
    history=[];sample_steps=2000;total_steps=round(args.duration/args.dt)
    assert total_steps%sample_steps==0
    sumU=np.zeros(sim.shape+(3,),dtype='float32');sumE=np.zeros(sim.shape,dtype='float32');samples=0
    try:
        for step in range(sample_steps,total_steps+1,sample_steps):
            sim.advance_flow(sample_steps);_,U,R=sim.arrays();seconds=step*args.dt
            speed=float(np.linalg.norm(U,axis=-1).max());mach=speed*args.dt/sim.dx*np.sqrt(3)
            if not np.isfinite(U).all() or not np.isfinite(R[fluid]).all() or R[fluid].min()<=0 or mach>.25:
                raise RuntimeError('Flow preparation stability guard reached')
            if seconds>args.duration-args.average_window+1e-9:
                E=cp.asnumpy(sim.dh.gpu_arrays['eddyNu'][sim.inner]);E[~fluid]=0
                sumU+=U;sumE+=E;samples+=1
            row=dict(flow_relaxation_s=seconds,flow_steps=step,max_speed_m_s=speed,max_Mach=mach,
                density_min=float(R[fluid].min()),density_max=float(R[fluid].max()),
                wallclock_s=time.perf_counter()-start,**port_rates(sim,U))
            history.append(row);save_json(folder/'history.json',history)
            if step%(5*sample_steps)==0:print(json.dumps(row),flush=True)
        meanU=sumU/samples;meanE=sumE/samples
        final=dict(history[-1],**port_rates(sim,meanU))
        final.update(averaging_window_s=samples*sample_steps*args.dt,averaged_samples=samples)
        if final['volume_imbalance_fraction']>.02:raise RuntimeError('Mean port volume imbalance exceeds 2 percent')
        save_fields(folder/'flow_fields.npz',velocity_half=meanU,eddy_nu_half=meanE,
            materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
        meta.update(status='flow_prepared_for_heat_history',final=final,wallclock_s=time.perf_counter()-start)
        save_json(folder/'metadata.json',meta);print('RESULT',json.dumps(meta),flush=True)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),last=history[-1] if history else None)
        save_json(folder/'metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',required=True)
    p.add_argument('--dt',type=float,default=.00002);p.add_argument('--duration',type=float,default=1.6)
    p.add_argument('--average-window',type=float,default=.8);run(p.parse_args())
