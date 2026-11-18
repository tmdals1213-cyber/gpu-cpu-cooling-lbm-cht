# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portable cold-start air-flow preparation; no auxiliary thermal equilibrium.

This is an alternative to the omitted historical warm-start fields, not a replay
of their exact velocity. Physics/discretization and the 1200 rpm actuator are
shared with Revision C. The isothermal preparation interval is bounded.
"""
import argparse,json,time
import numpy as np
from case_setup import ROOT,build_buffered,FAN
import cupy as cp
from checkpoint_io import save_fields,save_json
from prepare_forced_flow import port_rates


def run(args):
    if not 0<args.average_window<=args.duration:raise ValueError('Invalid averaging interval')
    out=ROOT/args.name;out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():raise FileExistsError('Use a new output root; previous results are preserved')
    start=time.perf_counter();sim,full,offset,meta=build_buffered(args.case,200.,args.dt)
    sim.put('T',np.zeros(sim.shape));fluid=sim.ids==0
    x,y,z=np.meshgrid(*[np.arange(n)+.5 for n in sim.shape],indexing='ij',sparse=True)
    disk=((x-120)**2+(y-100)**2<60**2)&(z>=98)&(z<102)&fluid;disk_gpu=cp.asarray(disk)
    block=2000;steps=round(args.duration/args.dt)
    if steps<block or steps%block:raise ValueError('Preparation duration must be a multiple of 2000 flow steps')
    totalU=np.zeros(sim.shape+(3,),dtype='float32');totalE=np.zeros(sim.shape,dtype='float32');samples=0;dp=0.;history=[]
    meta.update(status='running',method='Cold-start bounded isothermal GPU LBM flow preparation',
        approximation='No thermal-equilibrium solve. All air stays at 25 C; this differs from the archived CPU-fan warm-start field.',
        thermal_equilibrium_converged=False,dt_s=args.dt,duration_s=args.duration,average_window_s=args.average_window,
        reference_replay=False)
    save_json(out/'metadata.json',meta)
    try:
        for step in range(block,steps+1,block):
            t=step*args.dt;ramp=min(1.,t/.5);sim.set_port_velocity_scale(ramp)
            if args.case=='cpu_fan':
                lu=sim.dh.gpu_arrays[sim.hydro.velocity_data_name][sim.inner]
                q=max(0.,-2*float(cp.sum(lu[...,2][disk_gpu]))/4*sim.dx**3/sim.dt)
                ratio=1200/FAN['rated_rpm']*ramp
                dp=.8*dp+.2*FAN['shutoff_Pa']*(ratio**2-(q/FAN['free_flow_m3_s'])**2)
                sim.put('drive2',np.where(disk,-dp/(meta['fluid']['rho']*.004)*sim.dt**2/sim.dx,0.))
            sim.advance_flow(block);_,U,R=sim.arrays()
            speed=float(np.linalg.norm(U,axis=-1).max());mach=speed*sim.dt/sim.dx*np.sqrt(3)
            if not np.isfinite(U).all() or not np.isfinite(R[fluid]).all() or R[fluid].min()<=0 or mach>.25:
                raise RuntimeError('Flow stability guard reached')
            if t>args.duration-args.average_window+1e-9:
                E=cp.asnumpy(sim.dh.gpu_arrays['eddyNu'][sim.inner]);E[~fluid]=0
                totalU+=U;totalE+=E;samples+=1
            row=dict(time_s=t,max_Mach=mach,max_speed_m_s=speed,fan_dp_Pa=dp,**port_rates(sim,U))
            history.append(row);save_json(out/'history.json',history)
            if step%(block*10)==0:print(json.dumps(row),flush=True)
        if not samples:raise RuntimeError('No averaging samples; increase the averaging window')
        U=totalU/samples;E=totalE/samples;final=dict(history[-1],**port_rates(sim,U),averaging_window_s=samples*block*sim.dt)
        save_fields(out/'flow_fields.npz',velocity_half=U,eddy_nu_half=E,materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
        if final['volume_imbalance_fraction']>.02:raise RuntimeError('Mean volume imbalance exceeds 2%; inspect diagnostics and adjust preparation duration')
        meta.update(status='flow_prepared_for_heat_history',final=final,wallclock_s=time.perf_counter()-start)
        save_json(out/'metadata.json',meta)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),last=history[-1] if history else None)
        save_json(out/'metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['ventilation','cpu_fan'],required=True)
    p.add_argument('--name',default='flow');p.add_argument('--dt',type=float,required=True)
    p.add_argument('--duration',type=float,default=4.);p.add_argument('--average-window',type=float,default=1.)
    run(p.parse_args())
