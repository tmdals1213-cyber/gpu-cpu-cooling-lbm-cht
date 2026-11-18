# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical thermal transient in an explicitly declared pre-established CFD flow.

This reduced workflow is for forced cooling when a fully coupled physical-time
run is too costly. Heat capacities, contact resistances and power are unchanged.
Flow time fluctuations and transient buoyancy feedback are not resolved here.
"""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from case_setup import ROOT,geometry,configuration,MATERIALS,CONTACT,AIR,ALL_FACES
from checkpoint_io import save_fields,save_json
from thermal_convergence import evaluate_temperature
import cupy as cp
from thermal import ThermalOperator


def thermal_model(case,lb_dt):
    """Rebuild the same physical thermal grid without allocating LBM PDFs."""
    props=copy.deepcopy(MATERIALS)
    if case=='cold_plate':
        from cold_plate import geometry as water_geometry,configuration as water_configuration,WATER
        full,offset=water_geometry();meta=water_configuration();dx=.00025;fluid=WATER;reference=30.
        props[7]=props[8]
        ranges=[(40,216),(0,320),(48,80)]
        ports=[dict(axis=1,side=0,ranges=ranges),dict(axis=1,side=1,ranges=ranges)]
    else:
        full,offset=geometry(case);meta=configuration(case);dx=.001;fluid=AIR;reference=25.
        ports=meta['ports']
    ids=full[:full.shape[0]//2].copy();shape=ids.shape
    k=np.zeros(shape+(3,));capacity=np.zeros(shape);q=np.zeros(shape)
    for material,(rho,heatcap,conductivity) in props.items():
        if material==0:rho,heatcap,conductivity=fluid['rho'],fluid['cp'],[fluid['k']]*3
        mask=ids==material;k[mask]=conductivity;capacity[mask]=rho*heatcap*dx**3
    q[ids==2]=90/np.sum(ids==2);q[ids==4]=10/np.sum(ids==4)
    assert np.all(capacity>0) and abs(2*q.sum()-200)<1e-8
    wallG=np.zeros(shape);wallGT=np.zeros(shape)
    for axis,side in ALL_FACES:
        if (axis,side)==(0,1):continue
        face=[slice(None)]*3;face[axis]=0 if side==0 else -1;face=tuple(face)
        conductance=dx**2/(dx/(2*k[face+(axis,)])+1/5.)
        wallG[face]+=conductance;wallGT[face]+=conductance*(25-reference)
    coords=np.meshgrid(*[np.arange(n)+.5 for n in shape],indexing='ij',sparse=True)
    masks=[]
    for port in ports:
        mask=np.ones(shape,dtype=bool)
        for axis,(lo,hi) in enumerate(port['ranges']):mask&=(coords[axis]>=lo)&(coords[axis]<hi)
        face=np.zeros(shape,dtype=bool);sl=[slice(None)]*3;sl[port['axis']]=0 if port['side']==0 else -1
        face[tuple(sl)]=True;mask&=face;masks.append(mask);wallG[mask]=0;wallGT[mask]=0
    arrays={}
    for axis in range(3):
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        left,right=tuple(left),tuple(right)
        resistance=dx/(2*k[left+(axis,)])+dx/(2*k[right+(axis,)])
        for (a,b),value in CONTACT.items():
            resistance+=value*(((ids[left]==a)&(ids[right]==b))|((ids[left]==b)&(ids[right]==a)))
        conductance=np.zeros(shape);conductance[left]=dx**2/resistance
        arrays[f'G{axis}']=cp.asarray(conductance)
    sim=SimpleNamespace(ids=ids,shape=shape,dx=dx,dt=lb_dt,capacity=capacity,source_w=q,
        wallG=wallG,wallGT=wallGT,case_conductivity=k,fluid_rhocp=fluid['rho']*fluid['cp'],
        ports=ports,port_masks=masks,sgs_constant=None if case=='cold_plate' else .12,
        turbulent_prandtl=.9,inner=(slice(None),)*3,dh=SimpleNamespace(gpu_arrays=arrays))
    return sim,full,offset,meta,reference


def run(args):
    if args.heat_dt is None:args.heat_dt=.5 if args.case=='cold_plate' else 2.
    if not 0<args.heat_dt<=2.:raise ValueError('Heat step must be positive and <=2 s; final event/settling phase uses <=1 s')
    folder=ROOT/args.name;folder.mkdir(exist_ok=True)
    assert not (folder/'history.json').exists(),'Use a new output name'
    sim,full,offset,meta,reference=thermal_model(args.case,args.lb_dt)
    flow=np.load(args.flow);U=flow['velocity_half'].copy();U[sim.ids!=0]=0
    flow_meta_path=Path(args.flow).parent/'metadata.json'
    flow_meta=json.loads(flow_meta_path.read_text()) if flow_meta_path.exists() else {}
    flow_record=flow_meta.get('final',flow_meta.get('last',{}))
    E=flow['eddy_nu_half'] if 'eddy_nu_half' in flow else None
    assert U.shape==sim.shape+(3,)
    op=ThermalOperator(sim,sim.case_conductivity)
    if args.case!='cold_plate':
        from heatpipes import apply_implicit_conductance
        apply_implicit_conductance(op)
    op.update(cp.asarray(U,dtype='float64'),None if E is None else cp.asarray(E,dtype='float64'))
    meta.update(status='running',method='Physical transient heat in pre-established CFD mean flow',
        time_axis='Physical thermal seconds; flow held fixed',flow_source=str(args.flow),
        flow_preparation_status=flow_meta.get('status'),
        flow_preparation_method=flow_meta.get('method'),
        flow_average_window_s=flow_record.get('averaging_window_s',flow_meta.get('average_window_s')),
        auxiliary_thermal_equilibrium_converged=flow_meta.get('thermal_equilibrium_converged',False),
        initial_temperature_from_flow_file=False,
        no_artificial_thermal_acceleration=True,thermal_mass_physical=True,
        flow_dt_s=args.lb_dt,thermal_dt_s=args.heat_dt,initial_C=reference,
        thermal_step_schedule_s=dict(first_second=.05,until_five_seconds=.25,after_five_seconds=args.heat_dt,
            near_limit_or_plateau=min(args.heat_dt,1.)),
        thermal_fine_phase_trigger='CPU >= 90 C or absolute successive CPU rate < 0.05 C/s after t=5 s; once enabled, keep <=1 s steps',
        limit_event_time_resolution_s=None,
        temperature_settling_criterion=dict(rate_limit_C_per_s=.01,hold_s=10.,rate_window_s=1.),
        approximation=('Constant-property forced laminar water flow, zero buoyancy; flow is independent of temperature'
            if args.case=='cold_plate' else 'Mean flow retained; flow startup, fluctuations and evolving buoyancy feedback omitted'))
    save_json(folder/'metadata.json',meta)
    save_fields(folder/'model.npz',materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
    save_fields(folder/'flow_fields.npz',velocity_half=U,**({} if E is None else dict(eddy_nu_half=E)))
    T=cp.zeros(sim.shape,dtype='float64');history=[];t=0.;energy_out=0.;start=time.perf_counter();next_save=0.
    fine_phase=args.heat_dt<=1.;last_dt=None
    cpu=cp.asarray(np.isin(sim.ids,[2,3,4]));capacity=cp.asarray(sim.capacity)
    def peak(mask):return float(cp.max(T[cp.asarray(mask)]))+reference
    def snapshot(final=False):
        ta=cp.asnumpy(T)+reference
        save_fields(folder/f'frame_{len(list(folder.glob("frame_*.npz"))):04d}.npz',T_half=ta.astype('float32'),time_s=t)
        if final:save_fields(folder/'last_fields.npz',T_half=ta.astype('float32'),velocity_half=U,
            materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset,time_s=t)
    try:
        while True:
            balance=op.balances(T);stored=2*float(cp.sum(T*capacity));supplied=200*t
            row=dict(time_s=t,CPU_C=float(cp.max(T[cpu]))+reference,IHS_C=peak(sim.ids==5),
                base_C=peak(sim.ids==8),fins_C=peak(sim.ids==7),fluid_peak_C=peak(sim.ids==0),
                minimum_C=float(cp.min(T))+reference,heat_out_W=2*balance['boundary_heat_W'],
                stored_J=stored,supplied_J=supplied,cumulative_heat_out_J=energy_out,
                energy_residual_J=supplied-stored-energy_out,
                energy_residual_fraction=abs(supplied-stored-energy_out)/max(supplied,1.),
                volume_imbalance_fraction=balance['volume_imbalance'],wallclock_s=time.perf_counter()-start,
                previous_heat_dt_s=last_dt,fine_thermal_phase=fine_phase)
            if args.case=='cold_plate':
                out=sim.port_masks[1];weights=np.maximum(U[...,1][out],0.)
                row['coolant_outlet_bulk_C']=float(cp.sum(cp.asarray(weights)*T[cp.asarray(out)]))/float(weights.sum())+reference
                row['coolant_peak_C']=row['fluid_peak_C']
            else:row['air_peak_C']=row['fluid_peak_C']
            row.update(evaluate_temperature(history+[row]))
            stop=row['CPU_C']>=95;settled=row['temperature_settled'];end=t>=args.max_time-1e-8
            if not np.isfinite(row['CPU_C']) or row['minimum_C']<24.99:raise RuntimeError('Non-finite or undershooting heat solution')
            if t>1. and row['energy_residual_fraction']>.01:raise RuntimeError('Major heat-history energy imbalance')
            if t>=next_save-1e-8 or stop or settled or end:
                snapshot(final=stop or settled or end)
                interval=min(args.save_interval,.1 if t<1.-1e-8 else .25 if t<5.-1e-8 else args.save_interval)
                next_save=t+interval
            history.append(row);save_json(folder/'history.json',history)
            if len(history)==1 or t%5<1e-8 or stop or settled or end:print(json.dumps(row),flush=True)
            if stop or settled or end:
                status='temperature_limit_reached' if stop else 'CPU_temperature_settled' if settled else 'physical_time_window_complete_not_steady'
                meta.update(status=status,final=row,physical_time_s=t,wallclock_s=time.perf_counter()-start,
                    limit_event_time_resolution_s=last_dt)
                save_json(folder/'metadata.json',meta);print('RESULT',json.dumps(meta),flush=True);return
            if t>=5. and len(history)>=2:
                previous=history[-2]
                rate=abs((row['CPU_C']-previous['CPU_C'])/(t-previous['time_s']))
                fine_phase=fine_phase or row['CPU_C']>=90. or rate<.05
            step_cap=min(args.heat_dt,1.) if fine_phase else args.heat_dt
            dt=min(step_cap,.05 if t<1.-1e-8 else .25 if t<5.-1e-8 else step_cap,args.max_time-t)
            T,info=op.step_implicit(T,dt);t=round(t+dt,10)
            last_dt=dt
            energy_out+=2*op.balances(T)['boundary_heat_W']*dt
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),last=history[-1] if history else None)
        save_json(folder/'metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['ventilation','cpu_fan','cold_plate'],required=True)
    p.add_argument('--flow',required=True);p.add_argument('--name',required=True)
    p.add_argument('--lb-dt',type=float,default=.00002);p.add_argument('--heat-dt',type=float,help='Air: maximum 2 s, reduced to <=1 s near limit/plateau; water default 0.5 s. Startup uses smaller steps.')
    p.add_argument('--max-time',type=float,default=1200.);p.add_argument('--save-interval',type=float,default=2.)
    run(p.parse_args())
