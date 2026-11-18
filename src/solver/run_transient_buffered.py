# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical-time CHT in the unchanged case, with flow-only exterior ducts.

Thermal fluxes, storage, heat sources, and buoyancy stay inside the physical
case. Both velocity and temperature advance every physical time step. Saved
steady states are deliberately not used as a CPU heating initial condition.
"""
from pathlib import Path
import argparse,json,time,math
import numpy as np
from case_setup import ROOT,build_buffered,FAN
from checkpoint_io import save_fields,save_json
from thermal_convergence import evaluate_temperature
import cupy as cp


def physical_thermal_domain(sim):
    flow=sim.flow_backend
    crop=(slice(None),slice(32,232),slice(0,320))
    def padded(value):
        out=np.zeros(flow.shape,dtype=np.float64);out[crop]=value;return out
    # This mask is used by heat advection/eddy diffusion and thermal buoyancy,
    # not by the independently constructed hydrodynamic geometry boundaries.
    flow.put('fluid',padded(sim.ids==0))
    flow.put('invC',padded(1/sim.capacity))
    flow.put('wallG',padded(sim.wallG));flow.put('wallGT',padded(sim.wallGT))
    flow.put('Q',padded(sim.source_w))
    conductance=sim.wallG.copy()
    for a in range(3):
        # Positive-face conductance on the last physical cell must vanish.
        # Port conduction is handled separately by the incoming port term.
        g=cp.asnumpy(flow.dh.gpu_arrays[f'G{a}'][flow.inner][crop]).copy()
        edge=[slice(None)]*3;edge[a]=-1;g[tuple(edge)]=0
        flow.put(f'G{a}',padded(g))
        left=[slice(None)]*3;right=left.copy();left[a]=slice(None,-1);right[a]=slice(1,None)
        left,right=tuple(left),tuple(right)
        conductance[left]+=g[left];conductance[right]+=g[left]
    for i,mask in enumerate(sim.port_masks):flow.put(f'port{i}',padded(mask))
    # The initial conductive CFL is necessary but does not bound later
    # advection/SGS diffusion; the instantaneous bound is recorded below.
    sim.conductance_sum=conductance
    flow.reset()
    return flow


def boundary_heat(sim,T,U,R):
    loss=float(np.sum(sim.wallG*T-sim.wallGT));vin=vout=0.;minlet=moutlet=0.
    for port,mask in zip(sim.ports,sim.port_masks):
        a=port['axis'];un=U[...,a][mask]*(-1 if port['side']==0 else 1)
        loss+=float(np.sum((np.maximum(un,0)*sim.fluid_rhocp*sim.dx**2+
            np.where(un<0,2*sim.case_conductivity[...,a][mask]*sim.dx,0))*T[mask]))
        vin+=float(-np.minimum(un,0).sum())*sim.dx**2
        vout+=float(np.maximum(un,0).sum())*sim.dx**2
        minlet+=float((-np.minimum(un,0)*R[mask]).sum())*sim.dx**2*sim.fluid_density
        moutlet+=float((np.maximum(un,0)*R[mask]).sum())*sim.dx**2*sim.fluid_density
    return dict(heat_out_W=2*loss,volume_in_m3_s=2*vin,volume_out_m3_s=2*vout,
        mass_in_kg_s=2*minlet,mass_out_kg_s=2*moutlet,
        volume_imbalance_fraction=abs(vin-vout)/max(vin,vout,1e-20))


def stability_bound(sim,U,E):
    fluid=sim.ids==0;diagonal=sim.conductance_sum.copy()
    for a in range(3):
        left=[slice(None)]*3;right=left.copy();left[a]=slice(None,-1);right[a]=slice(1,None)
        left,right=tuple(left),tuple(right);both=fluid[left]&fluid[right]
        extra=np.where(both,sim.fluid_rhocp*sim.dx**3/sim.dt/sim.turbulent_prandtl*.5*(E[left]+E[right]),0)
        flux=np.where(both,sim.fluid_rhocp*sim.dx**2*.5*(U[left+(a,)]+U[right+(a,)]),0)
        diagonal[left]+=extra+np.maximum(flux,0);diagonal[right]+=extra-np.minimum(flux,0)
    for port,mask in zip(sim.ports,sim.port_masks):
        a=port['axis'];un=U[...,a][mask]*(-1 if port['side']==0 else 1)
        diagonal[mask]+=np.maximum(un,0)*sim.fluid_rhocp*sim.dx**2+np.where(un<0,2*sim.case_conductivity[...,a][mask]*sim.dx,0)
    return float(np.max(sim.dt*diagonal/sim.capacity))


def run(args):
    folder=ROOT/args.name;folder.mkdir(exist_ok=True)
    assert not (folder/'history.json').exists(),'Use a new --name; existing physical-time history is preserved'
    init_start=time.perf_counter()
    sim,full,offset,meta=build_buffered(args.case,args.power,args.dt)
    flow=physical_thermal_domain(sim)
    from heatpipes import install_transient_graph
    heatpipe_integration=install_transient_graph(flow)
    cpu=np.isin(sim.ids,[2,3,4]);fluid=sim.ids==0
    cells=int(np.prod(flow.shape));source=2*float(sim.source_w.sum())
    assert abs(source-args.power)<1e-9
    meta.update(status='running',method='Fully coupled physical-time LBM + conservative explicit thermal FVM',
        heatpipe_integration=heatpipe_integration,
        time_axis='Physical seconds from uniform 25 C and initially resting air',
        initial_C=25.,power_W=args.power,source_W={'CCD1':args.power*.45,'CCD2':args.power*.45,'IOD':args.power*.1},
        no_artificial_thermal_acceleration=True,thermal_mass_physical=True,
        thermal_domain='Physical case only; exterior flow ducts held at ambient without buoyancy or thermal storage',
        grid_type='Uniform 1 mm Cartesian; local mesh refinement is not implemented in this driver',
        time_step_accuracy_validated=False,flow_LB_relaxation_time=flow.tau,
        purpose='PR/model demonstration; detailed time-step independence study not required by user',
        acceptance='Finite fields, explicit thermal stability, no major energy imbalance; underlying LBM velocity guard retained',
        temperature_settling_criterion=dict(rate_limit_C_per_s=args.steady_rate,
            hold_s=args.steady_hold,rate_window_s=1.,
            description='Absolute CPU peak temperature rate below limit throughout hold interval; physical time only'),
        setup_wallclock_s=time.perf_counter()-init_start)
    save_json(folder/'metadata.json',meta)
    history=[];previous=None;energy_out=0.;mass_added=0.;initial_mass=None
    start=time.perf_counter();step=0;dp=0.
    target_steps=int(round(args.max_time/sim.dt/flow.batch))*flow.batch
    sample_steps=max(flow.batch,int(round(args.sample_interval/sim.dt/flow.batch))*flow.batch)
    save_interval=max(sample_steps,int(round(args.save_interval/sim.dt/flow.batch))*flow.batch)
    next_save=0;last_U=None
    x,y,z=np.meshgrid(*[np.arange(n)+.5 for n in sim.shape],indexing='ij',sparse=True)
    disk=((x-120)**2+(y-100)**2<60**2)&(z>=98)&(z<102)&fluid
    disk_gpu=cp.asarray(disk)
    if args.restart:
        # Restore complete populations, including the exterior flow buffers.
        # Reinitialising only from velocity/density would discard stress and
        # introduce a new numerical startup into the physical-time record.
        saved=np.load(args.restart)
        old_dt=float(saved['dt_s'])
        assert float(saved['power_W'])==args.power
        assert np.array_equal(saved['materials_half'],sim.ids)
        physical_restart_time=int(saved['steps'])*old_dt
        step=round(physical_restart_time/sim.dt)
        assert step%flow.batch==0 and abs(step*sim.dt-physical_restart_time)<1e-8
        assert target_steps>step
        sim.put('T',saved['T_half'].astype('float64')-25.)
        if old_dt==sim.dt:
            for key,name in [('velocity_lattice',flow.hydro.velocity_data_name),('density',flow.hydro.density_data_name),('eddy_nu','eddyNu')]:
                flow.dh.gpu_arrays[name][...]=cp.asarray(saved[key])
            flow.dh.gpu_arrays[flow.hydro._pdf_arr_name][...]=cp.asarray(saved['pdf'])
            restart_kind='Complete LBM populations and thermal state; unchanged dt and geometry'
        else:
            assert args.allow_dt_change,'A changed-dt restart requires explicit --allow-dt-change'
            ratio=sim.dt/old_dt
            flow.dh.gpu_arrays[flow.hydro.velocity_data_name][...]=cp.asarray(saved['velocity_lattice'])*ratio
            flow.dh.gpu_arrays[flow.hydro.density_data_name][...]=1+(cp.asarray(saved['density'])-1)*ratio**2
            flow.dh.gpu_arrays['eddyNu'][...]=cp.asarray(saved['eddy_nu'])*ratio
            # Keep physical T, U and pressure. Non-equilibrium kinetic stresses
            # are reinitialised, so this is an approximate macroscopic restart,
            # not an exact rescaling of all populations. Record that explicitly.
            flow.hydro.set_pdf_fields_from_macroscopic_values()
            restart_kind='Physical temperature/velocity/pressure retained; equilibrium PDF reinitialisation after numerical time-step reduction'
        flow.time=step*sim.dt;energy_out=float(saved['cumulative_heat_out_J'])
        mass_added=float(saved['mass_added_kg']);initial_mass=float(saved['initial_mass_kg']);dp=float(saved['fan_dp_Pa'])
        if old_dt!=sim.dt:
            density=cp.asnumpy(sim.dh.gpu_arrays[flow.hydro.density_data_name][sim.inner])
            initial_mass=2*float(density[fluid].astype('float64').sum())*sim.fluid_density*sim.dx**3
            mass_added=0.
            meta.update(previous_dt_s=old_dt,mass_balance_scope='Restart segment after pressure-density unit rescaling',
                restart_limitation='Non-equilibrium stresses are reinitialised; no claim of exact kinetic-state continuity')
        cp.cuda.get_current_stream().synchronize()
        meta.update(restart_from=str(Path(args.restart).resolve()),restart_physical_time_s=step*sim.dt,
            restart_kind=restart_kind)
        save_json(folder/'metadata.json',meta)
    start_step=step
    checkpoint_steps=max(flow.batch,int(round(args.checkpoint_interval/sim.dt/flow.batch))*flow.batch)
    next_checkpoint=step+checkpoint_steps
    def write_restart(T):
        temporary=folder/'flow_restart.pending'
        with temporary.open('wb') as stream:
            np.savez(stream,T_half=T+25.,pdf=cp.asnumpy(flow.dh.gpu_arrays[flow.hydro._pdf_arr_name]),
                velocity_lattice=cp.asnumpy(flow.dh.gpu_arrays[flow.hydro.velocity_data_name]),
                density=cp.asnumpy(flow.dh.gpu_arrays[flow.hydro.density_data_name]),
                eddy_nu=cp.asnumpy(flow.dh.gpu_arrays['eddyNu']),materials_half=sim.ids,
                dt_s=sim.dt,power_W=args.power,steps=step,cumulative_heat_out_J=energy_out,
                mass_added_kg=mass_added,initial_mass_kg=initial_mass,fan_dp_Pa=dp)
        temporary.replace(folder/'flow_restart.npz')
    try:
        while True:
            T,U,R=sim.arrays();t=step*sim.dt
            if not np.isfinite(R[fluid]).all() or np.min(R[fluid])<=0:
                raise RuntimeError('Nonfinite or nonpositive fluid density')
            balance=boundary_heat(sim,T,U,R)
            mass=2*float(R[fluid].astype('float64').sum())*sim.fluid_density*sim.dx**3
            if initial_mass is None:initial_mass=mass
            if previous is not None:
                energy_out+=.5*(previous['heat_out_W']+balance['heat_out_W'])*(t-previous['time_s'])
                mass_added+=.5*(previous['mass_in_kg_s']-previous['mass_out_kg_s']+
                    balance['mass_in_kg_s']-balance['mass_out_kg_s'])*(t-previous['time_s'])
            stored=2*float(np.sum(T*sim.capacity));supplied=source*t
            row=dict(time_s=t,steps=step,CPU_C=float(T[cpu].max()+25),IHS_C=float(T[sim.ids==5].max()+25),
                base_C=float(T[sim.ids==8].max()+25),fins_C=float(T[sim.ids==7].max()+25),
                air_peak_C=float(T[fluid].max()+25),minimum_C=float(T.min()+25),stored_J=stored,
                supplied_J=supplied,cumulative_heat_out_J=energy_out,
                energy_residual_J=supplied-stored-energy_out,
                energy_residual_fraction=abs(supplied-stored-energy_out)/max(supplied,1e-20),
                air_mass_kg=mass,mass_balance_error_fraction_initial_mass=abs(mass-initial_mass-mass_added)/initial_mass,
                max_speed_m_s=float(np.linalg.norm(U,axis=-1).max()),
                density_min=float(R[fluid].min()),density_max=float(R[fluid].max()),
                max_Mach=float(np.linalg.norm(U,axis=-1).max()*sim.dt/sim.dx*np.sqrt(3)),
                compute_wallclock_s=time.perf_counter()-start,**balance)
            trend=evaluate_temperature(history+[row],args.steady_rate,args.steady_hold)
            row.update(trend)
            stop=row['CPU_C']>=95;end=step>=target_steps;settled=trend['temperature_settled']
            if step>=next_save or stop or end or settled:
                E=cp.asnumpy(sim.dh.gpu_arrays['eddyNu'][sim.inner])
                row['thermal_explicit_diagonal_CFL']=stability_bound(sim,U,E)
                if row['thermal_explicit_diagonal_CFL']>=.9:raise RuntimeError('Thermal explicit stability margin exhausted')
                if row['max_Mach']>.25:raise RuntimeError('Configured Ma_LBM > 0.25 screening guard reached; not evidence of established divergence')
                if row['minimum_C']<24.99:raise RuntimeError('Temperature undershoot exceeds tolerance')
                if t>.2 and row['energy_residual_fraction']>.05:raise RuntimeError('Major transient energy imbalance')
                row['velocity_change_since_saved']=None if last_U is None else float(np.linalg.norm(U-last_U)/max(np.linalg.norm(U),1e-20))
                last_U=U.copy()
                save_fields(folder/f'frame_{len(list(folder.glob("frame_*.npz"))):04d}.npz',
                    T_half=(T+25).astype('float32'),velocity_half=U,time_s=t)
                next_save=step+save_interval
            history.append(row);save_json(folder/'history.json',history);print(json.dumps(row),flush=True);previous=row
            if args.checkpoint and (step>=next_checkpoint or stop or end or settled):
                write_restart(T);next_checkpoint=step+checkpoint_steps
            if stop or end or settled:
                status=('temperature_limit_reached' if stop else 'CPU_temperature_settled'
                        if settled else 'physical_time_window_complete_not_steady')
                meta.update(status=status,final=row,physical_time_s=t,steps=step,
                    limit_event_time_resolution_s=sample_steps*sim.dt,
                    compute_wallclock_s=time.perf_counter()-start,
                    total_wallclock_s=time.perf_counter()-init_start,
                    observed_compute_seconds_per_physical_second=(time.perf_counter()-start)/((step-start_step)*sim.dt) if step>start_step else None)
                save_fields(folder/'last_fields.npz',T_half=(T+25).astype('float32'),velocity_half=U,
                    density_half=R,materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset,time_s=t)
                save_json(folder/'metadata.json',meta);print('RESULT',json.dumps(meta),flush=True);return
            chunk=min(sample_steps,target_steps-step)
            if args.case!='natural':sim.set_port_velocity_scale(min(1.,(t+chunk*sim.dt)/.5))
            if args.case=='cpu_fan':
                lu=sim.dh.gpu_arrays[sim.hydro.velocity_data_name][sim.inner]
                q=max(0.,-2*float(cp.sum(lu[...,2][disk_gpu]))/4*sim.dx**3/sim.dt)
                ratio=1200/FAN['rated_rpm']*min(1.,(t+chunk*sim.dt)/.5)
                target=FAN['shutoff_Pa']*(ratio**2-(q/FAN['free_flow_m3_s'])**2);dp=.8*dp+.2*target
                sim.put('drive2',np.where(disk,-dp/(meta['fluid']['rho']*.004)*sim.dt**2/sim.dx,0.))
            flow.advance(chunk);step+=chunk
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),last=history[-1] if history else None)
        save_json(folder/'metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['natural','ventilation','cpu_fan'],default='natural')
    p.add_argument('--name',required=True);p.add_argument('--dt',type=float,default=.00008)
    p.add_argument('--power',type=float,default=200.);p.add_argument('--max-time',type=float,default=1.)
    p.add_argument('--sample-interval',type=float,default=.04);p.add_argument('--save-interval',type=float,default=.2)
    p.add_argument('--checkpoint',action='store_true');p.add_argument('--restart')
    p.add_argument('--allow-dt-change',action='store_true')
    p.add_argument('--checkpoint-interval',type=float,default=10.)
    p.add_argument('--steady-rate',type=float,default=.01,help='CPU temperature settling threshold in C/s')
    p.add_argument('--steady-hold',type=float,default=10.,help='Physical seconds continuously below the threshold')
    run(p.parse_args())
