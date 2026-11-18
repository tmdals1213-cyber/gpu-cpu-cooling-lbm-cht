# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled stationary/mean-flow CHT screening of the revised air cases.

LBM is relaxed with buoyancy from the current temperature. A conservative
steady thermal equation is solved in the updated velocity and relaxed back
into LBM. Outer iterations are NOT physical CPU warm-up time. Forced cases
average disjoint flow samples; resolved u'T' covariance is not retained.
Above-limit mathematical fixed-load equilibria are not operating CPU states.
"""
import argparse,json,time,collections
from pathlib import Path
import numpy as np
from case_setup import ROOT,build,build_buffered,FAN
import cupy as cp
from thermal import ThermalOperator
from checkpoint_io import save_fields,save_json

def run(args):
    folder=ROOT/(args.name or args.case+'_steady');folder.mkdir(exist_ok=True)
    assert not (folder/'history.json').exists(),'Use a new --name; previous histories are preserved'
    start=time.perf_counter()
    if args.buffered:sim,full,offset,meta=build_buffered(args.case,200.,args.dt or .00004)
    else:sim,full,offset,meta=build(args.case,200.,args.dt,reservoir_pressure=args.reservoir_pressure,natural_sgs=args.natural_sgs)
    op=ThermalOperator(sim,sim.case_conductivity)
    from heatpipes import apply_implicit_conductance
    apply_implicit_conductance(op)
    cpu=np.isin(sim.ids,[2,3,4]);fluid=sim.ids==0
    window_count=args.average_blocks or (8 if args.case=='cpu_fan' else 4)
    if window_count<1:raise ValueError('At least one mean-flow block is required')
    T=np.zeros(sim.shape);lastU=None;history=[];windows=collections.deque(maxlen=window_count if sim.sgs_constant else 1)
    x,y,z=np.meshgrid(*[np.arange(n)+.5 for n in sim.shape],indexing='ij',sparse=True)
    disk=((x-120)**2+(y-100)**2<60**2)&(z>=98)&(z<102)&fluid
    disk_gpu=cp.asarray(disk);dp=0.;stable=0;flow_steps=0;previous_candidate=None;fan_seeded=False
    if args.seed:
        saved=np.load(args.seed);T=saved['T_half'].astype('float64')-25
        seed_metadata_path=Path(args.seed).parent/'metadata.json'
        seed_metadata=json.loads(seed_metadata_path.read_text()) if seed_metadata_path.exists() else {}
        if args.case=='cpu_fan' and seed_metadata.get('case')=='cpu_fan':
            seed_record=seed_metadata.get('final',seed_metadata.get('last',{}))
            dp=float(seed_record.get('fan_dp_Pa',0.));fan_seeded=True
        U=saved['instantaneous_velocity_half'] if 'instantaneous_velocity_half' in saved else saved['velocity_half']
        R=saved['density_half'] if 'density_half' in saved else np.ones(sim.shape)
        U=U.copy();U[sim.ids!=0]=0
        sim.put('T',T)
        vel=sim.dh.cpu_arrays[sim.hydro.velocity_data_name];vel.fill(0);vel[sim.inner]=U*sim.dt/sim.dx
        rho=sim.dh.cpu_arrays[sim.hydro.density_data_name];rho.fill(1);rho[sim.inner]=1+(R-1)*(sim.dt/args.seed_dt)**2
        if hasattr(sim,'flow_backend'):
            backend=sim.flow_backend
            fv=backend.dh.cpu_arrays[sim.hydro.velocity_data_name][backend.inner]
            fr=backend.dh.cpu_arrays[sim.hydro.density_data_name][backend.inner]
            fv[:,:32,:320]=U[:,0,None,:,:]*sim.dt/sim.dx
            fv[:,32:232,320:]=U[:,:,None,-1,:]*sim.dt/sim.dx
            fv[backend.ids!=0]=0
            pressure_scale=(sim.dt/args.seed_dt)**2
            fr[:,:32,:320]=1+(R[:,0,None,:]-1)*pressure_scale*np.linspace(0,1,32)[None,:,None]
            fr[:,32:232,320:]=1+(R[:,:,-1,None]-1)*pressure_scale*np.linspace(1,0,32)[None,None,:]
        backend=getattr(sim,'flow_backend',sim)
        backend.dh.to_gpu(sim.hydro.velocity_data_name);backend.dh.to_gpu(sim.hydro.density_data_name)
        sim.hydro.set_pdf_fields_from_macroscopic_values();cp.cuda.get_current_stream().synchronize()
        meta.update(seed_from=str(args.seed),seed_dt_s=args.seed_dt,
            seed_method='Macroscopic temperature, velocity and pressure retained; equilibrium PDF reinitialisation',
            seed_limitation='Not exact non-equilibrium kinetic-state continuity',
            fan_actuator_restarted_at_full_rpm=fan_seeded,seed_fan_dp_Pa=dp if fan_seeded else None)
    meta.update(status='running',method='Coupled steady temperature / relaxed buoyant LBM',
        time_axis='Numerical outer iteration, not physical heating time',
        forced_flow_average=f'{window_count} rolling disjoint blocks; block duration recorded per iteration. Explicit resolved velocity-temperature covariance omitted',
        convergence_criteria=(dict(candidate_CPU_range_last_five_K=.5,relaxed_CPU_update_K=.5,
            minimum_flow_relaxation_s=2.,mean_velocity_change=.1,
            energy_residual_fraction=.01,volume_imbalance_fraction=.02,consecutive_passes=3,
            scope='Practical stationary mean-flow screening; physical heating convergence is assessed separately')
            if args.pr_screening else dict(candidate_max_update_K=.2,mean_velocity_change=.03,
            energy_residual_fraction=.005,volume_imbalance_fraction=.01,consecutive_passes=5)),
        physical_temperature_limit_C=95.,over_limit_meaning='Mathematical fixed-load equilibrium only; not actual operation beyond the specified limit')
    save_json(folder/'metadata.json',meta)
    for it in range(args.max_iter):
        control=folder/'control.json'
        controls=json.loads(control.read_text()) if control.exists() else {}
        if 'schedule' in controls:
            eligible=[stage for stage in controls['schedule'] if it>=int(stage['activate_after_iteration'])]
            controls=max(eligible,key=lambda stage:int(stage['activate_after_iteration'])) if eligible else {}
        elif it<int(controls.get('activate_after_iteration',0)):controls={}
        sample_steps=int(controls.get('sample_steps',args.sample_steps));samples=int(controls.get('samples',5))
        max_step=float(controls.get('max_temperature_step',15. if args.pr_screening else 8.));relaxation=float(controls.get('relaxation',.7 if args.pr_screening else .5))
        sumU=np.zeros(sim.shape+(3,),dtype='float32');sumE=np.zeros(sim.shape,dtype='float32')
        for sample in range(samples):
            ramp=1. if fan_seeded else min(1.,(flow_steps+sample_steps)*sim.dt/.5)
            # A seed from the already ventilated case retains its case intake;
            # only a new CPU-fan actuator is ramped up from zero.
            if args.case!='natural':sim.set_port_velocity_scale(1. if args.seed else ramp)
            if args.case=='cpu_fan':
                lu=sim.dh.gpu_arrays[sim.hydro.velocity_data_name][sim.inner]
                qfan=max(0.,-2*float(cp.sum(lu[...,2][disk_gpu]))/4*sim.dx**3/sim.dt)
                ratio=1200/FAN['rated_rpm']*ramp
                target=FAN['shutoff_Pa']*(ratio**2-(qfan/FAN['free_flow_m3_s'])**2)
                dp=.8*dp+.2*target
                sim.put('drive2',np.where(disk,-dp/(meta['fluid']['rho']*.004)*sim.dt**2/sim.dx,0.))
            sim.advance_flow(sample_steps);flow_steps+=sample_steps
            _,U,R=sim.arrays()
            if not np.isfinite(R[fluid]).all() or np.min(R[fluid])<=0:
                raise RuntimeError('Nonfinite or nonpositive fluid density')
            sumU+=U/samples
            if sim.sgs_constant:
                E=cp.asnumpy(sim.dh.gpu_arrays['eddyNu'][sim.inner]);E[~fluid]=0;sumE+=E/samples
        windows.append((sumU,sumE,samples*sample_steps*sim.dt))
        meanU=np.mean([v[0] for v in windows],axis=0);meanE=np.mean([v[1] for v in windows],axis=0)
        velocity_change=None if lastU is None else float(np.linalg.norm(meanU-lastU)/max(np.linalg.norm(meanU),1e-20))
        lastU=meanU.copy()
        op.update(cp.asarray(meanU,dtype='float64'),cp.asarray(meanE,dtype='float64') if sim.sgs_constant else None)
        candidate,inner=op.solve(T,rtol=1e-7);candidate=cp.asnumpy(candidate)
        delta=candidate-T;change=float(np.abs(delta).max());factor=min(relaxation,max_step/max(change,1e-20))
        T+=factor*delta;sim.put('T',T)
        balance=op.balances(cp.asarray(T));cbalance=op.balances(cp.asarray(candidate))
        temps=dict(CPU_C=float(T[cpu].max()+25),IHS_C=float(T[sim.ids==5].max()+25),
            base_C=float(T[sim.ids==8].max()+25),fins_C=float(T[sim.ids==7].max()+25))
        row=dict(iteration=it+1,**temps,candidate_CPU_C=float(candidate[cpu].max()+25),
            candidate_IHS_C=float(candidate[sim.ids==5].max()+25),
            candidate_base_C=float(candidate[sim.ids==8].max()+25),
            candidate_fins_C=float(candidate[sim.ids==7].max()+25),
            candidate_air_peak_C=float(candidate[fluid].max()+25),
            candidate_air_mean_C=float(candidate[fluid].mean()+25),
            candidate_energy_residual_fraction=abs(2*cbalance['boundary_heat_W']/200-1),
            averaging_window_s=sum(v[2] for v in windows),
            block_duration_s=samples*sample_steps*sim.dt,
            temperature_candidate_change_K=change,relaxation=factor,mean_velocity_change=velocity_change,
            heat_out_W=2*balance['boundary_heat_W'],energy_residual_fraction=abs(2*balance['boundary_heat_W']/200-1),
            volume_imbalance_fraction=balance['volume_imbalance'],
            volume_in_CFM=2*balance['volume_in_m3_s']/.00047194745,volume_out_CFM=2*balance['volume_out_m3_s']/.00047194745,
            max_speed_m_s=float(np.linalg.norm(U,axis=-1).max()),max_Mach=float(np.linalg.norm(U,axis=-1).max()*sim.dt/sim.dx*np.sqrt(3)),
            density_min=float(R[fluid].min()),density_max=float(R[fluid].max()),
            flow_relaxation_s=flow_steps*sim.dt,flow_steps=flow_steps,thermal_solver=inner,wallclock_s=time.perf_counter()-start)
        if args.case=='cpu_fan':row.update(fan_dp_Pa=dp,fan_flow_CFM=-2*float(meanU[...,2][disk].sum())/4*sim.dx**2/.00047194745)
        criteria=meta['convergence_criteria']
        candidates=[h['candidate_CPU_C'] for h in history[-4:]]+[row['candidate_CPU_C']]
        row['candidate_CPU_range_last_five_K']=max(candidates)-min(candidates)
        row['relaxed_CPU_update_K']=abs(row['candidate_CPU_C']-row['CPU_C'])
        temperature_ok=(len(candidates)==5 and row['candidate_CPU_range_last_five_K']<criteria['candidate_CPU_range_last_five_K']
            and row['relaxed_CPU_update_K']<criteria['relaxed_CPU_update_K']
            and flow_steps*sim.dt>=criteria['minimum_flow_relaxation_s']) if args.pr_screening else change<criteria['candidate_max_update_K']
        good=(it>=12 and ramp==1. and temperature_ok
            and velocity_change is not None and velocity_change<criteria['mean_velocity_change']
            and row['energy_residual_fraction']<criteria['energy_residual_fraction']
            and balance['volume_imbalance']<criteria['volume_imbalance_fraction'])
        stable=stable+1 if good else 0;row['consecutive_passes']=stable;history.append(row)
        save_json(folder/'history.json',history);print(json.dumps(row),flush=True)
        if True:
            save_fields(folder/'latest_fields.npz',T_half=(T+25).astype('float32'),velocity_half=meanU,
                instantaneous_velocity_half=U,eddy_nu_half=meanE,density_half=R,materials=full,materials_half=sim.ids,
                dx=sim.dx,offset_mm=offset,iteration=it+1)
        if stable>=criteria['consecutive_passes']:
            # Final algebraic thermal solution eliminates under-relaxation
            # energy residual; the preceding coupled update must already pass.
            T=candidate;sim.put('T',T);balance=cbalance
            meta.update(status='coupled_mean_flow_PR_screening' if args.pr_screening else 'coupled_mean_flow_screening_converged' if sim.sgs_constant else 'coupled_steady_converged',
                CPU_C=float(T[cpu].max()+25),IHS_C=float(T[sim.ids==5].max()+25),
                base_C=float(T[sim.ids==8].max()+25),fins_C=float(T[sim.ids==7].max()+25),
                heat_out_W=2*balance['boundary_heat_W'],energy_residual_fraction=abs(2*balance['boundary_heat_W']/200-1),
                volume_imbalance_fraction=balance['volume_imbalance'],final=row,wallclock_s=time.perf_counter()-start,
                limit_exceeded=bool(T[cpu].max()+25>95))
            save_fields(folder/'steady_fields.npz',T_half=(T+25).astype('float32'),velocity_half=meanU,
                eddy_nu_half=meanE,materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
            save_json(folder/'metadata.json',meta);print('RESULT',json.dumps(meta),flush=True);return
    meta.update(status='not_converged',last=history[-1]);save_json(folder/'metadata.json',meta)
    raise RuntimeError('Coupled steady convergence criteria not met')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['natural','ventilation','cpu_fan'],required=True)
    p.add_argument('--dt',type=float);p.add_argument('--max-iter',type=int,default=100)
    p.add_argument('--seed');p.add_argument('--seed-dt',type=float,default=.0001)
    p.add_argument('--reservoir-pressure',action='store_true');p.add_argument('--natural-sgs',action='store_true')
    p.add_argument('--buffered',action='store_true');p.add_argument('--name');p.add_argument('--pr-screening',action='store_true')
    p.add_argument('--average-blocks',type=int,help='Default 8 for CPU-fan, 4 for other forced cases; block durations and resulting window are recorded per iteration')
    p.add_argument('--sample-steps',type=int,default=1000);run(p.parse_args())
