# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-way case-air heating driven by the calculated water/package history.

Both air and water use their separately computed fixed CFD flow fields. Air
heat extraction is measured, but is not fed back into the fine water solution.
"""
import argparse,json,time
import numpy as np
from scipy.ndimage import map_coordinates
from case_setup import ROOT
from fixed_flow_heat import thermal_model
from water_case_air import PrescribedBodyOperator
from temperature_series import TemperatureSeries
from checkpoint_io import save_json,save_fields
import cupy as cp


def run(args):
    folder=ROOT/args.name;folder.mkdir(exist_ok=True)
    assert not (folder/'history.json').exists(),'Use a new output name'
    air_meta=json.loads((ROOT/'water_case_air_steady/metadata.json').read_text())
    assert air_meta['status']=='one_way_case_air_screening_converged'
    water_meta=json.loads((ROOT/'cold_plate_heat/metadata.json').read_text())
    assert water_meta['status']=='CPU_temperature_settled'
    series=TemperatureSeries([ROOT/'cold_plate_heat'])
    end=series.times[-1]
    sim,full,offset,meta,_=thermal_model('water_case_air',air_meta['dt_s'])
    fixed=sim.ids!=0;sim.source_w.fill(0);sim.capacity[fixed]=0.
    coords=np.argwhere(fixed).astype(float)+.5
    fine_coords=((coords-np.array([88.,60.,20.]))/.25-.5).T
    def body_temperature(t):
        value=np.zeros(sim.shape)
        value[fixed]=map_coordinates(series.sample(t),fine_coords,order=1,mode='nearest')-25.
        return value
    flow=np.load(ROOT/'water_case_air_steady/steady_fields.npz');U=flow['velocity_half'];E=flow['eddy_nu_half']
    op=PrescribedBodyOperator(sim,sim.case_conductivity,body_temperature(0.))
    op.update(cp.asarray(U,dtype='float64'),cp.asarray(E,dtype='float64'))
    def update_body(t):
        op.Tfixed=cp.asarray(body_temperature(t))
        rhs=cp.zeros(sim.shape,dtype='float64')
        for a in range(3):
            left=[slice(None)]*3;right=left.copy();left[a]=slice(None,-1);right[a]=slice(1,None)
            left,right=tuple(left),tuple(right);g=op.G[a][left]
            lf=op.fixed[left];rf=op.fixed[right]
            rhs[left]+=cp.where(~lf&rf,g*op.Tfixed[right],0.)
            rhs[right]+=cp.where(lf&~rf,g*op.Tfixed[left],0.)
        op.rhs=rhs.ravel();op.normb=float(cp.linalg.norm(op.rhs))
    meta.update(status='running',power_W=200.,air_volumetric_source_W=0.,cooler=air_meta['cooler'],
        method='Physical case-air heat transient in fixed CFD mean flow, with time-varying prescribed package/block temperature',
        thermal_coupling='One-way from calculated fine water/solid history; air heat extraction is not fed back into the water model',
        thermal_dt_s=args.dt,initial_air_C=25.,water_initial_C=30.,
        water_source='cold_plate_heat',air_flow_source='water_case_air_steady/steady_fields.npz')
    save_json(folder/'metadata.json',meta)
    save_fields(folder/'model.npz',materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset)
    save_fields(folder/'flow_fields.npz',velocity_half=U,eddy_nu_half=E)
    unknown=cp.zeros(sim.shape,dtype='float64');capacity=cp.asarray(sim.capacity)
    t=0.;incoming=outgoing=0.;history=[];start=time.perf_counter()
    try:
        while True:
            balance=op.balances(unknown);heat=op.body_heat(unknown);out=2*balance['boundary_heat_W']
            stored=2*float(cp.sum(capacity*unknown));residual=incoming-outgoing-stored
            T=cp.asnumpy(unknown)+25.;T[fixed]=cp.asnumpy(op.Tfixed)[fixed]+25.
            row=dict(time_s=t,air_peak_C=float(T[~fixed].max()),body_to_air_W=heat,heat_out_W=out,
                air_stored_J=stored,cumulative_body_heat_J=incoming,cumulative_heat_out_J=outgoing,
                air_energy_residual_J=residual,air_energy_residual_fraction=abs(residual)/max(abs(incoming),1.),
                wallclock_s=time.perf_counter()-start)
            if not np.isfinite(T).all() or T.min()<24.99:raise RuntimeError('Invalid air-temperature field')
            if t>1 and row['air_energy_residual_fraction']>.01:raise RuntimeError('Air transient energy imbalance')
            save_fields(folder/f'frame_{len(history):04d}.npz',T_half=T.astype('float32'),time_s=t)
            history.append(row);save_json(folder/'history.json',history)
            if t%5<1e-8 or t>=end:print(json.dumps(row),flush=True)
            if t>=end-1e-8:
                save_fields(folder/'last_fields.npz',T_half=T.astype('float32'),velocity_half=U,
                    materials=full,materials_half=sim.ids,dx=sim.dx,offset_mm=offset,time_s=t)
                meta.update(status='one_way_air_history_complete',physical_time_s=t,final=row)
                save_json(folder/'metadata.json',meta);return
            dt=min(args.dt,end-t);update_body(t+dt)
            unknown,info=op.step_implicit(unknown,dt);t=round(t+dt,10)
            incoming+=op.body_heat(unknown)*dt;outgoing+=2*op.balances(unknown)['boundary_heat_W']*dt
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),last=history[-1] if history else None)
        save_json(folder/'metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='water_case_air_transient');p.add_argument('--dt',type=float,default=.5);run(p.parse_args())
