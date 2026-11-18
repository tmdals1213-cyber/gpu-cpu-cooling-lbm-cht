# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Separate local cold-plate CHT domain: coolant inside copper, electronics dry.

The radiator/loop is represented by a prescribed 30 C supply, 1 L/min.
External air around the package conducts heat but is not a water-flow region.
"""
import argparse,json,time,copy
import numpy as np
from case_setup import ROOT,CPU,MATERIALS,CONTACT,ALL_FACES
from transient import package_geometry,TransientCHT
import cupy as cp

DX=.00025
SIZE=(64,80,24)
WATER=dict(name='Water-like coolant proxy',rho=1000.,cp=4200.,k=.6,nu=1e-6,
    reference='Rounded constant-property model; not a specified glycol/additive product')

def geometry():
    ids,_,offset=package_geometry(DX*1000,SIZE,4.)
    ids[ids==7]=0;ids[ids==0]=6
    def set_box(bounds,mat):
        sl=tuple(slice(round(lo/(DX*1000)),round(hi/(DX*1000))) for lo,hi in bounds);ids[sl]=mat
    set_box([(8,56),(0,80),(10,22)],8)
    set_box([(10,54),(0,80),(12,20)],0)
    fins=list(range(11,32,2));fins+=[63-x for x in fins]
    for x in fins:set_box([(x,x+1),(20,60),(12,20)],7)
    assert np.all(ids[ids==2]==2) and np.sum(ids==4)>0
    mirrored=ids[::-1].copy();a=mirrored==2;b=mirrored==3;mirrored[a]=3;mirrored[b]=2
    assert np.array_equal(ids,mirrored)
    # Water appears exclusively inside the closed block/channel network.
    wet=np.argwhere(ids==0)*DX*1000+DX*500
    assert np.all((wet[:,0]>=10)&(wet[:,0]<54)&(wet[:,2]>=12)&(wet[:,2]<20))
    return ids,offset

def configuration():
    shape=np.array(SIZE)/(DX*1000)
    return dict(case='cold_plate',status='setup_only_no_new_temperature_prediction',cpu=CPU,
        local_domain_mm=list(SIZE),dx_mm=DX*1000,computed_half_cells=int(np.prod(shape)//2),
        power_W=200.,coolant=WATER,supply_C=30.,flow_L_min=1.,ambient_C=25.,
        target_CPU_C=85.,temperature_limit_C=95.,
        cold_plate=dict(material='Copper',base_mm=2.,channel_clear_width_mm=1.,channel_height_mm=8.,
            fin_thickness_mm=1.,active_channel_length_mm=40.,geometry='Simplified parallel-channel block with inlet/outlet plenums'),
        wet_contact='Coolant contacts internal metal channel walls only; package surroundings are dry air',
        loop='Pump/radiator outside local CFD domain, represented by fixed inlet temperature and flow',
        external_air='Conduction-only local surrounding air approximation; no full-case air CFD in this case',
        grid_converged=False,temperature_accuracy_validated=False,
        contacts_m2K_W={str(k):v for k,v in CONTACT.items()})

def build(power=200.):
    full,offset=geometry();ids=full[:full.shape[0]//2].copy()
    k=np.zeros(ids.shape+(3,));cap=np.zeros(ids.shape);q=np.zeros(ids.shape)
    props=copy.deepcopy(MATERIALS);props[7]=props[8]
    for m,(rho,cp0,cond) in props.items():
        if m==0:rho,cp0,cond=WATER['rho'],WATER['cp'],[WATER['k']]*3
        mask=ids==m;k[mask]=cond;cap[mask]=rho*cp0
    q[ids==2]=power*.45/np.sum(ids==2);q[ids==4]=power*.05/np.sum(ids==4)
    assert abs(2*q.sum()-power)<1e-8 and np.all(cap>0)
    ranges=[(10/.25,54/.25),(0,80/.25),(12/.25,20/.25)]
    ports=[dict(axis=1,side=0,ranges=ranges,velocity_m_s=[0,1/60000/(.044*.008),0],ramp_velocity=True),
           dict(axis=1,side=1,ranges=ranges)]
    walls={f:-5. for f in ALL_FACES if f!=(0,1)}
    sim=TransientCHT(ids,k,cap,q,DX,.00002,WATER['nu'],WATER['rho']*WATER['cp'],0.,
        contacts=CONTACT,wall_temperatures=walls,wall_h={f:5. for f in walls},
        symmetry_faces=((0,1),),flow_dtype='float32',ports=ports,fluid_density=WATER['rho'])
    sim.case_conductivity=k
    return sim,full,offset

def solve():
    from thermal import ThermalOperator
    folder=ROOT/'cold_plate_run';folder.mkdir(exist_ok=True)
    start=time.perf_counter();sim,full,offset=build();hist=[];previous=None;stable=0
    for it in range(200):
        sim.set_port_velocity_scale(min(1.,(it+1)/10))
        sim.advance_flow(1000);T,U,R=sim.arrays()
        qin=2*float(U[...,1][sim.port_masks[0]].sum())*DX**2*60000
        qout=2*float(U[...,1][sim.port_masks[1]].sum())*DX**2*60000
        change=None if previous is None else float(np.linalg.norm(U-previous)/max(np.linalg.norm(U),1e-15))
        row=dict(iteration=it+1,flow_relaxation_s=(it+1)*1000*sim.dt,inlet_L_min=qin,outlet_L_min=qout,
            velocity_change=change,max_speed_m_s=float(np.linalg.norm(U,axis=-1).max()),wallclock_s=time.perf_counter()-start)
        hist.append(row);previous=U.copy()
        stable=stable+1 if it>=20 and change<.002 and abs(qout/qin-1)<.01 else 0
        (folder/'flow_history.json').write_text(json.dumps(hist,indent=2))
        if (it+1)%5==0:print('FLOW',json.dumps(row),flush=True)
        if stable>=5:break
    if stable<5:raise RuntimeError('Cold-plate laminar flow did not meet convergence checks')
    np.savez_compressed(folder/'flow_fields.npz',velocity_half=U.astype('float32'),density_half=R.astype('float32'))
    op=ThermalOperator(sim,sim.case_conductivity);op.update(cp.asarray(U,dtype='float64'))
    steady,info=op.solve(np.zeros(sim.shape),rtol=1e-8);bal=op.balances(steady)
    ta=cp.asnumpy(steady)+30;mask=np.isin(sim.ids,[2,3,4]);out=sim.port_masks[1]
    weights=np.maximum(U[...,1][out],0)
    outlet=float(np.sum(weights*ta[out])/np.sum(weights))
    result=configuration();result.update(status='steady_fixed_property_single_phase',
        CPU_C=float(ta[mask].max()),IHS_C=float(ta[sim.ids==5].max()),cold_plate_C=float(ta[np.isin(sim.ids,[7,8])].max()),
        coolant_outlet_bulk_C=outlet,heat_out_W=2*bal['boundary_heat_W'],energy_residual_fraction=abs(2*bal['boundary_heat_W']/200-1),
        flow=hist[-1],flow_steps=len(hist)*1000,thermal_solver=info,thermal_balance_half=bal,
        thermal_balance_scope='Computed half domain; all temperatures are absolute C',
        flow_heat_decoupling='Exact for this constant-property forced-flow model with buoyancy disabled; no turbulent mean-flow approximation',
        limitation='Simplified water block/package, conduction-only surrounding air, prescribed supply, no radiator CFD, no grid convergence or product-temperature validation',
        wallclock_s=time.perf_counter()-start)
    assert result['energy_residual_fraction']<1e-5 and bal['volume_imbalance']<.01
    np.savez_compressed(folder/'steady_fields.npz',T_half=ta.astype('float32'),velocity_half=U.astype('float32'),
        materials=full,materials_half=sim.ids,dx=DX,offset_mm=offset)
    (folder/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)

def run(args):
    conf=configuration();full,offset=geometry()
    conf['material_cell_counts']={str(m):int(np.sum(full==m)) for m in np.unique(full)}
    conf['water_containment_geometry_check']=True
    (ROOT/'cold_plate_setup.json').write_text(json.dumps(conf,indent=2))
    if not args.startup_check:return
    start=time.perf_counter();sim,full,offset=build()
    sim.set_port_velocity_scale(.1);sim.advance(1000);T,U,R=sim.arrays()
    result=dict(status='startup_check_only',physical_time_s=sim.time,
        CPU_C=float(T[np.isin(sim.ids,[2,3,4])].max()+30),max_velocity_m_s=float(np.linalg.norm(U,axis=-1).max()),
        finite_fields=True,wallclock_s=time.perf_counter()-start,
        note='Early startup at 10% inlet flow; not a converged temperature or full-flow validation')
    (ROOT/'cold_plate_startup_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--startup-check',action='store_true');p.add_argument('--solve',action='store_true')
    args=p.parse_args()
    if args.solve:solve()
    else:run(args)
