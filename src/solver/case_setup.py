# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Revision C: six finite-conductance heatpipes in the existing air cooler.

The 200 W source, package, case and fan settings stay unchanged. This builder
provides copper-level pipe/interface conductance. The three active thermal
drivers attach the extra internal pipe conductance from heatpipes.py.
"""
from pathlib import Path
import sys,json,copy
import numpy as np
from types import SimpleNamespace
from runtime_paths import OUTPUT as ROOT
from model import AIR,OIL,CPU,FAN,PROPERTIES,CONTACTS,ALL_FACES
from transient import package_geometry,TransientCHT
from heatpipes import add_geometry, specification

DOMAIN=(240,200,320)
DX=.001
MATERIALS=copy.deepcopy(PROPERTIES)
MATERIALS[8]=(8960.,385.,[390.]*3)  # copper cooler base
MATERIALS[9]=(1200.,1400.,[.2]*3)  # fan frame
MATERIALS[10]=(4500.,400.,[390.]*3)  # heatpipe shell/interface; internal G is enhanced separately
CONTACT=copy.deepcopy(CONTACTS)
CONTACT.pop((5,7));CONTACT[(5,8)]=2e-5
CONTACT[(5,10)]=2e-5

def ports(case):
    if case=='immersion':
        return [dict(axis=1,side=0,ranges=[(60,180),(0,200),(20,60)],velocity_m_s=[0,5/60000/(.12*.04),0],ramp_velocity=True),
                dict(axis=1,side=1,ranges=[(60,180),(0,200),(260,300)])]
    low=dict(axis=1,side=0,ranges=[(40,200),(0,200),(20,140)])
    high=dict(axis=2,side=1,ranges=[(40,200),(40,160),(0,320)])
    if case!='natural':
        low.update(velocity_m_s=[0,40*.00047194745/(.16*.12),0],ramp_velocity=True,sine_rect=True)
    # Equal reference/reduced pressure, with Boussinesq buoyancy in natural
    # convection. No prescribed natural-convection flow rate or fan pressure.
    return [low,high]

def geometry(case):
    ids,spec,offset=package_geometry(1.,DOMAIN,24.)
    ids[ids==7]=0
    if case=='water_case_air':
        # Air sees a sealed water-block envelope. Internal channels are solved
        # separately on the existing 0.25 mm water grid.
        ids[96:144,60:140,30:42]=8
        return ids,offset
    ids[60:180,40:160,30:36]=8
    fins=list(range(62,120,5));fins+= [239-x for x in fins]
    for x in fins:ids[x:x+1,40:160,36:90]=7
    add_geometry(ids)
    if case=='cpu_fan':
        ids[58:60,38:162,90:104]=9;ids[180:182,38:162,90:104]=9
        ids[60:180,38:40,90:104]=9;ids[60:180,160:162,90:104]=9
    mirror=ids[::-1].copy();a=mirror==2;b=mirror==3;mirror[a]=3;mirror[b]=2
    assert np.array_equal(ids,mirror)
    return ids,offset

def fields(ids,power,fluid):
    k=np.zeros(ids.shape+(3,));cap=np.zeros(ids.shape);q=np.zeros(ids.shape)
    for m,(rho,cp,conductivity) in MATERIALS.items():
        if m==0:rho,cp,conductivity=fluid['rho'],fluid['cp'],[fluid['k']]*3
        mask=ids==m;k[mask]=conductivity;cap[mask]=rho*cp
    for m,share in [(2,.45),(3,.45),(4,.1)]:q[ids==m]=power*share/np.sum(ids==m)
    assert np.all(cap>0) and abs(q.sum()-power)<1e-8
    return k,cap,q

def configuration(case,power=200.):
    fluid=OIL if case=='immersion' else AIR
    return dict(revision='C',case=case,status='setup_only_no_new_temperature_prediction',
        cpu=CPU,domain_mm=list(DOMAIN),domain_L=float(np.prod(DOMAIN)/1e6),
        dx_mm=1.,computed_half_cells=int(np.prod(DOMAIN)//2),power_W=power,
        source_W={'CCD1':.45*power,'CCD2':.45*power,'IOD':.1*power},
        ambient_C=25.,initial_C=30. if case=='immersion' else 25.,
        supply_C=30. if case=='immersion' else 25.,fluid=fluid,
        temperature_reference_C=30. if case=='immersion' else 25.,
        sustained_load_design_target_C=85.,near_limit_C=90.,temperature_limit_C=95.,
        stop_meaning='Temperature-limit event for a fixed heat load; NOT hardware shutdown',
        power_control='Prescribed thermal load, no emulation of vendor boost/throttling',
        ports=ports(case),passive_pressure='Equal reduced/gauge reference pressure at both openings; buoyancy determines flow' if case=='natural' else None,
        case_flow_CFM=40. if case in ['ventilation','cpu_fan'] else None,
        case_fan_model='Prescribed inlet through-flow + reference-pressure outlet; enclosure-fan operating point not resolved' if case in ['ventilation','cpu_fan'] else None,
        cpu_fan_rpm=1200. if case=='cpu_fan' else None,
        fan=FAN if case=='cpu_fan' else None,
        liquid_flow_L_min=5. if case=='immersion' else None,
        external_wall_h_W_m2K=5.,wall_resistance=0.,wall_heat_capacity=0.,
        cooler=dict(size_mm=[120,120,60],base='6 mm copper',fin_material='Aluminium',
            fin_count=24,fin_thickness_mm=1,fin_height_mm=54,clear_gap_mm=4,
            copper_aluminium_interface='Perfect bonded contact assumption',heatpipes=False if case=='water_case_air' else specification(),
            description='Manufacturable simplified bonded-fin cooler; not a calibrated commercial product'),
        contacts_m2K_W={str(a):b for a,b in CONTACT.items()},
        immersion_render='Fully submerged dielectric bath; visible fill level above hardware, supply and return hoses; free-surface dynamics not solved' if case=='immersion' else None,
        grid_converged=False,temperature_accuracy_validated=False,
        next_validation=['Zero-heat pressure-opening rest state','Low/high opening buoyancy and mass balance',
            'Fin-channel and fan through-flow','Thermal and grid convergence before temperature claims'])

def build(case,power=200.,dt=None,reservoir_pressure=False,natural_sgs=False):
    full,offset=geometry(case);conf=configuration(case,power)
    if reservoir_pressure:
        for port in conf['ports']:
            if 'velocity_m_s' not in port:port['equilibrium_pressure']=True
        conf['pressure_implementation']='Equilibrium incoming populations at the prescribed-density reservoir boundary; velocity is extrapolated, not imposed'
    # Avoid allocating full-domain coefficient arrays; full heat-source counts
    # are reconstructed exactly by assigning half of total load to the half mesh.
    ids=full[:DOMAIN[0]//2].copy();fluid=conf['fluid']
    k=np.zeros(ids.shape+(3,));cap=np.zeros(ids.shape);q=np.zeros(ids.shape)
    for m,(rho,cp,conductivity) in MATERIALS.items():
        if m==0:rho,cp,conductivity=fluid['rho'],fluid['cp'],[fluid['k']]*3
        mask=ids==m;k[mask]=conductivity;cap[mask]=rho*cp
    # The left half contains CCD1 and half of IOD; CCD2 is mirrored.
    q[ids==2]=power*.45/np.sum(ids==2)
    q[ids==4]=power*.05/np.sum(ids==4)
    assert abs(2*q.sum()-power)<1e-8
    reference=conf['temperature_reference_C'];walls={f:25.-reference for f in ALL_FACES if f!=(0,1)}
    timestep=dt or (.0001 if case=='natural' else (.0005 if case=='immersion' else .00001))
    sim=TransientCHT(ids,k,cap,q,DX,timestep,fluid['nu'],fluid['rho']*fluid['cp'],9.81*fluid['beta'],
        contacts=CONTACT,wall_temperatures=walls,wall_h={f:5. for f in walls},
        symmetry_faces=((0,1),),flow_dtype='float32',ports=conf['ports'],fluid_density=fluid['rho'],
        external_drive=case=='cpu_fan',sgs_constant=.12 if case in ['ventilation','cpu_fan'] or natural_sgs else None)
    conf['dt_s']=timestep
    conf['sgs_constant']=sim.sgs_constant
    sim.case_conductivity=k
    return sim,full,offset,conf

def write_setups():
    ROOT.mkdir(parents=True,exist_ok=True);checks=[]
    for case in ['natural','ventilation','cpu_fan']:
        conf=configuration(case);ids,offset=geometry(case)
        qcounts={str(m):int(np.sum(ids==m)) for m in np.unique(ids)}
        conf['material_cell_counts']=qcounts;conf['package_offset_mm']=offset.tolist()
        for port in conf['ports']:
            a=port['axis'];face=0 if port['side']==0 else -1
            sl=[slice(int(lo),int(hi)) for lo,hi in port['ranges']];sl[a]=face
            assert np.all(ids[tuple(sl)]==0),(case,'blocked port')
        (ROOT/f'{case}_setup.json').write_text(json.dumps(conf,indent=2))
        checks.append(dict(case=case,symmetric=True,ports_clear=True,total_heat_W=200.,
            fluid_cells=int(np.sum(ids==0)),all_solid_parts_present=True))
    area_old=2*(.160*.144+.160*.192+.144*.192)
    area_new=2*(.240*.200+.240*.320+.200*.320)
    sanity=dict(old_volume_L=.160*.144*.192*1000,new_volume_L=15.36,
        old_closed_wall_UA_W_K=5*area_old,new_closed_wall_UA_W_K=5*area_new,
        old_uniform_wall_rise_to_reject_200W_K=200/(5*area_old),
        note='Ideal uniform wall temperature, prescribed h=5, no radiation. This is a resistance sanity check, not a predicted CPU temperature.',
        air_bulk_rise_at_40CFM_200W_K=200/(40*.00047194745*AIR['rho']*AIR['cp']),
        immersion_bulk_rise_at_5Lpm_200W_K=200/(5/60000*OIL['rho']*OIL['cp']),
        target_CPU_to_air_resistance_K_W=(85-25)/200,
        target_CPU_to_liquid_resistance_K_W=(85-30)/200)
    (ROOT/'setup_checks.json').write_text(json.dumps(dict(geometry_checks=checks,energy_sanity=sanity),indent=2))
    print(json.dumps(sanity,indent=2),flush=True)

def build_buffered(case,power=200.,dt=.00004):
    """Move pressure planes outside the physical case using two flow-only ducts.

    Thermal assembly remains strictly 240 x 200 x 320 mm. Exterior fluid has
    zero thermal buoyancy and a viscous sponge only in its outermost 16 mm.
    Solid duct frames keep the physical aperture and case no-slip wall intact.
    """
    full,offset=geometry(case);meta=configuration(case,power)
    ids=full[:120].copy();k=np.zeros(ids.shape+(3,));cap=np.zeros(ids.shape);q=np.zeros(ids.shape)
    for m,(rho,cp0,conductivity) in MATERIALS.items():
        if m==0:rho,cp0,conductivity=AIR['rho'],AIR['cp'],[AIR['k']]*3
        mask=ids==m;k[mask]=conductivity;cap[mask]=rho*cp0
    q[ids==2]=power*.45/np.sum(ids==2);q[ids==4]=power*.05/np.sum(ids==4)
    pad=32;shape=(120,200+pad,320+pad);crop=(slice(None),slice(pad,pad+200),slice(0,320))
    flow_ids=np.full(shape,9,dtype='uint8');flow_ids[crop]=ids
    flow_ids[40:120,:pad,20:140]=0
    flow_ids[40:120,40+pad:160+pad,320:]=0
    fk=np.full(shape+(3,),AIR['k']);fc=np.full(shape,AIR['rho']*AIR['cp']);fq=np.zeros(shape)
    fk[crop]=k;fc[crop]=cap;fq[crop]=q
    flow_ports=copy.deepcopy(meta['ports'])
    for port in flow_ports:
        port['ranges'][1]=[v+pad for v in port['ranges'][1]]
    flow_ports[0]['ranges'][1]=[0,shape[1]]
    flow_ports[1]['ranges'][2]=[0,shape[2]]
    yy=np.arange(shape[1])[None,:,None]+.5;zz=np.arange(shape[2])[None,None,:]+.5
    sponge=np.maximum(np.clip((16-yy)/16,0,1)**2,np.clip((zz-336)/16,0,1)**2)
    visc=np.broadcast_to(AIR['nu']+(.0005-AIR['nu'])*sponge,shape)
    flow=TransientCHT(flow_ids,fk,fc,fq,DX,dt,AIR['nu'],AIR['rho']*AIR['cp'],9.81*AIR['beta'],
        contacts=CONTACT,symmetry_faces=((0,1),),flow_dtype='float32',ports=flow_ports,
        fluid_density=AIR['rho'],external_drive=case=='cpu_fan',sgs_constant=.12,viscosity_field=visc)
    # Physical boundary heat loss and physical port masks, independent of the
    # outer hydrodynamic pressure-plane location.
    wallG=np.zeros(ids.shape);masks=[]
    for axis,side in ALL_FACES:
        if (axis,side)==(0,1):continue
        sl=[slice(None)]*3;sl[axis]=0 if side==0 else -1;sl=tuple(sl)
        wallG[sl]+=DX**2/(DX/(2*k[sl+(axis,)])+1/5.)
    coords=np.meshgrid(*[np.arange(n)+.5 for n in ids.shape],indexing='ij',sparse=True)
    for port in meta['ports']:
        mask=np.ones(ids.shape,dtype=bool)
        for axis,(lo,hi) in enumerate(port['ranges']):mask&=(coords[axis]>=lo)&(coords[axis]<hi)
        face=np.zeros(ids.shape,dtype=bool);sl=[slice(None)]*3;sl[port['axis']]=0 if port['side']==0 else -1
        face[tuple(sl)]=True;mask&=face;masks.append(mask);wallG[mask]=0
    ghost_crop=(slice(None),slice(pad,pad+202),slice(0,322))
    def cropped_arrays(source):return {name:array[ghost_crop] for name,array in source.items()}
    dh=SimpleNamespace(gpu_arrays=cropped_arrays(flow.dh.gpu_arrays),cpu_arrays=cropped_arrays(flow.dh.cpu_arrays))
    def put(name,value):
        padded=np.zeros(shape);padded[crop]=value;flow.put(name,padded)
    def arrays():
        t,u,r=flow.arrays();return t[crop].copy(),u[crop].copy(),r[crop].copy()
    view=SimpleNamespace(ids=ids,shape=ids.shape,dx=DX,dt=dt,inner=(slice(1,-1),)*3,
        capacity=cap*DX**3,source_w=q,case_conductivity=k,wallG=wallG,wallGT=np.zeros(ids.shape),
        ports=meta['ports'],port_masks=masks,sgs_constant=.12,turbulent_prandtl=.9,
        fluid_rhocp=AIR['rho']*AIR['cp'],fluid_density=AIR['rho'],dh=dh,hydro=flow.hydro,
        put=put,arrays=arrays,advance_flow=flow.advance_flow,set_port_velocity_scale=flow.set_port_velocity_scale,
        flow_backend=flow)
    meta.update(dt_s=dt,sgs_constant=.12,flow_domain_mm=[240,232,352],flow_computed_half_cells=int(np.prod(shape)),
        flow_domain_origin_mm=[0,-32,0],
        external_buffer='32 mm inlet and outlet flow-only ducts; sponge in outer 16 mm reaches nu=0.0005 m2/s. Physical case air viscosity unchanged.',
        thermal_domain_mm=list(DOMAIN),buffer_thermal='Outside physical thermal domain, ambient temperature and no buoyancy; not included in heat balance',
        pressure_implementation='Fixed-density pressure at exterior duct end, original physical aperture retained')
    return view,full,offset,meta

if __name__=='__main__':write_setups()
