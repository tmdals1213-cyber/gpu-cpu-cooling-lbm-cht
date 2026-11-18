"""Small GPU LBM/CHT startup + closed-box heat conservation, not full-case validation."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/solver'))
import numpy as np
from transient import TransientCHT
from runtime_paths import OUTPUT


def main():
    start=time.perf_counter();shape=(24,24,24);ids=np.zeros(shape,dtype=np.uint8);ids[8:16,8:16,8:16]=1
    k=np.full(shape+(3,),.0263);k[ids==1]=390.
    cap=np.full(shape,1.184*1007.);cap[ids==1]=8960.*385.
    q=np.zeros(shape);q[ids==1]=.5/np.sum(ids==1)
    dx=.001;dt=.000005
    sim=TransientCHT(ids,k,cap,q,dx,dt,1.562e-5,1.184*1007.,0.,flow_dtype='float32')
    sim.advance(1000);T,U,R=sim.arrays()
    supplied=.5*sim.time;stored=float(np.sum(T*cap*dx**3));error=abs(stored-supplied)/supplied
    assert np.isfinite(T).all() and np.isfinite(U).all() and np.isfinite(R).all()
    assert np.min(R[ids==0])>0 and error<1e-5,(error,stored,supplied)
    report=dict(status='passed',test='Small closed-box GPU flow/heat startup and energy conservation',
        cells=int(ids.size),physical_time_s=sim.time,relative_energy_residual=error,
        max_speed_m_s=float(np.linalg.norm(U,axis=-1).max()),wallclock_s=time.perf_counter()-start,
        scope='Not full-domain, timestep, grid or product validation')
    (OUTPUT/'smoke_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
