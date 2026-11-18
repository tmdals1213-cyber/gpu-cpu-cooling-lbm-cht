# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry, conservative local diffusion, and split/implicit consistency checks."""
import json
from types import SimpleNamespace
from case_setup import ROOT,geometry,configuration
from heatpipes import LocalPipeDiffusion,apply_implicit_conductance,PIPE_ID,INTERNAL_K,SHELL_K
import numpy as np
from scipy.ndimage import label
import cupy as cp

full,_=geometry('cpu_fan')
components,count=label(full==PIPE_ID)
assert count==6, count
mirror=full[::-1].copy();a=mirror==2;b=mirror==3;mirror[a]=3;mirror[b]=2
assert np.array_equal(full,mirror)
ids=full[:120].copy();shape=ids.shape
# A transposed/Fortran storage check catches indexing errors in raw GPU access.
T=cp.zeros(tuple(n+2 for n in shape),dtype='float64',order='F')
rng=np.random.default_rng(2026);initial=rng.uniform(0,50,size=shape)
T[1:-1,1:-1,1:-1]=cp.asarray(initial)
flow=SimpleNamespace(ids=ids,shape=shape,dx=.001,dt=.0002,
    dh=SimpleNamespace(gpu_arrays={'T':T}))
local=LocalPipeDiffusion(flow)
mask=ids==PIPE_ID;old=initial[mask].copy()
expected=old.copy();nb=cp.asnumpy(local.neighbours)
for _ in range(local.substeps):
    delta=np.zeros_like(expected)
    for j in range(6):
        valid=nb[:,j]>=0;delta[valid]+=expected[nb[valid,j]]-expected[valid]
    expected+=local.factor*delta
local.step();cp.cuda.get_current_stream().synchronize()
result=cp.asnumpy(T[1:-1,1:-1,1:-1]);actual=result[mask]
error=float(np.max(np.abs(actual-expected)))
energy_relative=float(abs(actual.sum()-old.sum())/abs(old.sum()))
assert error<1e-11 and energy_relative<1e-12
assert actual.min()>=old.min() and actual.max()<=old.max()
assert np.array_equal(result[~mask],initial[~mask])
G=[cp.full(shape,SHELL_K*.001) for _ in range(3)]
op=SimpleNamespace(sim=flow,G=G)
apply_implicit_conductance(op)
face_counts=[]
for axis in range(3):
    left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
    left,right=tuple(left),tuple(right);both=mask[left]&mask[right]
    check=cp.asnumpy(G[axis][left]);assert np.allclose(check[both],INTERNAL_K*.001)
    assert np.allclose(check[~both],SHELL_K*.001)
    face_counts.append(int(both.sum()))
data=dict(status='passed',connected_independent_full_pipes=count,symmetric=True,
    half_pipe_nodes=int(mask.sum()),substeps=local.substeps,local_CFL=local.diagonal_cfl,
    max_GPU_CPU_difference_K=error,relative_energy_drift=energy_relative,
    unchanged_non_pipe_temperatures=True,internal_face_counts=face_counts,
    boundary_wall=configuration('cpu_fan')['external_wall_h_W_m2K'],
    scope='Implementation conservation and geometry; not product calibration or mesh independence')
(ROOT/'heatpipe_verification.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
