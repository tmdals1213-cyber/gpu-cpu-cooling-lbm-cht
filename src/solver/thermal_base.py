# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path


import sys, json, time, argparse, gc


import numpy as np


import cupy as cp


from cupyx.scipy.sparse.linalg import LinearOperator, bicgstab


from scipy.ndimage import map_coordinates


class ThermalOperator:
    def __init__(self,sim,k):
        self.sim=sim;self.shape=sim.shape;self.n=int(np.prod(sim.shape));self.k=k
        self.strides=[self.shape[1]*self.shape[2],self.shape[2],1]
        self.G=[cp.ascontiguousarray(sim.dh.gpu_arrays[f'G{a}'][sim.inner]) for a in range(3)]
        self.fluid=cp.asarray(sim.ids==0)
        self.q=cp.asarray(sim.source_w.ravel())
        self.baseD=cp.asarray(sim.wallG)
        self.rhs=self.q+cp.asarray(sim.wallGT.ravel())
        self.func=cp.RawKernel(r'''
        extern "C" __global__ void apply(const double* x,const double* c,double* y,
                            int n,int nx,int ny,int nz){
           int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
           int ix=i/(ny*nz),iy=(i/nz)%ny,iz=i%nz;
           double v=c[i]*x[i];
           if(ix>0)v+=c[n+i]*x[i-ny*nz];
           if(ix<nx-1)v+=c[2*n+i]*x[i+ny*nz];
           if(iy>0)v+=c[3*n+i]*x[i-nz];
           if(iy<ny-1)v+=c[4*n+i]*x[i+nz];
           if(iz>0)v+=c[5*n+i]*x[i-1];
           if(iz<nz-1)v+=c[6*n+i]*x[i+1];
           y[i]=v;
        }''','apply')
        self.coeff=cp.zeros((7,)+self.shape,dtype=cp.float64)
        self.operator=LinearOperator((self.n,self.n),matvec=self.apply,dtype=cp.float64)
        self.preconditioner=LinearOperator((self.n,self.n),matvec=lambda x:x/self.diag,dtype=cp.float64)
        self.normb=float(cp.linalg.norm(self.rhs))

    def apply(self,x):
        out=cp.empty_like(x)
        self.func(((self.n+255)//256,),(256,),
            (x,self.coeff,out,np.int32(self.n),*[np.int32(n) for n in self.shape]))
        return out

    def update(self,u):
        self.coeff.fill(0);D=self.coeff[0];D[:]=self.baseD
        self.boundary=[]
        for a in range(3):
            left,right=[slice(None)]*3,[slice(None)]*3
            left[a],right[a]=slice(None,-1),slice(1,None)
            left,right=tuple(left),tuple(right)
            g=self.G[a][left]
            flux=(1.184*1007.*self.sim.dx**2)*.5*(u[left+(a,)]+u[right+(a,)])
            flux=cp.where(self.fluid[left]&self.fluid[right],flux,0.)
            pos,neg=cp.maximum(flux,0),cp.minimum(flux,0)
            D[left]+=g+pos;D[right]+=g-neg
            self.coeff[2*a+2][left]=-g+neg
            self.coeff[2*a+1][right]=-g-pos
        for axis,side in self.sim.open_faces:
            face=[slice(None)]*3;face[axis]=0 if side==0 else -1;face=tuple(face)
            un=u[face+(axis,)]*(-1 if side==0 else 1)
            # Incoming ambient theta=0; outgoing scalar has zero normal gradient.
            adv=cp.maximum(un,0)*1.184*1007.*self.sim.dx**2
            inlet_g=cp.where(un<0,2*.0263*self.sim.dx,0.)
            D[face]+=adv+inlet_g
            self.boundary.append((face,adv,inlet_g,un))
        self.diag=D.ravel()
        assert bool(cp.all(self.diag>0))

    def solve(self,x0,rtol=2e-7):
        start=time.perf_counter();calls=[0];x=cp.asarray(x0).ravel().copy();attempts=[]
        def callback(_):calls[0]+=1
        for restart in range(6):
            x,info=bicgstab(self.operator,self.rhs,x0=x,M=self.preconditioner,
                rtol=rtol,atol=0.,maxiter=20000,callback=callback)
            res=float(cp.linalg.norm(self.apply(x)-self.rhs)/self.normb)
            attempts.append([int(info),res])
            if not np.isfinite(res):raise RuntimeError('Nonfinite thermal solve')
            if res<max(1e-6,rtol*3):break
            print('thermal restart',attempts[-1],flush=True)
        if res>max(2e-6,rtol*10):raise RuntimeError(f'Thermal solve failed {res}')
        return x.reshape(self.shape),dict(iterations=calls[0],residual=res,
            runtime_s=time.perf_counter()-start,attempts=attempts)

    def balances(self,T):
        qcond=float(cp.sum(cp.asarray(self.sim.wallG)*T-cp.asarray(self.sim.wallGT)))
        qadv=0.;qinlet=0.;vout=0.;vin=0.
        for face,adv,g,un in self.boundary:
            qadv+=float(cp.sum(adv*T[face]));qinlet+=float(cp.sum(g*T[face]))
            vout+=float(cp.maximum(un,0).sum()*self.sim.dx**2)
            vin+=float(-cp.minimum(un,0).sum()*self.sim.dx**2)
        residual=float(cp.linalg.norm(self.apply(T.ravel())-self.rhs)/self.normb)
        return dict(boundary_heat_W=qcond+qadv+qinlet,advective_heat_W=qadv,
            conductive_heat_W=qcond+qinlet,volume_in_m3_s=vin,volume_out_m3_s=vout,
            volume_imbalance=abs(vin-vout)/max(vin,vout,1e-30),thermal_residual=residual)

