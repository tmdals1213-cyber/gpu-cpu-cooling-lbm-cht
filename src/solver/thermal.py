# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU conservative steady heat balance for masked inlet/outlet ports."""
from pathlib import Path
import sys
from runtime_paths import OUTPUT as ROOT
from thermal_base import ThermalOperator as BaseOperator
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import spilu,LinearOperator

class ThermalOperator(BaseOperator):
    def solve(self,x0,rtol=2e-7):
        # ILU(0) on the GPU preserves the original discrete operator. Reuse
        # the factor for repeated physical steps with the same mass shift.
        signature=getattr(self,'_mass_step',None)
        if not hasattr(self,'_ilu_signature') or self._ilu_signature!=signature:
            idx=cp.arange(self.n,dtype=cp.int32)
            rr=[];cc=[];vv=[]
            offsets=[0,-self.strides[0],self.strides[0],-self.strides[1],self.strides[1],-1,1]
            for coeff,offset in zip(self.coeff,offsets):
                value=coeff.ravel();valid=value!=0
                rr.append(idx[valid]);cc.append((idx+offset)[valid]);vv.append(value[valid])
            matrix=coo_matrix((cp.concatenate(vv),(cp.concatenate(rr),cp.concatenate(cc))),shape=(self.n,self.n)).tocsr()
            self._ilu=spilu(matrix,fill_factor=1)
            self.preconditioner=LinearOperator((self.n,self.n),matvec=self._ilu.solve,dtype=cp.float64)
            self._ilu_signature=signature
        return super().solve(x0,rtol)

    def step_implicit(self,old,dt):
        """Backward Euler in physical seconds, preserving all solid/fluid heat capacities."""
        old=cp.asarray(old,dtype='float64').reshape(self.shape)
        original_rhs=self.rhs;original_norm=self.normb
        mass=cp.asarray(self.sim.capacity)/dt
        self.coeff[0]+=mass;self.diag=self.coeff[0].ravel()
        self.rhs=original_rhs+(mass*old).ravel();self.normb=float(cp.linalg.norm(self.rhs))
        self._mass_step=dt
        try:new,info=self.solve(old,rtol=1e-9)
        finally:
            self.coeff[0]-=mass;self.diag=self.coeff[0].ravel()
            self.rhs=original_rhs;self.normb=original_norm
            self._mass_step=None
        return new,info

    def update(self,u,eddy_nu=None):
        if hasattr(self,'_ilu_signature'):del self._ilu_signature
        self.coeff.fill(0);D=self.coeff[0];D[:]=self.baseD
        self.boundary=[]
        for a in range(3):
            left,right=[slice(None)]*3,[slice(None)]*3
            left[a],right[a]=slice(None,-1),slice(1,None)
            left,right=tuple(left),tuple(right);g=self.G[a][left]
            if eddy_nu is not None:
                # Eddy viscosity is in lattice units, matching the LBM output.
                extra=self.sim.fluid_rhocp*self.sim.dx**3/self.sim.dt/self.sim.turbulent_prandtl*.5*(eddy_nu[left]+eddy_nu[right])
                g=g+cp.where(self.fluid[left]&self.fluid[right],extra,0.)
            flux=self.sim.fluid_rhocp*self.sim.dx**2*.5*(u[left+(a,)]+u[right+(a,)])
            flux=cp.where(self.fluid[left]&self.fluid[right],flux,0.)
            pos,neg=cp.maximum(flux,0),cp.minimum(flux,0)
            D[left]+=g+pos;D[right]+=g-neg
            self.coeff[2*a+2][left]=-g+neg;self.coeff[2*a+1][right]=-g-pos
        for port,mask_cpu in zip(self.sim.ports,self.sim.port_masks):
            mask=cp.asarray(mask_cpu);axis=port['axis'];sign=-1 if port['side']==0 else 1
            un=cp.where(mask,u[...,axis]*sign,0.)
            adv=cp.maximum(un,0)*self.sim.fluid_rhocp*self.sim.dx**2
            kg=cp.asarray(self.k[...,axis])
            inlet_g=cp.where(un<0,2*kg*self.sim.dx,0.)
            D+=adv+inlet_g
            self.boundary.append((adv,inlet_g,un))
        self.diag=D.ravel();assert bool(cp.all(self.diag>0))

    def balances(self,T):
        qcond=float(cp.sum(cp.asarray(self.sim.wallG)*T-cp.asarray(self.sim.wallGT)))
        qadv=0.;qinlet=0.;vout=0.;vin=0.
        for adv,g,un in self.boundary:
            qadv+=float(cp.sum(adv*T));qinlet+=float(cp.sum(g*T))
            vout+=float(cp.maximum(un,0).sum()*self.sim.dx**2)
            vin+=float(-cp.minimum(un,0).sum()*self.sim.dx**2)
        residual=float(cp.linalg.norm(self.apply(T.ravel())-self.rhs)/self.normb)
        return dict(boundary_heat_W=qcond+qadv+qinlet,advective_heat_W=qadv,
            conductive_heat_W=qcond+qinlet,volume_in_m3_s=vin,volume_out_m3_s=vout,
            volume_imbalance=abs(vin-vout)/max(vin,vout,1e-30),thermal_residual=residual)
