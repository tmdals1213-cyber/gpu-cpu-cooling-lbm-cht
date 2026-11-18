# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-equilibrium extrapolation pressure boundary for GPU LBM.

Copy the local non-equilibrium part; evaluate equilibrium at prescribed density
and extrapolated local velocity. Both ordinary and zero-centred PDFs are handled.
"""
import sympy as sp
from pystencils import Assignment,AssignmentCollection
from lbmpy.boundaries.boundaryconditions import LbBoundary


class NonEquilibriumPressure(LbBoundary):
    def __init__(self,density=1.,name=None):
        self.density=density
        super().__init__(name or 'Non-equilibrium pressure',calculate_force_on_boundary=False)

    def __call__(self,f_out,f_in,dir_symbol,inv_dir,lb_method,index_field,force_vector):
        eq=lb_method.get_equilibrium()
        eq=eq.new_with_substitutions({s:f_out(i) for i,s in enumerate(lb_method.pre_collision_pdf_symbols)})
        rho=lb_method.zeroth_order_equilibrium_moment_symbol
        inv=inv_dir[dir_symbol]
        bg=lb_method.conserved_quantity_computation.background_density if lb_method._zero_centered else 0
        directions=list(lb_method.stencil);branches=[]
        for i,direction in enumerate(directions):
            j=directions.index(tuple(-v for v in direction))
            eq_absolute=eq.main_assignments[j].rhs+lb_method.weights[j]*bg
            value=f_out(j)+(self.density-rho)/rho*eq_absolute
            branches.append((value,sp.Eq(dir_symbol,i)))
        new=sp.Piecewise(*branches,(0,True))
        ac=AssignmentCollection([Assignment(f_in(inv),new)],eq.subexpressions)
        ac=ac.new_without_unused_subexpressions();ac.topological_sort();return ac


class EquilibriumPressure(NonEquilibriumPressure):
    """Ambient reservoir: prescribed density and extrapolated local velocity.

    Incoming populations have equilibrium stress. This dissipative reservoir
    boundary is placed at the far case opening, not on the CPU or heatsink.
    """
    def __call__(self,f_out,f_in,dir_symbol,inv_dir,lb_method,index_field,force_vector):
        eq=lb_method.get_equilibrium()
        eq=eq.new_with_substitutions({s:f_out(i) for i,s in enumerate(lb_method.pre_collision_pdf_symbols)})
        rho=lb_method.zeroth_order_equilibrium_moment_symbol
        bg=lb_method.conserved_quantity_computation.background_density if lb_method._zero_centered else 0
        directions=list(lb_method.stencil);branches=[]
        for i,direction in enumerate(directions):
            j=directions.index(tuple(-v for v in direction));weight=lb_method.weights[j]
            value=self.density/rho*(eq.main_assignments[j].rhs+weight*bg)-weight*bg
            branches.append((value,sp.Eq(dir_symbol,i)))
        ac=AssignmentCollection([Assignment(f_in(inv_dir[dir_symbol]),sp.Piecewise(*branches,(0,True)))],eq.subexpressions)
        ac=ac.new_without_unused_subexpressions();ac.topological_sort();return ac
