# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finite equilibrium limit for lbmpy 2.1.1's explicit KBC entropy ratio."""
import sympy as sp
from pystencils import Assignment
from lbmpy.simplificationfactory import create_simplification_strategy

def equilibrium_safe_simplification(rule):
    # At exact equilibrium both entropy inner products vanish. The higher
    # order relaxation is immaterial there; select one, rather than 0/0.
    # The threshold is below the reliable squared non-equilibrium signal
    # for float32 PDFs. Shear relaxation (physical viscosity) is unchanged.
    norm=sum(sp.Symbol(f'entropicDh_{i}')**2/sp.Symbol(f'entropicFeq_{i}')
             for i in range(rule.method.stencil.Q))
    assignments=[]
    guarded=0
    for a in rule.subexpressions:
        if a.lhs==sp.Symbol('omega_free'):
            guarded+=1
            a=Assignment(a.lhs,sp.Piecewise((sp.Float(1),norm<sp.Float('1e-14')),(a.rhs,True)))
        assignments.append(a)
    if guarded!=1:
        raise ValueError('Missing entropy rate assignment: '+str([(str(a.lhs),str(a.rhs)[:100]) for a in rule.subexpressions if 'entropicD' in str(a.rhs)]))
    rule=rule.copy(subexpressions=assignments)
    return create_simplification_strategy(rule.method)(rule)
