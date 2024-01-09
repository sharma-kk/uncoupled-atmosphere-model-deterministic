from firedrake import *

def solve_for_v_given_q(q):
    """ given vorticity, solve for corresp. vel. Vel obtained needs to be projected using project(v,fucntion_space)"""
    msh = q.function_space().mesh()
    Vcg = FunctionSpace(msh, "CG", 1)

    psi = TrialFunction(Vcg)
    phi = TestFunction(Vcg)

    a = (inner(grad(psi), grad(phi))) * dx
    l = -q * phi * dx
    psi_0 = Function(Vcg)

    bc = DirichletBC(Vcg, 0.0, 'on_boundary')
    solve(a == l, psi_0, bcs = bc)
    return as_vector([-psi_0.dx(1), psi_0.dx(0)])

    
