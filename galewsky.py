import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
# import math

N = 64
mesh = UnitSquareMesh(N, N)

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

y_0 = 0.25 ; y_1 = 0.75; u_max = 1.0

u_1 = conditional(Or(y <= y_0, y >= y_1), 0.0, u_max*exp(1/((y - y_0)*(y - y_1)))/exp(-4/(y_1 - y_0)**2))
u_2 = 0.0

# U_1 = interpolate(u_1, V2)

u = project(as_vector([u_1, u_2]), V1)
g = project(as_vector([u_2, -u_1]), V1)

f = interpolate(div(g), V2)

theta  = TrialFunction(V2)
phi = TestFunction(V2)

a = -inner(grad(theta), grad(phi))*dx
L = f*phi*dx

theta = Function(V2)
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD) # this is required with Neumann bcs
solve(a == L, theta, nullspace=nullspace)

outfile = File("./results/galewsky.pvd")
u.rename("velocity")
theta.rename("temperature")
outfile.write(u, theta)

# the above code creates geostrophic balance conditions for theta and vel. We assume constant coriolis parameter. 