import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
# import math

N = 64
mesh = UnitSquareMesh(N, N)

V1 = FunctionSpace(mesh, "RT", 1)
V2 = FunctionSpace(mesh, "DG", 0)
W = V1*V2

x, y = SpatialCoordinate(mesh)

y_0 = 0.25 ; y_1 = 0.75; u_max = 1.0

u_1 = conditional(Or(y <= y_0, y >= y_1), 0.0, u_max*exp(1/((y - y_0)*(y - y_1)))/exp(-4/(y_1 - y_0)**2))
u_2 = 0.0

# U_1 = interpolate(u_1, V2)

u = project(as_vector([u_1, u_2]), V1)
g = project(as_vector([u_2, -u_1]), V1)

f = interpolate(div(g), V2)

sigma, theta = TrialFunctions(W)
tau, phi = TestFunctions(W)

n = FacetNormal(mesh)
a = (dot(sigma, tau) + div(tau)*theta + div(sigma)*phi)*dx
L = f*phi*dx 

w = Function(W)
solve(a == L, w)
sigma, theta = w.subfunctions

outfile = File("./results/galewsky_dirichlet_bc_comp_FE.pvd")
u.rename("velocity")
theta.rename("temperature")
outfile.write(u, theta)
