import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import time
import math
from firedrake import *
from utilities import *

N = 64

mesh = UnitSquareMesh(N,N)

V_1 = VectorFunctionSpace(mesh, "CG", 1)
V_2 = FunctionSpace(mesh, "CG", 1)

Z = V_1*V_2

uT = Function(Z)
u,T = split(uT)
v,phi = TestFunctions(Z)
u_ = Function(V_1)
To = Function(V_2)
q = Function(V_2)

Dt = 0.02
half = Constant(0.5)

x,y = SpatialCoordinate(mesh)

Ro_1 = Constant(10**4)
Ro_2 = Constant(10**3) # Rossby number
Re = Constant(10**4) # Reynolds number
Pe = Constant(10**2)



# i_vel = project(as_vector([Constant(0),Constant(0)]), V_1)
wei_vort = interpolate(sin(8*pi*x)*sin(8*pi*y) + 0.4*cos(6*pi*x)*cos(6*pi*y) 
                 + 0.3*cos(10*pi*x)*cos(4*pi*y) + 0.02*sin(2*pi*y) 
                 + 0.02*sin(2*pi*x), V_2)

i_vel = project(solve_for_v_given_q(wei_vort), V_1) # corresponding to Wei's initial condition

T_ = Function(V_2).interpolate(Constant(1))
bell_1 = 0.5*(1 + cos(math.pi*min_value(sqrt(pow(x-0.25, 2) + pow(y-0.75, 2))/0.1, 1.0)))
bell_2 = 0.5*(1 + cos(math.pi*min_value(sqrt(pow(x-0.75, 2) + pow(y-0.25, 2))/0.1, 1.0)))
To = Function(V_2).interpolate(1 + bell_1)
# alpha = 0.01
# beta = 8
# f_ext = project(as_vector([alpha*y*(y-1)*sin(beta*pi*x), 
#                            alpha*sin(beta*pi*x)*sin(beta*pi*y)]), V_1)
f = Function(V_1)
gamma = -Constant(1.0)

u_.assign(0)
# f.assign(0) 
f.assign(i_vel)
q.assign(wei_vort)

F = ( inner(u-u_,v)
    + Dt*half*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*half*(1/Ro_1)*(-(u[1]+u_[1])*v[0] +(u[0]+u_[0])*v[1])
    + Dt*half*(1/Ro_2)*inner((grad(T)+grad(T_)),v)
    + Dt *half *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    - Dt * inner(f, v) 
    + (T -T_)*phi + Dt*half*(inner(u_,grad(T_)) + inner(u,grad(T)))*phi
    - Dt*gamma*half*(T - To + T_ - To)* phi
    + Dt*half*(1/Pe)*inner((grad(T)+grad(T_)),grad(phi)) )*dx

bound_cond = [DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1,2)),
              DirichletBC(Z.sub(0).sub(1), Constant(0.0), (3,4))]

outfile = File("./results/m4.pvd")
T_.rename("atm_temp")
u_.rename("atm_vel")
To.rename("ocean_temp")
q.rename("atm_vort")
outfile.write(u_, T_, To, q)

t_start = Dt
t_end = Dt*4000

t = Dt
iter_n = 1
freq = 25
t_step = freq*Dt
current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, uT, bcs = bound_cond)
    u, T = uT.subfunctions
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)

        print("t=", round(t,4))
        T.rename("atm_temp")
        u.rename("atm_vel")
        q.assign(q_from_v(u))
        outfile.write(u, T, To, q)
    u_.assign(u)
    T_.assign(T)

    t += Dt
    iter_n +=1