'''
To run, restart, and plot using e.g. 16 processes:
    $ mpiexec -n 16 python3 main.py
    $ mpiexec -n 16 python3 main.py --restart
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import math
import logging
logger = logging.getLogger(__name__)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

dealias = 3/2 
pi = np.pi

############ parameter settings
Re = 100 # set Re=100 for testing
Wi = 30
Lmax = 200
beta = 0.7
epsilon = 4e-5
Lx, Lz = 2., 1.
Nx, Nz = 384, 192
stop_sim_time = 100 + 300*restart # Stopping criteria
timestepper = d3.RK222
############ parameter settings


# Bases
coords = d3.CartesianCoordinates('x','z')
dist = d3.Distributor(coords, dtype=np.float64)
# define the coordinate system
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
p = dist.Field(name='p', bases=(xbasis, zbasis))
Cxx = dist.Field(name='Cxx', bases=(xbasis, zbasis))
Cxz = dist.Field(name='Cxy', bases=(xbasis, zbasis))
Czz = dist.Field(name='Cyy', bases=(xbasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift = lambda A: d3.Lift(A, zbasis.derivative_basis(1), -1)
grad_u = d3.grad(u)
ux = u@ex
uz = u@ez

# Velocity gradient components
dx_ux = grad_u['x','x']
dz_ux = grad_u['x','z']
dx_uz = grad_u['z','x']
dz_uz = grad_u['z','z']

# FENE-P model terms (corrected trC adjustment for 2D)
trC = Cxx + Czz
f = 1 / (1 - (trC - 2)/Lmax**2)  # 2D adjustment (equilibrium trC=2)
Txx = (Cxx * f - 1) / Wi
Txz = (Cxz * f) / Wi
Tzz = (Czz * f - 1) / Wi

# Problem setup with corrected variables and equations
problem = d3.IVP([u, p, Cxx, Cxz, Czz, tau_p], namespace=locals())

# Momentum equation
problem.add_equation(
    "Re*dt(u) + Re*grad(p) - beta*div(grad(u)) = "
    "Re*dot(u, grad(u)) + (1-beta)*div(Txx*ex*ex + Txz*ex*ez + Txz*ez*ex + Tzz*ez*ez)"
)

# Incompressibility and pressure gauge
problem.add_equation("div(u) + tau_p = 0")

# Constitutive equations for FENE-P model
problem.add_equation("dt(Cxx) + dot(u, grad(Cxx)) = 2*Cxx*dx_ux + 2*Cxz*dz_ux - (f*(Cxx - 1))/Wi")
problem.add_equation("dt(Cxz) + dot(u, grad(Cxz)) = Cxz*(dx_ux + dz_uz) + Cxx*dx_uz + Czz*dz_ux - (f*Cxz)/Wi")
problem.add_equation("dt(Czz) + dot(u, grad(Czz)) = 2*Cxz*dx_uz + 2*Czz*dz_uz - (f*(Czz - 1))/Wi")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
