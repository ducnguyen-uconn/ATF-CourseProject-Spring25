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

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

# FENE-P Model Terms
trC = Cxx + Czz
f = 1 / ((1 - (trC - 3))/Lmax**2) 
Txx = (Cxx * f - 1) / Wi
Txz = (Cxz * f) / Wi
Tzz = (Czz * f - 1) / Wi

# Problem Setup
problem = d3.IVP([u, p, Cxx, Cxz, Czz], namespace=locals())

# Momentum equation (linear LHS)
problem.add_equation(
    "Re*dt(u) + Re*grad(p) - beta*div(grad(u)) = "
    "Re*dot(u, grad(u)) + (1-beta)*div(Txx*ex*ex + Txz*ex*ez + Txz*ez*ex + Tzz*ez*ez)"
)

# Incompressibility condition
problem.add_equation("div(u) = 0")
