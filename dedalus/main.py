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
# zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
# should use chebyshev for vertical direction bc of boundary layer
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-0.5*Lz, 0.5*Lz), dealias=dealias) 

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
p = dist.Field(name='p', bases=(xbasis, zbasis))
# Cxx = dist.Field(name='Cxx', bases=(xbasis, zbasis))
# Cxz = dist.Field(name='Cxy', bases=(xbasis, zbasis))
# Czz = dist.Field(name='Cyy', bases=(xbasis, zbasis))
# Cxx, Cxy, Czz canbe replaced by one C variable using TensorField
# read guide here: https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_2.html
C = dist.TensorField((coords, coords),name='C', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_C1 = dist.TensorField((coords, coords), name='tau_C1', bases=xbasis)
tau_C2 = dist.TensorField((coords, coords), name='tau_C2', bases=xbasis)
tau_C = dist.TensorField((coords, coords), name='tau_C')
# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
lap_u = d3.div(grad_u)
# grad_C = d3.grad(C)
grad_C = d3.grad(C) + ez*lift(tau_C1) # First-order reduction
lap_C = d3.div(grad_C)
trC = d3.trace(C)
ux = u@ex
uz = u@ez

baru = dist.Field(bases=(zbasis)) # base flow
baru['g'] = 1.0/4.0 - z**2 #???????????????????????

# Velocity gradient components
# dx_ux = grad_u['x','x']
# dz_ux = grad_u['x','z']
# dx_uz = grad_u['z','x']
# dz_uz = grad_u['z','z']

# define an identity tensor I = [1 0; 0 1]
I = dist.TensorField((coords, coords))
I['g'][0,0] = 1
I['g'][1,1] = 1



# FENE-P model terms (corrected trC adjustment for 2D)
# trC = Cxx + Czz
# f = 1 / (1 - (trC - 2)/Lmax**2)  # 2D adjustment (equilibrium trC=2)
# Txx = (Cxx * f - 1) / Wi
# Txz = (Cxz * f) / Wi
# Tzz = (Czz * f - 1) / Wi
# these all can be replace by only
T = (C/(1.0 - (trC - 3.0)/Lmax**2) - I) / Wi


# Problem setup with corrected variables and equations
# problem = d3.IVP([u, p, Cxx, Cxz, Czz, tau_p], namespace=locals())
problem = d3.IVP([u, p, C, tau_p, tau_u1, tau_u2], namespace=locals())

# Incompressibility and pressure gauge
# problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("trace(grad(u))+tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Momentum equation
# problem.add_equation(
#     "Re*dt(u) + Re*grad(p) - beta*div(grad(u)) = "
#     "Re*dot(u, grad(u)) + (1-beta)*div(Txx*ex*ex + Txz*ex*ez + Txz*ez*ex + Tzz*ez*ez)"
# )
problem.add_equation("Re*dt(u) + grad(p) - beta*lap_u + lift(tau_u2) = - Re*u@grad_u + (1.0-beta)*div(T)")
problem.add_equation("dt(C) = epsilon*lap_C - T- u@grad_C + C@grad_u + grad_u.T@C") 

# add boundary condition
################### NO-SLIP #####################
problem.add_equation("u(z=0) = 0")
problem.add_equation("u(z=Lz) = 0")
# problem.add_equation("C(z=0) = 0") # dont need this
# problem.add_equation("C(z=Lz) = 0")
################### NO-SLIP #####################

# Constitutive equations for FENE-P model
# problem.add_equation("dt(Cxx) + dot(u, grad(Cxx)) = 2*Cxx*dx_ux + 2*Cxz*dz_ux - (f*(Cxx - 1))/Wi")
# problem.add_equation("dt(Cxz) + dot(u, grad(Cxz)) = Cxz*(dx_ux + dz_uz) + Cxx*dx_uz + Czz*dz_ux - (f*Cxz)/Wi")
# problem.add_equation("dt(Czz) + dot(u, grad(Czz)) = 2*Cxz*dx_uz + 2*Czz*dz_uz - (f*(Czz - 1))/Wi")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

if not restart:
    p.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    u.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    C.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    file_handler_mode = 'overwrite'
    initial_timestep = 0.02
    max_timestep = 0.02
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s1.h5')
    max_timestep = 0.02
    file_handler_mode = 'append'
    logger.info('Imported last-step data successfully')

dataset = solver.evaluator.add_file_handler('snapshots', sim_dt=1.0, max_writes=1000, mode=file_handler_mode)
dataset.add_task(u@ex, name='velocity_u')
dataset.add_task(u@ez, name='velocity_w')
dataset.add_task(C, name='C')

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=100, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_timestep, cadence=10,
             max_dt=max_timestep,
             safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5
             )
CFL.add_velocity(u)

xg = xbasis.global_grid(dist, scale=dealias)
zg = zbasis.global_grid(dist, scale=dealias)
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)  
        if (solver.iteration-1) % 100 == 0:
            logger.info('Completed iteration {}, time={:.3f}, dt={:.10f}'.format(solver.iteration, solver.sim_time, timestep)) 
            ug = u.allgather_data('g')
            
            if dist.comm.rank == 0:
                # plot w (vertical velocity)
                plt.figure(figsize=(10,3))
                plt.pcolormesh(xg.ravel(),zg.ravel(),ug[1].transpose(),cmap='jet')
                plt.colorbar() 
                plt.xticks([0,Lx])
                plt.yticks([0,Lz])
                plt.xlabel(r'$x$')
                plt.ylabel(r'$z$')
                plt.title("t = {:.3f}".format(solver.sim_time))
                plt.savefig('snapshots/uz={:010.3f}.png'.format(solver.sim_time), bbox_inches='tight')
                plt.close()
            ###########################
            if math.isnan(np.max(ug)):
                logger.error('NaN values')
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()