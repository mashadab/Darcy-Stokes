# Stokes solver
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 04/09/2022
import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')

# import Python libraries
from scipy.sparse import bmat,csr_matrix
import matplotlib.pyplot as plt

# import personal libraries
from classfun import *  #importing the classes and relevant functions
from build_stokes_grid_fun import build_stokes_grid
from build_stokes_ops_fun import build_stokes_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from quiver_plot import quiver_plot

#problem parameter
mu  = 1.0  #Viscosity nondimensionalized 

#building grid
Gridp.xmin = 0.0 ; Gridp.xmax = 1 ; Gridp.Nx   = 5
Gridp.ymin = 0.0 ; Gridp.ymax = 1 ; Gridp.Ny   = 5
Grid = build_stokes_grid(Gridp)

#simulation name
simulation_type = 'lid_driven_cavity_flow'   #lid_driven_cavity_flow or 'no_flow' 
simulation_name = f'stokes_solver_test{simulation_type}_domain{Gridp.xmax-Gridp.xmin}by{Gridp.ymax-Gridp.ymin}_N{Gridp.Nx}by{Gridp.Ny}'

#building operators
D, Edot, Dp, Gp, I = build_stokes_ops(Grid)

A  = 2.0 * mu * D @ Edot
L  = bmat([[A, -Gp], [Dp, None]],format="csr")
fs = csr_matrix((Grid.N, 1), dtype=np.float64)



#Boundary conditions
if 'lid_driven_cavity_flow' in simulation_type:
    BC.dof_dir =  np.concatenate((Grid.dof_pene, Grid.Vx.dof_ymax[1:len(Grid.Vx.dof_ymax)-1] , Grid.dof_pc))
    BC.g = np.transpose([np.concatenate((np.zeros(Grid.N_pene), np.ones(Grid.p.Nx-1),[0.0]))])

else:#no flow
    BC.dof_dir =  np.concatenate((Grid.dof_solid_bnd, Grid.dof_pc))
    BC.g        = np.concatenate((np.zeros(Grid.N_solid_bnd), [0.0]))  

BC.dof_neu = np.array([])
[B,N,fn] = build_bnd(BC,Grid,I)

#Solving for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N)
v = u[:Grid.p.Nf,:]; p = u[Grid.p.Nf+1:,:] #Extracting velocity and pressure inside

#Plotting
quiver_plot(simulation_name,Grid,v)
