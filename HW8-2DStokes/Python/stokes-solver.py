# Stokes solver
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 02/12/2021
import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')

from scipy.sparse import bmat,csr_matrix
import matplotlib.pyplot as plt

# import personal libraries
from classfun import *  #importing the classes and relevant functions
from build_stokes_grid_fun import build_stokes_grid
from build_stokes_ops_fun import build_stokes_ops

from comp_algebraicmean_optimized import comp_algebraic_mean
from comp_harmonicmean import comp_harmonicmean
from mobilityfun import mobility
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from flux_upwindfun2D_optimized import flux_upwind  #instead of from adv_opfun2D import adv_opfun
from time import perf_counter
from eval_phase_behavior import eval_phase_behavior, enthalpyfromT
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
'''
#Boundary conditions
if 'lid_driven_cavity_flow' in simulation_type:
    Grid.dof_no_pene = np.concatenate((Grid.Vx.dof_xmin, Grid.Vx.dof_xmax, Grid.Vx.N+Grid.Vy.dof_ymin , Grid.Vx.N+Grid.Vy.dof_ymax))
    Grid.N_no_pene   = np.shape(Grid.dof_no_pene)
    BC.dof_dir =  np.array([np.concatenate((Grid.dof_no_pene, Grid.Vx.dof_ymax[1:len(Grid.Vx.dof_ymax)-1] , [Grid.p.Nf+1]))])
                                #set x_max x-vel  set x_min x-vel     set y_min y-vel               set y_max y-vel          set pressure
    BC.g = np.transpose([np.concatenate((np.zeros(Grid.N_no_pene), np.ones(Grid.p.Nx-1),[0.0]))])

else:#no flow
    BC.dof_dir =  np.array([np.concatenate((Grid.Vx.dof_xmax , Grid.Vx.dof_xmin, Grid.Vx.N+Grid.Vy.dof_ymin , Grid.Vx.N+Grid.Vy.dof_ymax, [Grid.p.Nf+1]))])
                                #set x_max x-vel  set x_min x-vel     set y_min y-vel               set y_max y-vel          set pressure
    BC.g = 0.0*np.zeros((np.shape(BC.dof_dir)[1],1))  

BC.dof_neu = np.array([])
[B,N,fn] = build_bnd(BC,Grid,I)

#Solving for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N)
v = u[:Grid.p.Nf,:]; p = u[Grid.p.Nf+1:,:] #Extracting velocity and pressure inside

#Plotting
quiver_plot(simulation_name,Grid,v)
'''

## Plotting the solution
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(20,7))
ax1.set_title(f'D, nz={D.nnz}')
ax1.spy(D)

ax2.set_title(f'Edot, nz={Edot.nnz}')
ax2.spy(Edot)

ax3.set_title(f'A, nz={A.nnz}')
ax3.spy(A)

ax4.set_title(f'L, nz={L.nnz}')
ax4.spy(L)

plt.tight_layout()