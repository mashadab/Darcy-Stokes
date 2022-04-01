import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')

# import python libraries
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D_latest import build_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen
from comp_mean_matrix import comp_mean
from solve_Helmholtz import solve_Helmholtz
from solve_Poisson import solve_Poisson

#Simulation parameters
Param.HD = 25;                 # Dimensionless ice shell thickness    [-]
Param.LD = 25;                 # Dimensionless width of the domain    [-]
Param.DD = 3;                  # Dimensionless thickness of melt lens [-]
Param.phi_melt   = 2e-1;       # Porosity of near surface melt
Param.phi_shell  = 2e-1;       # Porosity of near surface melt
Param.m = 1;                   # compaction viscosity exponent [-]
Param.n = 2;                   # compaction viscosity exponent [-]
Param.tDmax = 4;               # Dimensionless simulation time [-]
Param.theta = .5;              # Crank-Nicholson (implicit)


# Build grid and operator
Grid.xmin = 0; Grid.xmax = Param.HD; Grid.Nx = 100
Grid.ymin = 0; Grid.ymax = Param.LD; Grid.Ny = 100
Grid        = build_grid(Grid)
[Xc,Zc] = np.meshgrid(Grid.xc,Grid.yc)
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))
Zc_col  = np.reshape(np.transpose(Zc), (Grid.N,-1))

[D,G,C,I,M] = build_ops(Grid)
Gamma       = np.zeros((Grid.N,1))

## Build boundary conditions

#1. Helmholtz equation 
BC.h.dof_dir   = np.array([])
BC.h.dof_f_dir = np.array([]) 
BC.h.g         = np.array([]) 
BC.h.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.h.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.h.qb        = np.array([[0],[0]])

[B_h,N_h,fn_h]     = build_bnd(BC.h, Grid, I)

#2. Poisson equation 
BC.u.dof_dir   = np.array([Grid.dof_xmin[0]])
BC.u.dof_f_dir = np.array([Grid.dof_f_xmin[0]]) 
BC.u.g         = np.array([[0]]) 
BC.u.dof_neu   = np.array([Grid.dof_xmax[0]])
BC.u.dof_f_neu = np.array([Grid.dof_f_xmax[0]])
BC.u.qb        = np.array([[0]])

[B_u,N_u,fn_u]     = build_bnd(BC.u, Grid, I)

#3. Porosity evolution equation 
BC.phi.dof_dir  = np.array([])
BC.phi.dof_f_dir= np.array([]) 
BC.phi.g        = np.array([]) 
BC.phi.dof_neu  = np.hstack([Grid.dof_xmin,Grid.dof_xmax])
BC.phi.dof_f_neu= np.hstack([Grid.dof_f_xmin,Grid.dof_f_xmax])
BC.phi.qb       = np.transpose([np.hstack([np.zeros_like(Grid.dof_xmin),np.zeros_like(Grid.dof_xmax)])])

[B_phi,N_phi,fn_phi] = build_bnd(BC.phi, Grid, I)

## Initial condition
phiD = 1 + 0.1*np.cos(2*np.pi/Param.HD*Zc_col)

## Solve the equations
#1. Helmholtz equation 
[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phiD,Param.n,Param.m,Grid,B_h,N_h,fn_h,BC.h,Zc_col,Gamma)

#2. Poisson equation 
[uD,vD]    = solve_Poisson(D,G,I,phiD,Param.m,pD,Grid,B_u,N_u,fn_u,BC.u)

#3. Porosity evolution equation 
[phiD,Av]  = evolve_porosity(D,I,phiD,vD,pD,B_phi,N_phi,BC.phi,Grid,Param.phi_shell,Param.theta,.1);





'''

#Plotting the solution
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(10,7))
fig.suptitle('Modified Helmholtz and Poisson problems')

ax1.plot(phi,zc,label='$\phi_D$')
ax1.set_xlabel(r'$\phi_D$')
ax1.set_ylabel(r'$z_D$')

ax2.plot(hD,zc,label=r'$h_D$')
ax2.plot(uD,zc,label=r'$u_D$')
ax2.set_xlabel(r'$h_D,u_D$')
ax2.legend(loc='best')

ax3.plot(pD,zc)
ax3.set_xlabel(r'$p_D$')

ax4.plot(qD,zf,label=r'$q_D$')
ax4.plot(vD,zf,label=r'$v_D$')
ax4.set_xlabel(r'$q_D,v_D$')
ax4.legend(loc='best')
plt.tight_layout()

'''

