import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')

# import python libraries
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D import build_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen
from comp_mean_matrix import comp_mean
from solve_Helmholtz import solve_Helmholtz


# Parameters
n_exp = 3
m_exp = 1
HD    = 30


# Build grid and operator
Grid.xmin = 0; Grid.xmax = HD; Grid.Nx = 350
Grid        = build_grid(Grid)
zc          = Grid.xc
zf          = Grid.xf
[D,G,C,I,M] = build_ops(Grid)
Gamma       = np.zeros((Grid.N,1))

# Build boundary conditions
BC.dof_dir   = np.array([])
BC.dof_f_dir = np.array([]) 
BC.g         = np.array([]) 
BC.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.qb        = np.array([[0],[0]])

[B,N,fn]     = build_bnd(BC, Grid, I)

#Solve the Helmholtz equation 
phi = 1 + 0.1*np.cos(2*np.pi/HD*zc);
[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B,N,fn,BC,zc,Gamma);

#Plotting the solution
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(10,7))
fig.suptitle('Modified Helmholtz problem')

ax1.plot(phi,zc)
ax1.set_xlabel(r'$\phi_D$')
ax1.set_ylabel(r'$z_D$')

ax2.plot(hD,zc)
ax2.set_xlabel(r'$h_D$')

ax3.plot(pD,zc)
ax3.set_xlabel(r'$p_D$')

ax4.plot(qD,zf)
ax4.set_xlabel(r'$qD$')

plt.tight_layout()



