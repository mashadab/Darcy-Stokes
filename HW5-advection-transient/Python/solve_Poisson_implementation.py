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
from solve_Poisson import solve_Poisson

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


#1. Solve the Helmholtz equation 
# Build boundary conditions
BC.h.dof_dir   = np.array([])
BC.h.dof_f_dir = np.array([]) 
BC.h.g         = np.array([]) 
BC.h.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.h.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.h.qb        = np.array([[0],[0]])

[B_h,N_h,fn_h]     = build_bnd(BC.h, Grid, I)

#Solve the Helmholtz equation 
phi = 1 + 0.1*np.cos(2*np.pi/HD*zc);
[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B_h,N_h,fn_h,BC.h,zc,Gamma)

#2. Solve the Poisson equation 
# Build boundary conditions
BC.u.dof_dir   = np.array([Grid.dof_xmin[0]])
BC.u.dof_f_dir = np.array([Grid.dof_f_xmin[0]]) 
BC.u.g         = np.array([[HD]]) 
BC.u.dof_neu   = np.array([Grid.dof_xmax[0]])
BC.u.dof_f_neu = np.array([Grid.dof_f_xmax[0]])
BC.u.qb        = np.array([[0]])

[B_u,N_u,fn_u]     = build_bnd(BC.u, Grid, I)

#Solve the Helmholtz equation 
phi = 1 + 0.1*np.cos(2*np.pi/HD*zc);
[uD,vD] = solve_Poisson(D,G,I,phi,m_exp,pD,Grid,B_u,N_u,fn_u,BC.u)

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



