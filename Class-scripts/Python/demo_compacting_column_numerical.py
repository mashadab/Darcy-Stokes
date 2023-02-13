
#Lecture 12 (Feb 24): Solving the flow problem for an Instanteneous Compacting Column Using Numerics
#Author: Mohammad Afzal Shadab and Marc Hesse

#append paths
import sys
sys.path.insert(1,'../../HW1-Numerics/Python')
sys.path.insert(1,'../../HW2-BC_LBVP/Python')
sys.path.insert(1,'../../HW3-Hetero-Fluxes-NeuBC/Python')

#import library and modules
import numpy as np
import matplotlib.pyplot as plt
from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D import build_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen


##############################################################################
#Analytic solutions
##############################################################################
class analytical:
    #dimensionless fluid overpressure head
    hD = lambda zD,HD: zD + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                            (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                            
    #dimensionless relative volumetric flux of fluid w.r.t. solid velocity: qD = - d hD/ dzD 
    qD = lambda zD,HD: -1 - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                            (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                            
    #dimensionless fluid pressure pD = hD - zD
    pD = lambda zD,HD: (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                       (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
    
    #dimensionless solid velocity potential, setting c4 = 0
    uD = lambda zD,HD: -zD - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                             (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
    
    #dimensionless solid velocity vD = - d uD/d zD 
    vD = lambda zD,HD:  1  + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                             (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)

#####################################################################################
#Numerical simulation
#####################################################################################

#parameters
HD = 0.1  #dimensionless domain length

#building grid and operators
Grid.xmin = 0; Grid.xmax = HD; Grid.Nx = 50

Grid = build_grid(Grid)         #building grid
zc   = Grid.xc; zf   = Grid.xf; za = np.linspace(0,HD,1000)
[D,G,C,I,M] = build_ops(Grid)   #building operators

#####################################################################################
# 1. Modified Helmholtz problem (for overpressure head hD)
#####################################################################################
#boundary conditions:
BC.h.dof_dir   = np.array([]) 
BC.h.dof_f_dir = np.array([])
BC.h.g         = np.array([]) 
BC.h.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.h.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.h.qb        = np.array([[0],[0]])

[B_h,N_h,fn_h] = build_bnd(BC.h,Grid,I) #building constraint mat. B, null space N, flux's source term fn

#building operators  and rhs
L_h          = - D @ G + I
fs_h         =   np.transpose([zc])

#solving the linear boundary value problems
hD           = solve_lbvp(L_h, fs_h + fn_h, B_h, BC.h.g, N_h) #fn is zero vector
pD           = hD - zc

#compute the flux
flux         = lambda h: - G @ np.transpose([h])
res          = lambda h,cell: L_h[cell,:] @ np.transpose([h]) - fs_h[cell,:]
qD           = comp_flux_gen(flux, res, hD, Grid, BC.h)

#####################################################################################
# 2. Poisson problem (for solid velocity potential uD)
#####################################################################################
#boundary conditions:
BC.u.dof_dir   = Grid.dof_xmin
BC.u.dof_f_dir = Grid.dof_f_xmin
BC.u.g         = np.array([analytical.uD(Grid.xc[BC.u.dof_dir-1][0],HD)]) 
BC.u.dof_neu   = np.array([])
BC.u.dof_f_neu = np.array([])
BC.u.qb        = np.array([])

[B_u,N_u,fn_u] = build_bnd(BC.u,Grid,I) #building constraint mat. B, null space N, flux's source term fn


#building operators and rhs
L_u          = - D @ G
fs_u         =   np.transpose([pD])

#solving the linear boundary value problems
uD           = solve_lbvp(L_u, fs_u + fn_u, B_u, BC.u.g, N_u) #fn is zero vector

#compute flux
flux         = lambda u: -G @ u
res          = lambda u,cell: L_u[cell,:] @ u - fs_u[cell,:]
vD           = comp_flux_gen(flux, res, uD, Grid, BC.u)

#plotting 
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(15,7))
fig.suptitle(f'Analytic solutions to HD = {HD}')

ax1.plot(hD,zc,'k.',label=r'Numerical')
ax1.plot(analytical.hD(za,HD),za,'r-',label=r'Analytical')
ax1.set_ylabel('$z_D$')
ax1.set_xlabel('$h_D$')
ax1.legend(loc = 'best')

ax2.plot(pD,zc,'k.',label=r'N')
ax2.plot(analytical.pD(za,HD),za,'r-',label=r'A')
ax2.set_xlabel('$p_D$')

ax3.plot(uD,zc,'k.',label=r'N')
ax3.plot(analytical.uD(za,HD),za,'r-',label=r'A')
ax3.set_xlabel('$u_D$')

ax4.plot(qD,zf,'k.',label=r'N')
ax4.plot(analytical.qD(za,HD),za,'r-',label=r'A')
ax4.set_xlabel('$q_D$')

ax5.plot(vD,zf,'k.',label=r'N')
ax5.plot(analytical.vD(za,HD),za,'r-',label=r'A')
ax5.set_xlabel('$v_D$')

plt.tight_layout()
