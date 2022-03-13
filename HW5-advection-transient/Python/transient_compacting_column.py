import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')

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
from flux_upwind import flux_upwind


def plotting(phi,hD,uD,qD,vD,pD,zc,zf,i,dt):
    #Plotting the solution
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(10,7))
    fig.suptitle(f'Modified Helmholtz and Poisson problems at t= {(i+1)*dt}',)
    
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

# Parameters
n_exp = 3
m_exp = 1
phic  = 1e-3

# Dimensionless parameters
tmax  = 10
HD    = 50
Nt    = tmax*100
theta = 1 #theta: 1 = forward Euler (explicit), 0.5 = Crank Nicolson, 0 = backward Euler

# Build grid and operator
Grid.xmin = 0; Grid.xmax = HD; Grid.Nx = 300
Grid        = build_grid(Grid)
zc          = Grid.xc
zf          = Grid.xf
[D,G,C,I,M] = build_ops(Grid)
Gamma       = np.zeros((Grid.N,1))

############################################################
#1. Modified Helmholtz equation 
############################################################
# Build boundary conditions
BC.h.dof_dir   = np.array([])
BC.h.dof_f_dir = np.array([]) 
BC.h.g         = np.array([]) 
BC.h.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.h.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.h.qb        = np.array([[0],[0]])
[B_h,N_h,fn_h] = build_bnd(BC.h, Grid, I)

############################################################
#2. Poisson equation 
############################################################
# Build boundary conditions
BC.u.dof_dir   = np.array([Grid.dof_xmin[0]])
BC.u.dof_f_dir = np.array([Grid.dof_f_xmin[0]]) 
BC.u.g         = np.array([[HD]]) 
BC.u.dof_neu   = np.array([Grid.dof_xmax[0]])
BC.u.dof_f_neu = np.array([Grid.dof_f_xmax[0]])
BC.u.qb        = np.array([[0]])
[B_u,N_u,fn_u] = build_bnd(BC.u, Grid, I)

############################################################
#3. Porosity evolution
############################################################
# Build boundary conditions
BC.phi.dof_dir   = np.array([])
BC.phi.dof_f_dir = np.array([]) 
BC.phi.g         = np.array([]) 
BC.phi.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.phi.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.phi.qb        = np.array([[0],[0]])
[B_phi,N_phi,fn_phi]= build_bnd(BC.phi, Grid, I)

#Initial condition
phi = np.ones(Grid.Nx)
dt  = tmax/Nt

#Steady state
#phi = 1 + 0.1*np.cos(2*np.pi/HD*zc);
#[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B_h,N_h,fn_h,BC.h,zc,Gamma)
#[uD,vD]    = solve_Poisson(D,G,I,phi,m_exp,pD,Grid,B_u,N_u,fn_u,BC.u)


for i in range(0,Nt):

    if (i+1)%100 ==0: plotting(phi,hD,uD,qD,vD,pD,zc,zf,i,dt)

    [hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B_h,N_h,fn_h,BC.h,zc,Gamma)

    [uD,vD]    = solve_Poisson(D,G,I,phi,m_exp,pD,Grid,B_u,N_u,fn_u,BC.u)
    print(f'i = {i},t = {dt*i},Summing the porosity = {np.sum(phi)}')
    
    A  = flux_upwind(vD,Grid)
    P  = sp.spdiags(pD, 0, Grid.Nx, Grid.Nx)
    L  = phic * D @ A - P
    IM = I + dt * (1-theta) * L
    EX = I - dt * theta * L
    phi= solve_lbvp(IM, EX @ phi, B_phi, BC.phi.g, N_phi)
    
    if np.max(phi*phic) > 1 or np.min(phi*phic) < 0 : break

