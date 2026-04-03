# import python libraries
import numpy as np
#import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from scipy.linalg import solve as fullspsolve
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
# import personal libraries
from build_gridfun2D import build_grid, build_grid_unstructured, plot_grid_unstructured
from build_opsfun2D import build_ops, build_ops_unstructured
from build_bndfun_optimized import build_bnd
#from mobilityfun import mobility

import scipy.sparse as sp
import scipy.sparse.linalg as linalg

def solve_lbvp(L,f,B,g,N):

    # author: Mohammad Afzal Shadab
    # date: 2/25/2020
    # Description
    # Computes the solution $u$ to the linear differential problem given by
    #
    # $$\mathcal{L}(u)=f \quad x\in \Omega $$
    #
    # with boundary conditions
    #
    # $$\mathcal{B}(u)=g \quad x\in\partial\Omega$$.
    #
    # Input:
    # L = matrix representing the discretized linear operator of size N by N, 
    #     where N is the number of degrees of freedom
    # f = column vector representing the discretized r.h.s. and contributions
    #     due non-homogeneous Neumann BC's of size N by 1
    # B = matrix representing the constraints arising from Dirichlet BC's of
    #     size Nc by N
    # g = column vector representing the non-homogeneous Dirichlet BC's of size
    #     Nc by 1.
    # N = matrix representing a orthonormal basis for the null-space of B and
    #     of size N by (N-Nc).
    # Output:
    # u = column vector of the solution of size N by 1
      
    if B.nnz == 0:
        u = np.transpose([linalg.spsolve(L, f)])
    else:
        
        up = np.transpose([sp.csr_matrix.transpose(B) @ linalg.spsolve((B @ sp.csr_matrix.transpose(B)),g)])
        
        u0 = np.transpose([N @ linalg.spsolve(sp.csr_matrix.transpose(N) @ L @ N,sp.csr_matrix.transpose(N) @ (f-L @ up))])
        
        u = u0 + up

    return u;



class grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx = []

class Param:
    def __init__(self):
        self.dof_dir = []       # identify cells on Dirichlet bnd
        self.dof_f_dir = []     # identify faces on Dirichlet bnd
        self.dof_neu = []       # identify cells on Neumann bnd
        self.dof_f_neu = []     # identify faces on Neumann bnd
        self.g = []             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
        self.qb = []            

#grid and operators
grid.xmin = 0.0; grid.xmax = 1.0; grid.Nx   = 4; grid.scale_x   = np.transpose([np.linspace(1,4,4)])#;grid.type   = 'fixed_wdth'
grid.ymin = 0.0; grid.ymax = 1.0; grid.Ny   = 3
grid = build_grid_unstructured(grid)
[D,G,C,I,M]=build_ops_unstructured(grid)
#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([grid.dof_xmax])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([grid.dof_f_xmax])   # identify faces on Neumann bnd
Param.qb = 1.0*np.ones((len(Param.dof_neu[0]),1))  # set flux at Neumann bnd
Param.g  = Param.qb*grid.xc[Param.dof_dir[0,0]-1]#np.array([[0.0],[0.0]]) # set head at Dirichlet bnd #analytic solution
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
fs = np.zeros([grid.N,1])                       # r.h.s. (zero)
L = -D@G                        # Laplacian

f = fs+fn

xc_unstr = grid.xc

u_unstr = solve_lbvp(L,f,B,Param.g,N)                 # Solve linear boundary value problem


#structured
grid.xmin = 0.0; grid.xmax = 1.0; grid.Nx   = 4;
grid.ymin = 0.0; grid.ymax = 1.0; grid.Ny   = 3
grid = build_grid(grid)
xc_str = grid.xc

[D,G,C,I,M]=build_ops(grid)

#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([grid.dof_xmax])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([grid.dof_f_xmax])   # identify faces on Neumann bnd
Param.qb = 1.0*np.ones((len(Param.dof_neu[0]),1))  # set flux at Neumann bnd
Param.g  = Param.qb*grid.xc[Param.dof_dir[0,0]-1]
#Param.g  = np.array([[0.0],[0.0]])                      # set head at Dirichlet bnd
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
fs = np.zeros([grid.N,1])                       # r.h.s. (zero)
L = -D@G                        # Laplacian
f = fs+fn

u_str = solve_lbvp(L,f,B,Param.g,N)                 # Solve linear boundary value problem

#plot


#plot
fig, ax= plt.subplots()
ax.plot(xc_unstr,u_unstr[::grid.Ny],'bo',label='Unstructured')
ax.plot(xc_str,u_str[::grid.Ny],'rX',label='Structured')
legend = ax.legend(loc='upper left', shadow=False, fontsize='x-large',frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('Head')

