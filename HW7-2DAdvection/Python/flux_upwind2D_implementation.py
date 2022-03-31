import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')


# import python libraries
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D import build_ops
from flux_upwindfun2D_optimized import flux_upwind
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen


Grid.xmin = 0; Grid.xmax=1; Grid.Nx = 4
Grid.ymin = 0; Grid.ymax=1; Grid.Ny = 3
Grid = build_grid(Grid)
[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
[D,G,C,I,M] = build_ops(Grid)

L = -D @ G
fs = np.zeros((Grid.N,1))

flux = lambda u: - G @ u
res  = lambda u,cell: L[cell,:] @ u - fs[cell,:]

#Boundary conditions
BC.dof_dir   = np.hstack([1,Grid.dof_xmin[1:Grid.Ny],Grid.dof_ymin[1:Grid.Nx] ])
BC.dof_f_dir = np.hstack([Grid.dof_f_xmin,Grid.dof_f_ymin[1:Grid.Nx]])
BC.g         = np.hstack([0.5,np.ones(Grid.Ny-1),np.zeros(Grid.Nx-1)])
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb        = np.array([])

[B,N,fn]     = build_bnd(BC, Grid, I)

u = solve_lbvp(L, fs+fn, B, BC.g, N)
v = comp_flux_gen(flux, res, u, Grid, BC)

A = flux_upwind(v, Grid)