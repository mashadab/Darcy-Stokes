import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D import build_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
        
#building grid and operators
Grid.xmin = 0; Grid.xmax = 50; Grid.Nx = 100

Grid = build_grid(Grid)
[D,G,C,I,M] = build_ops(Grid)

L    = -D @ G + I #Helmholtz operator
fs   = np.transpose([Grid.xc])
flux = lambda h: - G @ h
res  = lambda h,cell:   L[cell,:]  @ h- fs[cell,:]

#Boundary conditions
BC.dof_dir   = np.array([])
BC.dof_f_dir = np.array([]) 
BC.g         = np.array([]) 
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([]) 
BC.qb        = np.array([])

B,N,fn = build_bnd(BC,Grid,I)

#Solve LBVP
hD      = solve_lbvp(L, fs + fn, B, BC.g, N)

#Plot results
plt.figure(figsize=(10,7.5))
plt.semilogy(Grid.xc,hD)
plt.xlabel(r'$x$')
plt.ylabel(r'$h$')