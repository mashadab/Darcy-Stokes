#Lecture 15 (March 19): 3D Python operator implementation
#Author: Mohammad Afzal Shadab and Marc Hesse

import sys
sys.path.insert(1,'../../HW1-Numerics/Python')
sys.path.insert(1,'../../HW2-BC_LBVP/Python')

#import library and modules
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import time

from classfun import *
from build_gridfun3D import build_grid3D
from build_opsfun3D  import build_ops3D
from solve_lbvpfun_optimized import solve_lbvp
from build_bndfun_optimized import build_bnd
import scipy.sparse as sp

#################################################################################
#Solving the numerical problem
#################################################################################

Grid.xmin = 0; Grid.xmax = 3; Grid.Nx = 50
Grid.ymin = 0; Grid.ymax = 3; Grid.Ny = 50
Grid.zmin = 0; Grid.zmax = 3; Grid.Nz = 50

Grid = build_grid3D(Grid)
[D,G,C,I,M] = build_ops3D(Grid); L = - D*G
fs = np.zeros((Grid.N,1))

#boundary conditions
BC.dof_dir   = np.vstack([Grid.dof_xmin,Grid.dof_xmax])
BC.dof_f_dir = np.vstack([Grid.dof_f_xmin,Grid.dof_f_xmin])
BC.g         = np.vstack([np.ones_like(Grid.dof_xmin),np.zeros_like(Grid.dof_xmax)])

BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb        = np.array([])

[B,N,fn] = build_bnd(BC,Grid,I)

[X,Z,Y] = np.meshgrid(Grid.xc,Grid.zc,Grid.yc)
X_col = np.transpose([X.flatten()]);Y_col = np.transpose([Y.flatten()]);Z_col = np.transpose([Z.flatten()])

#source term
fs = np.zeros((Grid.N,1))
Xo = (Grid.xmax-Grid.xmin)/2; Yo=(Grid.ymax-Grid.ymin)/2; Zo=(Grid.zmax-Grid.zmin)/2
fs[((X_col-Xo)**2+(Y_col-Yo)**2+(Z_col-Zo)**2)<0.5] = 1.0 

t1 = time.perf_counter()
u = solve_lbvp(L, fs+fn, B, BC.g, N)
t2 = time.perf_counter()
print('Time elapsed',t2-t1)
U = u.reshape(Grid.Nz,Grid.Nx,Grid.Ny)


#####################################################################################
'''
Plotting
'''

# Create and plot structured grid
grid = pv.StructuredGrid(X,Y,Z)

# Add the data values to the cell data
grid.point_data["values"] = U.flatten(order="F")  # Flatten the array!

#grid.plot(show_edges=True,show_grid=True)

slices = grid.slice_orthogonal(x=1.5, y=1.5, z=1.5)
slices.plot(show_edges=True,show_grid=True)
plotter = pv.Plotter(off_screen=True)

plotter.add_mesh(slices, show_edges=True)
plotter.add_axes(line_width=5, labels_off=False)
plotter.show_grid()
#plotter.show(screenshot='heat_equation.png')
plotter.save_graphic("heat_equation.pdf") 

#####################################################################################
#plotting 
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,7))
fig.suptitle(f'Numerical results on centerlines')

ax1.plot(Grid.xc,U[int(Grid.Ny/2),:,int(Grid.Nz/2)],'r-',label=r'$Numerical$')
ax1.set_ylabel('$d2f$')
ax1.set_xlabel('$x$')

ax2.plot(Grid.xc,U[:,int(Grid.Nx/2),int(Grid.Nz/2)],'r-',label=r'$Numerical$')
ax2.set_ylabel('$d2f$')
ax2.set_xlabel('$y$')

ax3.plot(Grid.xc,U[int(Grid.Ny/2),int(Grid.Nx/2),:],'r-',label=r'$Numerical$')
ax3.set_ylabel('$d2f$')
ax3.set_xlabel('$z$')


plt.tight_layout()
