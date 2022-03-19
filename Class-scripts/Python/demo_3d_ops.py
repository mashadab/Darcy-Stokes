
#Lecture 15 (March 12): 3D Python operator
#Author: Mohammad Afzal Shadab and Marc Hesse

#In two dimensions we will extensively use two functions for plotting: meshgrid and reshape

import sys
sys.path.insert(1,'../../HW1-Numerics/Python')

#import library and modules
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples

from classfun import *
from build_gridfun3D import build_grid3D
from build_opsfun3D  import build_ops3D
import scipy.sparse as sp


def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = sp.diags(ixs)
    res = D * M
    return res

Nx =4; Ny=3; Nz=2
Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = Nx
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = Ny
Grid.zmin = 0; Grid.zmax = 1; Grid.Nz = Nz
Grid = build_grid3D(Grid)


#Build Identities
Ix = sp.eye(Nx); Iy = sp.eye(Ny); Iz = sp.eye(Nz)

  
#Three dimensional divergence
Dx = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')) , np.array(np.ones((Nx+1),'float64'))])/np.asarray(Grid.dx),np.array([0,1]),Nx,Nx+1).tocsr() # Dx^1
Dx = sp.kron(Iz,sp.kron(Dx, Iy)) #x component Dz^3

Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).tocsr() # Dy^1
Dy = sp.kron(Iz,sp.kron(Ix, Dy)) #y component Dy^3

Dz = sp.spdiags(([-np.array(np.ones((Nz+1),'float64')) , np.array(np.ones((Nz+1),'float64'))])/np.asarray(Grid.dz),np.array([0,1]),Nz,Nz+1).tocsr() # Dz^1
Dz = sp.kron(Dz,sp.kron(Ix, Iy)) #y component Dz^3

D  = sp.hstack([Dx , Dy, Dz]).tocsr()   

# Gradient
# Note this is only true in cartesian coordinates!
# For more general coordinate systems it is worth
# assembling G and D seperately.
dof_f_bnd = np.concatenate(np.array([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1,Grid.dof_f_zmin-1, Grid.dof_f_zmax-1]), axis=0 )# boundary faces
dof_f_bnd = np.transpose(dof_f_bnd)


G = -sp.csr_matrix.transpose(D)
G =  zero_rows(G,dof_f_bnd)

#Identity
I = (sp.eye(Grid.N)).tocsr()
M = ((1.0*(np.abs(G))>0))
M = 0.5*M

#plotting 
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,7))
fig.suptitle(f'Divergence for Nx = {Grid.Nx}, Ny = {Grid.Ny} and Nz = {Grid.Nz}')
ax1.set_title('Dx')
ax1.spy(Dx)
ax2.set_title('Dy')
ax2.spy(Dy)
ax3.set_title('Dz')
ax3.spy(Dz)
plt.tight_layout()


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,7))
fig.suptitle(f'Gradient for Nx = {Grid.Nx}, Ny = {Grid.Ny} and Nz = {Grid.Nz}')
ax1.set_title('Gx')
ax1.spy(G[0:Grid.Nfx,:])
ax2.set_title('Gy')
ax2.spy(G[Grid.Nfx:Grid.Nfx+Grid.Nfy,:])
ax3.set_title('Gz')
ax3.spy(G[Grid.Nf-Grid.Nfz:Grid.Nf,:])
plt.tight_layout()

L = -D @ G
fig = plt.figure(figsize=(10,10))
fig.suptitle(f'Laplacian for Nx = {Grid.Nx}, Ny = {Grid.Ny} and Nz = {Grid.Nz}')
plt.spy(L)


#################################################################################
#Testing the discrete divergence and gradients
#################################################################################

Grid.xmin = 0; Grid.xmax = 3; Grid.Nx = 100
Grid.ymin = 0; Grid.ymax = 3; Grid.Ny = 100
Grid.zmin = 0; Grid.zmax = 3; Grid.Nz = 100

Grid = build_grid3D(Grid)
[D,G,C,I,M] = build_ops3D(Grid); L = - D*G

f   = lambda x,y,z: np.exp(np.cos(2*np.pi*x))*np.exp(np.cos(2*np.pi*y))*np.exp(np.cos(2*np.pi*z))
d2f = lambda x,y,z: -2*np.pi**2*f(x,y,z)*[ 2*np.cos(2*np.pi*x) + np.cos(4*np.pi*x) + \
                                           2*np.cos(2*np.pi*y) + np.cos(4*np.pi*y) + \
                                           2*np.cos(2*np.pi*z) + np.cos(4*np.pi*z) - 3]

[X,Z,Y] = np.meshgrid(Grid.xc,Grid.zc,Grid.yc)

soln = f(np.transpose([X.flatten()]),np.transpose([Y.flatten()]),np.transpose([Z.flatten()]))

SOLN = soln.reshape(Grid.Nz,Grid.Nx,Grid.Ny)

print('Norm for reshaped solution',np.linalg.norm(f(X,Y,Z)-SOLN))


laplacian_numerical = D @ G @ f(np.transpose([X.flatten()]),np.transpose([Y.flatten()]),np.transpose([Z.flatten()]))
laplacian_numerical = laplacian_numerical.reshape(Grid.Nz,Grid.Nx,Grid.Ny)

laplacian_analytical = d2f(np.transpose([X.flatten()]),np.transpose([Y.flatten()]),np.transpose([Z.flatten()]))
laplacian_analytical = laplacian_analytical.reshape(Grid.Nz,Grid.Nx,Grid.Ny)

print('Norm for reshaped laplacian',np.linalg.norm(laplacian_analytical-laplacian_numerical))


#####################################################################################
'''
In the matrices X and Y, the -value increases with the row index, , and the 
-value increases with the column index,  and k  in the third direction.  
Since we index matrices as X(k,i,j) and Y(k,j,i), and Z(k,i,j), the first index 
is the -coordinate. This makes it natural to order our grid y-first - see below! 
'''

# Create and plot structured grid
grid = pv.StructuredGrid(X,Y,Z)

# Add the data values to the cell data
grid.point_data["values"] = laplacian_numerical.flatten(order="F")  # Flatten the array!

grid.plot(show_edges=True,show_grid=True)

# Add the data values to the cell data
grid.point_data["values"] = laplacian_analytical.flatten(order="F")  # Flatten the array!

grid.plot(show_edges=True,show_grid=True)

#####################################################################################
#plotting 
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,7))
fig.suptitle(f'Analytic vs numerical on centerlines')

ax1.plot(Grid.xc,laplacian_analytical[int(Grid.Ny/2),:,int(Grid.Nz/2)],'r-',label=r'$Analytical$')
ax1.plot(Grid.xc,laplacian_numerical[int(Grid.Ny/2),:,int(Grid.Nz/2)],'bo',label=r'$Numerical$')
ax1.set_ylabel('$d2f$')
ax1.set_xlabel('$x$')
ax1.legend(loc = 'best')

ax2.plot(Grid.xc,laplacian_analytical[:,int(Grid.Nx/2),int(Grid.Nz/2)],'r-',label=r'$Analytical$')
ax2.plot(Grid.xc,laplacian_numerical[:,int(Grid.Nx/2),int(Grid.Nz/2)],'bo',label=r'$Numerical$')
ax2.set_ylabel('$d2f$')
ax2.set_xlabel('$y$')

ax3.plot(Grid.xc,laplacian_analytical[int(Grid.Ny/2),int(Grid.Nx/2),:],'r-',label=r'$Analytical$')
ax3.plot(Grid.xc,laplacian_numerical[int(Grid.Ny/2),int(Grid.Nx/2),:],'bo',label=r'$Numerical$')
ax3.set_ylabel('$d2f$')
ax3.set_xlabel('$z$')


plt.tight_layout()
