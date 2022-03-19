
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
