import scipy.sparse as sp
import numpy as np

def build_ops3D(Grid):
    # author: Mohammad Afzal Shadab
    # date: 1/27/2020
    # description:
    # This function computes the discrete divergence and gradient matrices on a
    # regular staggered grid using central difference approximations. The
    # discrete gradient assumes homogeneous boundary conditions.
    # Input:
    # Grid = structure containing all pertinent information about the grid.
    # Output:
    # D = discrete divergence matrix
    # G = discrete gradient matrix
    # I = identity matrix

    Nx = Grid.Nx
    Ny = Grid.Ny
    Nz = Grid.Nz
    N  = Grid.N

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

    C = np.array([])
    I = (sp.eye(Grid.N)).tocsr()
    M = ((1.0*(np.abs(G))>0))
    M = 0.5*M

    return D,G,C,I,M;

def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = sp.diags(ixs)
    res = D * M
    return res