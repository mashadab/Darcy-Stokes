
import scipy.sparse as sp
import numpy as np

def build_ops(Grid):
    # author: Mohammad Afzal Shadab
    # date: 4/3/2026
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
    N  = Grid.N

    # Two dimensional divergence    
    #     Readable implementation
    #     # 2D divergence matrices
    
    if (Nx>1) and (Ny>1): #2D case
        #One diamentional divergence
        Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        
        #Two dimensional divergence
        Dy = sp.kron(np.eye(Nx), Dy) #y component Dy^2
        
        e  = np.array(np.ones(Ny*(Nx+1),'float64'))
        Dx = sp.spdiags(([-e , e])/np.asarray(Grid.dx),np.array([0,Ny]),N,(Nx+1)*Ny).tocsr()#.toarray() # 2D div-matrix in x-dir

        #D  = np.concatenate((Dx , Dy), axis=1)
        D  = sp.hstack([Dx , Dy]).tocsr()    
        dof_f_bnd = np.concatenate([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1])       # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)
        
    elif (Nx > 1) and (Ny == 1): #one dimensional in x direction
        D = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')),np.array(np.ones((Nx+1),'float64'))])/np.asarray(Grid.dx),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # 1D div-matrix in x-dir
        dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    elif (Nx == 1) and (Ny > 1): #one dimensional in y direction
        D = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')),np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # 1D div-matrix in y-dir
        dof_f_bnd = [Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    #print(D)
    G = -sp.csr_matrix.transpose(D)
    G =  zero_rows(G,dof_f_bnd)

    #Identity
    I = (sp.eye(Grid.N)).tocsr()
    
    #Curl matrix 
    C = []
    
    #Algebraic mean  
    if Grid.Nx>1 and Grid.Ny==1:
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))])         
        M      = Avg_x1.copy()
        
    elif Grid.Nx==1 and Grid.Ny>1:
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        M      = Avg_y1.copy()     
    
    elif Grid.Nx>1 and Grid.Ny>1:
        #Averaging in y-direction considering the zero-flux at boundary
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        Avg_y2 = sp.kron(sp.eye(Grid.Nx),Avg_y1)
    
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))]) 
        Avg_x2 = sp.kron(Avg_x1,sp.eye(Grid.Ny))         
        M      = sp.vstack([Avg_x2, Avg_y2])
    
    return D,G,C,I,M;



def build_ops_unstructured(Grid):
    # author: Mohammad Afzal Shadab
    # date: 04/03/2026
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
    N  = Grid.N

    # Two dimensional divergence    
    #     Readable implementation
    #     # 2D divergence matrices
    
    if (Nx>1) and (Ny>1): #2D case
        #One diamentional divergence
        Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))]),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        print(np.shape(Dy),np.shape(sp.kron(Grid.dy_scaled,np.ones((1,Ny+1)))))
        Dy = Dy/sp.kron(Grid.dy_scaled,np.ones((1,Ny+1)))
        
        #Two dimensional divergence
        Dy = sp.kron(np.eye(Nx), Dy) #y component Dy^2
        
        
        #One diamentional divergence - x
        Dx = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')) , np.array(np.ones((Nx+1),'float64'))]),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # Dx^1
    
        Dx = Dx/sp.kron(Grid.dx_scaled,np.ones((1,Nx+1)))        
        
        #Two dimensional divergence
        Dx = sp.kron(Dx,np.eye(Ny)) #y component Dx^2        
        
        #e  = np.array(np.ones(Ny*(Nx+1),'float64'))        
        #Dx = sp.spdiags(([-e , e]),np.array([0,Ny]),N,(Nx+1)*Ny).tocsr()#.toarray() # 2D div-matrix in x-dir

        #D  = np.concatenate((Dx , Dy), axis=1)
        D  = sp.hstack([Dx , Dy]).tocsr()    
        dof_f_bnd = np.concatenate([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1])       # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)
        
    elif (Nx > 1) and (Ny == 1): #one dimensional in x direction
        D  = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')) , np.array(np.ones((Nx+1),'float64'))]),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # Dx^1
        D  = D/sp.kron(Grid.dx_scaled,np.ones((Nx+1,1)))        
        dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    elif (Nx == 1) and (Ny > 1): #one dimensional in y direction
        D  = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))]),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        D  = D/sp.kron(Grid.dy_scaled,np.ones((Ny+1,1)))
        
        dof_f_bnd = [Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    #print(D)
    
    if (Nx > 1) and (Ny > 1): # 2D case
            # One dimensional gradient in y
            Gy = sp.spdiags(
                ([-np.array(np.ones((Ny),'float64')),
                   np.array(np.ones((Ny),'float64'))]),
                np.array([0,1]), Ny-1, Ny
            ).tocsr()
            Gy = Gy / np.kron(Grid.dyf_scaled[1:Ny], np.ones((1,Ny)))
            Gy = sp.vstack([np.zeros((1,Ny)), Gy, np.zeros((1,Ny))]).tocsr()

            # Two dimensional gradient
            Gy = sp.kron(np.eye(Nx), Gy).tocsr()
    
            # One dimensional gradient in x
            Gx = sp.spdiags(
                ([-np.array(np.ones((Nx),'float64')),
                   np.array(np.ones((Nx),'float64'))]),
                np.array([0,1]), Nx-1, Nx
            ).tocsr()
            Gx = Gx / np.kron(Grid.dxf_scaled[1:Nx], np.ones((1,Nx)))
            Gx = sp.vstack([np.zeros((1,Nx)), Gx, np.zeros((1,Nx))]).tocsr()
    
            # Two dimensional gradient
            Gx = sp.kron(Gx, np.eye(Ny)).tocsr()
    
            G  = sp.vstack([Gx, Gy]).tocsr()
    
    elif (Nx > 1) and (Ny == 1): # 1D in x
        G = sp.spdiags(
            ([-np.array(np.ones((Nx),'float64')),
               np.array(np.ones((Nx),'float64'))]),
            np.array([0,1]), Nx-1, Nx
        ).tocsr()
        G = G / np.kron(Grid.dxf_scaled[1:Nx], np.ones((1,Nx)))
        G = sp.vstack([np.zeros((1,Nx)), G, np.zeros((1,Nx))]).tocsr()

    elif (Nx == 1) and (Ny > 1): # 1D in y
        G = sp.spdiags(
            ([-np.array(np.ones((Ny),'float64')),
               np.array(np.ones((Ny),'float64'))]),
            np.array([0,1]), Ny-1, Ny
        ).tocsr()
        G = G / np.kron(Grid.dyf_scaled[1:Ny], np.ones((1,Ny)))
        G = sp.vstack([np.zeros((1,Ny)), G, np.zeros((1,Ny))]).tocsr()

    else: # Nx == 1 and Ny == 1
        G = sp.csr_matrix((2,1))

    
    G =  zero_rows(G,dof_f_bnd)

    #Identity
    I = (sp.eye(Grid.N)).tocsr()
    
    #Curl matrix 
    C = []
    
    #Algebraic mean  
    if Grid.Nx>1 and Grid.Ny==1:
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))])         
        M      = Avg_x1.copy()
        
    elif Grid.Nx==1 and Grid.Ny>1:
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        M      = Avg_y1.copy()     
    
    elif Grid.Nx>1 and Grid.Ny>1:
        #Averaging in y-direction considering the zero-flux at boundary
        Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
        Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
        Avg_y2 = sp.kron(sp.eye(Grid.Nx),Avg_y1)
    
        #Averaging in x-direction considering the zero-flux at boundary
        Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
        Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))]) 
        Avg_x2 = sp.kron(Avg_x1,sp.eye(Grid.Ny))         
        M      = sp.vstack([Avg_x2, Avg_y2])
    
    return D,G,C,I,M;


def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = sp.diags(ixs)
    res = D * M
    return res


class Grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx   = []

import matplotlib.pyplot as plt

def plot_matrix_compare(G_str, G_unstr, title1='Structured', title2='Unstructured'):
    A1 = G_str.toarray() if hasattr(G_str, 'toarray') else np.asarray(G_str)
    A2 = G_unstr.toarray() if hasattr(G_unstr, 'toarray') else np.asarray(G_unstr)

    vmin = min(A1.min(), A2.min())
    vmax = max(A1.max(), A2.max())

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, A, ttl in zip(axs, [A1, A2], [title1, title2]):
        im = ax.contourf(A, levels=30, vmin=vmin, vmax=vmax, cmap='coolwarm')
        if np.shape(G_str)[0]>np.shape(G_str)[1]: #Gradient
            plt.suptitle('Gradient',fontsize=50)
            ax.set_xlabel('Cell DOF')
            ax.set_ylabel('Face DOF')
        else: #Divergence
            plt.suptitle('Divergence',fontsize=50)
            ax.set_ylabel('Cell DOF')
            ax.set_xlabel('Face DOF')
        ax.set_title(ttl)    
        ax.invert_yaxis()

        # write numbers in each cell
        nr, nc = A.shape
        for i in range(nr):
            for j in range(nc):
                ax.text(j, i, f'{A[i,j]:.2g}', ha='center', va='center', color='k', fontsize=7)

    fig.colorbar(im, ax=axs, shrink=0.9, label='Value')
    plt.show()

#implementation
from build_gridfun2D import build_grid, build_grid_unstructured, plot_grid_unstructured
Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 4; Grid.scale_x = np.transpose([np.linspace(1,4,4)])
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = 3; Grid.scale_y = np.transpose([np.linspace(1,3,3)])

#plot grid structured
Grid_str = build_grid(Grid)
plot_grid_unstructured(Grid_str)
[D_str,G_str,C,I,M] = build_ops(Grid_str)
Gstr = G_str.toarray(); Dstr = D_str.toarray()

#plot grid unstructured
Grid_unstr = build_grid_unstructured(Grid)
plot_grid_unstructured(Grid_unstr)
[D_unstr,G_unstr,C,I,M] = build_ops_unstructured(Grid_unstr)
Gunstr = G_unstr.toarray(); Dunstr = D_unstr.toarray()

plot_matrix_compare(Gstr,Gunstr)
plot_matrix_compare(Dstr,Dunstr)
