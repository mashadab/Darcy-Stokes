# Stokes solver building operator
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 02/12/2021

import numpy as np
from build_opsfun2D_optimized import build_ops
from scipy.sparse import eye, bmat

def build_stokes_ops(grid):
    
    #Operator for pressure grid, x-velocity, and y-velocity.
    [Dp ,Gp, Ip]  = build_ops(grid.p)     
    [DVx,GVx,IVx] = build_ops(grid.Vx) 
    [DVy,GVy,IVy] = build_ops(grid.Vy)  

    #Extracting the gradient operator
    GVxx = GVx[0:grid.p.Nfx,:] ; GVxy = GVx[grid.p.Nfx:grid.p.Nf,:] 
    GVyx = GVy[0:grid.p.Nfx,:] ; GVyy = GVy[grid.p.Nfx:grid.p.Nf,:]     

    #Extracting the divergence operator
    DVxx = DVx[:,0:grid.p.Nfx] ; DVxy = DVx[:,grid.p.Nfx:grid.p.Nf] 
    DVyx = DVy[:,0:grid.p.Nfx] ; DVyy = DVy[:,grid.p.Nfx:grid.p.Nf]     
   
    #Making the divergence operator D
    D = bmat([[DVxx, None, DVxy], [None, DVyy, DVyx]],format="csr")
    
    #Making the Edot tensor
    Edot = bmat([[GVxx, None], [None, GVyy], [0.5*GVxy , 0.5*GVyx]],format="csr")    
    
    #Making the I operator
    I = eye(grid.p.Nf+grid.p.N , dtype=np.float64,format="csr")
    
    return D, Edot, Dp, Gp, I;