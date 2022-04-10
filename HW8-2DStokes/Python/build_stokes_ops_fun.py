# Stokes solver building operator
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 02/12/2021

import numpy as np
from build_opsfun2D_optimized import build_ops
from scipy.sparse import eye, bmat
import matplotlib.pyplot as plt

def build_stokes_ops(Grid):
    
    #Operator for pressure Grid, x-velocity, and y-velocity.
    [Dp ,Gp, Ip]  = build_ops(Grid.p)     
    [DVx,GVx,IVx] = build_ops(Grid.Vx) 
    [DVy,GVy,IVy] = build_ops(Grid.Vy)  

    #Extracting the gradient operator
    GVxx = GVx[0:Grid.Vx.Nfx,:] ; GVxy = GVx[Grid.Vx.Nfx:Grid.Vx.Nf,:] 
    GVyx = GVy[0:Grid.Vy.Nfx,:] ; GVyy = GVy[Grid.Vy.Nfx:Grid.Vy.Nf,:]     

    #Extracting the divergence operator
    DVxx = DVx[:,0:Grid.Vx.Nfx] ; DVxy = DVx[:,Grid.Vx.Nfx:Grid.Vx.Nf] 
    DVyx = DVy[:,0:Grid.Vy.Nfx] ; DVyy = DVy[:,Grid.Vy.Nfx:Grid.Vy.Nf]     
   
    #Making the divergence operator D
    D = bmat([[DVxx, None, DVxy], [None, DVyy, DVyx]],format="csr")
    
    #Making the Edot tensor
    Edot = bmat([[GVxx, None], [None, GVyy], [0.5*GVxy , 0.5*GVyx]],format="csr")    
    
    #Making the I operator
    I = eye(Grid.p.Nf+Grid.p.N , dtype=np.float64,format="csr")
    
    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(20,7))
    ax1.set_title(f'DVxx, nz={DVxx.nnz}')
    ax1.spy(DVxx)

    ax2.set_title(f'DVxy, nz={DVxy.nnz}')
    ax2.spy(DVxy)

    ax3.set_title(f'GVxx, nz={GVxx.nnz}')
    ax3.spy(GVxx)

    ax4.set_title(f'GVxy, nz={GVxy.nnz}')
    ax4.spy(GVxy)
    '''
    
    
    return D, Edot, Dp, Gp, I;