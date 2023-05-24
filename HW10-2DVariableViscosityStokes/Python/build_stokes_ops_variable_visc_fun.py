# Stokes solver building operator
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 02/12/2023

import sys
sys.path.insert(1, '../../HW6-2D_operators/Python/')

import numpy as np
from build_opsfun2D_latest import build_ops
from scipy.sparse import eye, bmat, vstack, kron
import matplotlib.pyplot as plt

def build_stokes_ops_variable_visc(Grid):
    
    #Operator for pressure Grid, x-velocity, and y-velocity.
    [Dp ,Gp,  _, Ip,  Mp] = build_ops(Grid.p)     
    [DVx,GVx, _, IVx, MVx] = build_ops(Grid.Vx) 
    [DVy,GVy, _, IVy, MVy] = build_ops(Grid.Vy)  

    #Extracting the gradient operator 
    GVx = GVx.tocsr(); GVy = GVy.tocsr(); 
    GVxx = GVx[0:Grid.Vx.Nfx,:] ; GVxy = GVx[Grid.Vx.Nfx:Grid.Vx.Nf,:] 
    GVyx = GVy[0:Grid.Vy.Nfx,:] ; GVyy = GVy[Grid.Vy.Nfx:Grid.Vy.Nf,:]     

    DVx = DVx.tocsr(); DVy = DVy.tocsr();
    #Extracting the divergence operator
    DVxx = DVx[:,0:Grid.Vx.Nfx] ; DVxy = DVx[:,Grid.Vx.Nfx:Grid.Vx.Nf] 
    DVyx = DVy[:,0:Grid.Vy.Nfx] ; DVyy = DVy[:,Grid.Vy.Nfx:Grid.Vy.Nf] 

    #Extracting the corner Mean operator 
    MVx = MVx.tocsr(); MVy = MVy.tocsr(); Mp = Mp.tocsr(); 
    Mpx  = Mp[0:Grid.p.Nfx,:]   ; Mpy  = Mp[Grid.p.Nfx:Grid.p.Nf,:] 
    MVxx = MVx[0:Grid.Vx.Nfx,:] ; MVxy = MVx[Grid.Vx.Nfx:Grid.Vx.Nf,:] 
    MVyx = MVy[0:Grid.Vy.Nfx,:] ; MVyy = MVy[Grid.Vy.Nfx:Grid.Vy.Nf,:]      
   
    Mc  = MVxy @ Mpx #Corner mean operator
    
    Ix = eye(Grid.p.Nx , dtype=np.float64,format="csr")
    Iy = eye(Grid.p.Ny , dtype=np.float64,format="csr")
    Ixx = vstack((Ix[0,:],Ix,Ix[-1,:]))
    Iyy = vstack((Iy[0,:],Iy,Iy[-1,:]))   
    
    Ixx = kron(Ixx,Iy)
    Iyy = kron(Ix,Iyy) 
    
    Ms  = vstack((Ixx,Iyy,Mc))
   
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
    
    
    return D, Edot, Dp, Gp, I, Ms;