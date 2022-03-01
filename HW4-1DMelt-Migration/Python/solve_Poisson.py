# import python libraries
import numpy as np
import scipy.sparse as sp

import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')

from comp_mean_matrix import comp_mean
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen

def solve_Poisson(D,G,I,phi,m,pD,Grid,B,N,fn,BC): # class

    # author: you
    # date: before it's due
    
    # Input: 
    # D = N by Nf discrete divergence operator
    # G = Nf by N discrete gradient operator
    # I = N by N identity
    # phi = Ny by Nx matrix with porosity field
    # m = compaction exponent
    # pD = N by 1 column vector of dimensionless overpressures
    # Grid = structure containing useful info about the grid
    # B = Nc by N constraint matrix
    # N = N by N-Nc basis for nullspace of constraint matrix
    # fn = r.h.s. vector for Neuman BC
    # BC = struture with releavant info about problem parameters
    
    # Output:
    # uD = N  by 1 column vector of dimensionless solid velocity potentials
    # vD = Nf by 1 column vector of dimensionless solid velocities
    
    # Porosity matrices
    Phi_m = sp.spdiags(phi**m,0,Grid.N,Grid.N)
    
    #Solve modified Helmholtz equation
    L  = - D @ G
    fs =   Phi_m @ np.transpose([pD])
    flux = lambda u: -G @ np.transpose([u])
    res  = lambda u,cell: L[cell,:] @ np.transpose([u]) - fs[cell,:]
    
    #Solve boundary value problem
    uD =   solve_lbvp(L,fs+fn,B,BC.g,N)
    vD =   comp_flux_gen(flux,res,uD[:,0],Grid,BC)

    return uD,vD;