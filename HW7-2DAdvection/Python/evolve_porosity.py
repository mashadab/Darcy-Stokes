# import python libraries
import scipy.sparse as sp
import numpy as np

import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')

from solve_lbvpfun_optimized import solve_lbvp
from flux_upwindfun2D_optimized import flux_upwind

def evolve_porosity(D,I,phiD,vD,pD,B,N,BC,Grid,phi_c,theta,dtD): # function

    # author: Mohammad Afzal Shadab, Marc Hesse
    # date: 31 March 2021
    
    # Input:
    # D = N by Nf discrete divergence operator
    # G = Nf by N discrete gradient operator
    # I = N by N identity
    # phiD = N by 1 column vector of scaled porosities
    # vD = Nf by 1 column vector of dimensionless solid velocities
    # pD = N by 1 column vector of dimensionless pressures
    # B = Nc by N constraint matrix
    # N = N by N-Nc basis for nullspace of constraint matrix
    # fn = r.h.s. vector for Neuman BC
    # Grid = structure containing info about grid
    # BC = struture with releavant info about problem parameters
    # Scales = stucture containg all characteristic scalse
    # theta = variable determining time integration (1 = FE, .5 = CN, 0 = BE)
    # dtD = dimensionless timestep
    
    # Output:
    # phiD = N by 1 column vector of dimensionless porosities
    # Av = Nf by N upwinf flux matrix
    
    
    #Solve porosity evolution equation
    Av  = flux_upwind(vD, Grid)
    L   = phi_c * D @ Av - sp.spdiags(pD, 0, Grid.N, Grid.N) 
    IM  = I + dtD * (1-theta) * L
    EX  = I - dtD * theta * L
    phiD= solve_lbvp(IM,EX@phiD,B,BC.g,N)

    if np.max(phiD*phi_c) > 1 or np.min(phiD*phi_c) < 0:
        raise Exception("Porosity outside  [0 1].\n")
    return phiD,Av;