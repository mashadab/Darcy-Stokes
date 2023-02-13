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

def solve_Helmholtz(D,G,I,M,phi,n,m,Grid,B,N,fn,BC,Zc,Gamma): # class

    # author: Mohammad Afzal Shadab
    # date: 03/01/2022
    
    # Input: (to be completed)
    # D = N by Nf discrete divergence operator
    # G = Nf by N discrete gradient operator
    # I = N by N identity
    # phi = Nx by 1 matrix with porosity field
    # n = permeability exponent
    # m = compaction exponent
    # B = Nc by N constraint matrix
    # N = N by N-Nc basis for nullspace of constraint matrix
    # fn = r.h.s. vector for Neuman BC
    # BC = struture with info for boundary condtions
    # Zc = Nx by 1 matrix containing the vertical coordinate of cell centers
    # Gamma = N by 1 vector containing melting term
    
    # Output:
    # hD = N by 1 column vector of dimensionless overpressure heads
    # pD = N by 1 column vector of dimensionless overpressure
    # qD = Nf by 1 column vector of dimensionless relative fluid fluxes
    
    # Porosity matrices
    Phi_n = comp_mean(phi**n, M, -1, Grid, 1)
    Phi_m = sp.spdiags((phi**m).T,0,Grid.N,Grid.N)
    
    #Solve modified Helmholtz equation
    L  = - D @ Phi_n @ G + Phi_m
    fs =   Phi_m @ Zc + Gamma
    flux = lambda h: -Phi_n @ (G @ np.transpose([h]))
    res  = lambda h,cell: L[cell,:] @ np.transpose([h]) - fs[cell,:]
    
    #Solve boundary value problem
    hD =   solve_lbvp(L,fs+fn,B,BC.g,N)
    qD =   comp_flux_gen(flux,res,hD,Grid,BC)
    pD =   np.transpose([hD]) - Zc

    return hD,pD,qD;