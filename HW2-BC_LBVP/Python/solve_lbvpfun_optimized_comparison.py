# import python libraries
import numpy as np
#import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from scipy.linalg import solve as fullspsolve
import time
import matplotlib.pyplot as plt
from scikits.umfpack import spsolve, splu
# import personal libraries
#import build_gridfun
#import build_opsfun
#from build_bndfun import build_bnd
#from mobilityfun import mobility

def solve_lbvp(L,f,B,g,N):

    # author: Mohammad Afzal Shadab
    # date: 2/25/2020
    # Description
    # Computes the solution $u$ to the linear differential problem given by
    #
    # $$\mathcal{L}(u)=f \quad x\in \Omega $$
    #
    # with boundary conditions
    #
    # $$\mathcal{B}(u)=g \quad x\in\partial\Omega$$.
    #
    # Input:
    # L = matrix representing the discretized linear operator of size N by N, 
    #     where N is the number of degrees of freedom
    # f = column vector representing the discretized r.h.s. and contributions
    #     due non-homogeneous Neumann BC's of size N by 1
    # B = matrix representing the constraints arising from Dirichlet BC's of
    #     size Nc by N
    # g = column vector representing the non-homogeneous Dirichlet BC's of size
    #     Nc by 1.
    # N = matrix representing a orthonormal basis for the null-space of B and
    #     of size N by (N-Nc).
    # Output:
    # u = column vector of the solution of size N by 1
    if B.nnz == 0:
        u = linalg.spsolve(L, f)
    else:
        tp1 = time.perf_counter()
        up = np.transpose([sp.csr_matrix.transpose(B) @ linalg.spsolve((B @ sp.csr_matrix.transpose(B)),g)])
        tp2 = time.perf_counter()
        
        th1 = time.perf_counter() 
        
        tN1=time.perf_counter()
        Nnew= sp.csr_matrix.transpose(N) @ L @ N
        tN2=time.perf_counter()
        
        trhs1=time.perf_counter()
        rhs = sp.csr_matrix.transpose(N) @ (f-L @ up)
        trhs2=time.perf_counter()

        '''
        tu0r1np=time.perf_counter() 
        print(sp.issparse(Nnew.toarray()))
        u0r = fullspsolve(Nnew.toarray(),rhs)
        tu0r2np=time.perf_counter() 
        ''' 
        
        tu0r1umft=time.perf_counter() 
        print(sp.issparse(Nnew))
        u0r = spsolve(Nnew,rhs)
        tu0r2umft=time.perf_counter() 
        
        th2 = time.perf_counter()  

        tu0r1=time.perf_counter() 
        plt.spy(Nnew)
        print(sp.issparse(Nnew))
        linalg.use_solver(useUmfpack=True) # enforce superLU over UMFPACK
        u0r = linalg.spsolve(Nnew,rhs)
        tu0r2=time.perf_counter() 
       
        u0 = np.transpose([N @ u0r])
        th2 = time.perf_counter()  
        
        print('Time to calculate particular solution', tp2 - tp1)       
        
        print('Time to calculate new N', tN2 - tN1) 
        print('Time to calculate new rhs', trhs2 - trhs1) 
        #print('Time to calculate reduced homogeneous solution solve', tu0r2np - tu0r1np)
        print('Time to calculate reduced homogeneous solution umft spsolve', tu0r2umft - tu0r1umft)
        print('Time to calculate reduced homogeneous solution scipy spsolve', tu0r2 - tu0r1)
        print('Time to calculate homogeneous solution', th2 - th1)
        u = u0 + up

    return u;


'''
class grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx = []

class Param:
    def __init__(self):
        self.dof_dir = []       # identify cells on Dirichlet bnd
        self.dof_f_dir = []     # identify faces on Dirichlet bnd
        self.dof_neu = []       # identify cells on Neumann bnd
        self.dof_f_neu = []     # identify faces on Neumann bnd
        self.g = []             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
        self.qb = []            

#grid and operators
grid.xmin = 0.0
grid.xmax = 1.0
grid.Nx   = 10
build_gridfun.build_grid(grid)
[D,G,I]=build_opsfun.build_ops(grid)
#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([grid.dof_xmax])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([grid.dof_f_xmax])   # identify faces on Neumann bnd
Param.qb = np.array([1.0])                      # set flux at Neumann bnd
Param.g  = np.array([0.0])                      # set head at Dirichlet bnd
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
fs = np.zeros([grid.N,1])                       # r.h.s. (zero)
L = -np.mat(D)*np.mat(G)                        # Laplacian

f = fs+fn

u = solve_lbvp(L,f,B,Param.g,N)                 # Solve linear boundary value problem

#plot
fig, ax= plt.subplots()
ax.plot(grid.xc,u,'r-',label='u')
legend = ax.legend(loc='upper left', shadow=False, fontsize='x-large')
ax.set_xlabel('Position')
ax.set_ylabel('Head')
'''
