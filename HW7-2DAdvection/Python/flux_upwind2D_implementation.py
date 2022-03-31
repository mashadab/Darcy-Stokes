import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')

# import python libraries
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D import build_ops
from flux_upwindfun2D_optimized import flux_upwind


Grid.xmin = 0; Grid.xmax=1; Grid.Nx = 10
Grid.ymin = 0; Grid.ymax=1; Grid.Ny = 10
Grid = build_grid(Grid)
[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
[D,G,C,I,M] = build_ops(Grid)


