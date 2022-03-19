
#Lecture 15 (March 12): 3D Python operator
#Author: Mohammad Afzal Shadab and Marc Hesse

#In two dimensions we will extensively use two functions for plotting: meshgrid and reshape

import sys
sys.path.insert(1,'../../HW1-Numerics/Python')

#import library and modules
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples

from classfun import *
from build_gridfun3D import build_grid3D


f = lambda x,y,z: x**2*y*z
g = lambda x,y,z: y
h = lambda x,y,z: z

Nx =4; Ny=3; Nz=2;
Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = Nx
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = Ny
Grid.zmin = 0; Grid.zmax = 1; Grid.Nz = Nz
Grid = build_grid3D(Grid)




