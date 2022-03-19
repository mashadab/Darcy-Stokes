
#Lecture 15 (March 12): 3D Python operator
#Author: Mohammad Afzal Shadab and Marc Hesse

#In two dimensions we will extensively use two functions for plotting: meshgrid and reshape

#import library and modules
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
##############################################################################
'''
The function meshgrid() takes three vectors x, y and z that contain the 
location of the grid points and generates matrices X, Y and Z that are used by 
all 3D Matlab plotting functions, in particular slice()
'''
##############################################################################
#Some functions

f = lambda x,y,z: x**2*y*z
g = lambda x,y,z: y
h = lambda x,y,z: z

Nx =4; Ny=3; Nz=2; N=Nx*Ny*Nz
x = np.linspace(0,1,Nx)
y = np.linspace(0,2,Ny)
z = np.linspace(0,3,Nz)
[X,Z,Y] = np.meshgrid(x,z,y)

