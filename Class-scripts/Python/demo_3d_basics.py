
#Lecture 15 (March 10): 3D Python basics
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

print('X:',X); print('Y:',Y); print('Z:',Z); print(X.shape)

print('Flatten X:',X.flatten()); 
print('Flatten Y:',Y.flatten()); 
print('Flatten Z:',Z.flatten()); 

print('Y first, X second and Z third')

arr = np.arange(27).reshape(3,3,3)

#####################################################################################
'''
In the matrices X and Y, the -value increases with the row index, , and the 
-value increases with the column index,  and k  in the third direction.  
Since we index matrices as X(k,i,j) and Y(k,j,i), and Z(k,i,j), the first index 
is the -coordinate. This makes it natural to order our grid y-first - see below! 
'''

# Create and plot structured grid
grid = pv.StructuredGrid(X,Y,Z)

# Add the data values to the cell data
grid.point_data["values"] = f(X,Y,Z).flatten(order="F")  # Flatten the array!

grid.plot(show_edges=True,show_grid=True)

# Add the data values to the cell data
grid.point_data["values"] = g(X,Y,Z).flatten(order="F")  # Flatten the array!

grid.plot(show_edges=True,show_grid=True)



#####################################################################################
#Reshaping

print(X)
print("Flatten X (first way):",np.transpose([X.flatten()]))
print("Flatten X (second way):",np.reshape(X,(N,-1)))


'''
Of course reshape() is the more general, it allows you to transform X into 
any matrix or vector with the same number of elements

1) From vector to matrix
Suppose the solution is given by g = g(x)
'''

print('Norm for reshaped x-coordinate',np.linalg.norm(X - np.reshape(X.flatten(),(Nz,Nx,Ny))))

soln = g(np.transpose([X.flatten()]),np.transpose([Y.flatten()]),np.transpose([Z.flatten()]))

SOLN = soln.reshape(Nz,Nx,Ny)

print('Norm for reshaped solution',np.linalg.norm(g(X,Y,Z)-SOLN))