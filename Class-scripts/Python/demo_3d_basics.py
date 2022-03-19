
#Lecture 15 (March 10): 3D Python basics
#Author: Mohammad Afzal Shadab and Marc Hesse

'#In two dimensions we will extensively use two functions for plotting: meshgrid and reshape'

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

Nx =4; Ny=3; Nz=2
x = np.linspace(1,Nx,Nx)
y = np.linspace(Nx+1,Nx+Ny+1,Ny)
z = np.linspace(Nx+Ny+1,Nx+Ny+Nz+1,Nz)
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
Since we index matrices as X(k,j,i) and Y(k,j,i), and Z(k,j,i), the first index 
is the -coordinate. This makes it natural to order our grid y-first - see below! 
'''

# Create and plot structured grid
grid = pv.StructuredGrid(X,Y,Z)

# Add the data values to the cell data
grid.point_data["values"] = g(X,Y,Z).flatten(order="F")  # Flatten the array!

grid.plot(show_edges=True,show_grid=True)






'''
#parameters
HD = 0.1  #dimensionless domain length

zf = np.linspace(0,HD,1000) #vertical location

#plotting 
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,7))
fig.suptitle(f'Analytic solutions to HD = {HD}')

ax1.plot(hD(zf,HD),zf,'r-',label=r'$h_D$')
ax1.plot(uD(zf,HD)+HD,zf,'b-',label=r'$u_D$')
ax1.set_ylabel('$z_D$')
ax1.set_xlabel('$h_D,u_D$')
ax1.legend(loc = 'best')

ax2.plot(qD(zf,HD),zf,'r-',label=r'$q_D$')
ax2.plot(vD(zf,HD),zf,'b-',label=r'$v_D$')
ax2.set_xlabel('$q_D,v_D$')
ax2.legend(loc = 'best')

ax3.plot(pD(zf,HD),zf,'r-')
ax3.set_xlabel('$p_D$')

plt.tight_layout()


#####################################################################################
# B. When compaction length delta is smaller than domain length H, HD= H/delta >> 1
#####################################################################################

#parameters
HD = 20  #dimensionless domain length

zf = np.linspace(0,HD,1000) #vertical location

#plotting 
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,7))
fig.suptitle(f'Analytic solutions to HD = {HD}')

ax1.plot(hD(zf,HD),zf,'r-',label=r'$h_D$')
ax1.plot(uD(zf,HD)+HD,zf,'b-',label=r'$u_D$')
ax1.set_ylabel('$z_D$')
ax1.set_xlabel('$h_D,u_D$')
ax1.legend(loc = 'best')

ax2.plot(qD(zf,HD),zf,'r-',label=r'$q_D$')
ax2.plot(vD(zf,HD),zf,'b-',label=r'$v_D$')
ax2.set_xlabel('$q_D,v_D$')
ax2.legend(loc = 'best')

ax3.plot(pD(zf,HD),zf,'r-')
ax3.set_xlabel('$p_D$')

plt.tight_layout()


#####################################################################################
# Bonus. Effect of changing domain length HD= H/delta on centerline qD and vD
#####################################################################################

#parameters
HD = np.logspace(-5,2,1000) #dimensionless domain length vector

#vertical location of the evaluation of flux is center of the domain
#plotting 
plt.figure(figsize=(12,8))
fig.suptitle(r'Effect of domain length on centerline fluxes: $q_D,v_D$')

plt.plot(HD,qD(HD/2,HD),'r-',label=r'$q_D$')
plt.plot(HD,vD(HD/2,HD),'b-',label=r'$v_D$')
plt.ylabel('$q_D,v_D$')
plt.xlabel('$HD$')
plt.legend(loc = 'best')

plt.tight_layout()
'''