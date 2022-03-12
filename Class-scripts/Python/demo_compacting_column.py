
#Lecture 12 (Feb 24): Solving the flow problem for an Instanteneous Compacting Column
#Author: Mohammad Afzal Shadab and Marc Hesse

"""
Here we consider the simplest non-trivial solution for the instantaneous compaction problem. 
Consider a column of hight  with uniform porosity that is closed at both the top and the bottom. 

For now we are just interested in the instantaneous solution for the flow problem. 
Choosing the compaction length delta and the initial porosity phi0, as characteristic scales the dimensionless flow problem simplifies


These equations are linear and second depends on the solution of the first. 
We are interested in a one-dimensional solution so that we have to solve the ODE's


The boundary conditions are no flow for both the solid and the melt: zero derivative for uD and vD
    
The only dimensionless governing parameter in this problem is the dimensionles domain height, HD.


Analytic solution
Equation 1 (mod. Helmholtz)
This equation is a non-homogeneous 2nd-order equation with constant coefficients. 
This can be solved decomposing the solution into a homogeneous and a particular solution, . 
The form of the homogeneous solution is determined by substituting  into the associated homogeneous equation. 
This shows that the homogeneous solution takes the exponential form

The particular solution hP takes the fomr of a polynominal of the same degree as the r.h.s.

Substituting into the equations and using the boundary conditions to determine the coefficients in the homogeneous solution hH
we have the dimensionless solution for the overpressure head hD and the associated Darcy flux qD.
The dimensionless over pressure is given by pD = hD - zD.
"""

#import library and modules
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#Analytic solutions
##############################################################################

#dimensionless fluid overpressure head
hD = lambda zD,HD: zD + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                        (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                        
#dimensionless relative volumetric flux of fluid w.r.t. solid velocity: qD = - d hD/ dzD 
qD = lambda zD,HD: -1 - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                        (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                        
#dimensionless fluid pressure
pD = lambda zD,HD: hD(zD,HD) - zD

#dimensionless solid velocity potential, setting c4 = 0
uD = lambda zD,HD: -zD - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                         (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)

#dimensionless solid velocity vD = - d uD/d zD 
vD = lambda zD,HD:  1  + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                         (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)

#####################################################################################
# A. When compaction length delta is greater than domain length H, HD= H/delta << 1
#####################################################################################

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