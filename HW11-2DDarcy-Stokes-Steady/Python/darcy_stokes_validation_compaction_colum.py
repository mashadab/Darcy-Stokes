# Darcy-Stokes solver
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 04/09/2023

import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')
sys.path.insert(1, '../../HW8-2DStokes/Python/')
sys.path.insert(1, '../../HW9-Streamlines/Python/')

# import Python libraries
from scipy.sparse import bmat,csr_matrix,spdiags
import matplotlib.pyplot as plt

# import personal libraries
from classfun import *  #importing the classes and relevant functions
from build_stokes_grid_fun import build_stokes_grid
from build_stokes_ops_Darcy_Stokes_fun import build_stokes_ops_Darcy_Stokes
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from quiver_plot import quiver_plot
from comp_streamfun import comp_streamfun
from plot_streamlines import plot_streamlines

def build_Zd(G,phi,m,mu,Grid): #building zeta^*_phi at cell centers
    Zd = (G / (phi**m) - 2/3) * mu * (1-phi);
    Zd = spdiags(Zd.T,0,Grid.N,Grid.N)
    return Zd

def build_Kd(k0,n,phi,mu_f,Grid,Mp): #building Kd at cell faces
    Kd = Mp @ ( k0 * phi**n / mu_f)
    Kd = spdiags(Kd.T,0,Grid.Nf,Grid.Nf)
    return Kd 

def build_RHS(phi,Kd,Grid,Mp,Dp,rho_f,rho_s,Gamma,grav):
    #fv at cell faces
    fv =-grav*( Mp @ ((rho_f - rho_s) *(1-phi))) * np.vstack([np.zeros((Grid.Nfx,1)),np.ones((Grid.Nfy,1))])
    
    #fh at cell centers
    fh =-(rho_f - rho_s)/(rho_f * rho_s) * Gamma * np.ones((Grid.N,1))
    
    F =  np.vstack([fv,fh])
    
    return F

#problem parameter
mu_max = 1e14 #Maximum solid viscosity [Pa.s]
phi_min= 1e-2; phi_max= 0.7 #Minimum and Maximum solid porosities [-]
G      = 1.0 #Coefficient in the bulk viscosity formulation [-]
k0     = 5.6e-11 #Relative permeability [m^2] {Meyer and Hewitt, 2017}
mu_f   = 1e-3#Viscosity of the fluid [Pa.s]
m      = 1   #Power law coefficient in compaction viscosity = (G/phi^m * mu_s) [-]
n      = 2   #Power law coefficient in porosity permeability relationship k = k0*phi^n [-]
rho_s  = 917 #Density of solid [kg/m^3]
rho_f  = 1e3 #Density of fluid [kg/m^3]
Gamma  = 0   #Rate of melting [kg/m^3-s]
grav   = 9.81#Acceleration due to gravity []m/s^2]
Delta_rho = rho_f - rho_s #Density difference between the two phases [kg/m^3]
vt     = 1e-3#Tangential velocity [m/s]

#Compaction length
phic = phi_min
delta0 = np.sqrt(k0*phic**n*mu_max/(phic**m*mu_f))
Kc = k0*Delta_rho*grav*phic**n/mu_f

#building grid
Gridp.xmin = 0.0*delta0 ; Gridp.xmax = 1*delta0; Gridp.Nx   = 5
Gridp.ymin = 0.0*delta0 ; Gridp.ymax = 1*delta0; Gridp.Ny   = 100
Gridp.geom = 'cartesian'
Grid = build_stokes_grid(Gridp)
[Xc,Yc] = np.meshgrid(Gridp.xc,Gridp.yc)
Xc_col = np.reshape(Xc.T,(Gridp.N,-1)); Yc_col = np.reshape(Yc.T,(Gridp.N,-1))

#Analytic solution
HD = Grid.p.ymax/delta0
zDa = np.linspace(0,HD,num=1000)
##############################################################################
#Analytic solutions
##############################################################################
class analytical:
    #dimensionless fluid overpressure head
    hD = lambda zD,HD: zD + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                            (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                            
    #dimensionless relative volumetric flux of fluid w.r.t. solid velocity: qD = - d hD/ dzD 
    qD = lambda zD,HD: -1 - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                            (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
                            
    #dimensionless fluid pressure pD = hD - zD
    pD = lambda zD,HD: (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) + \
                       (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
    
    #dimensionless solid velocity potential, setting c4 = 0
    uD = lambda zD,HD: -zD - (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                             (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)
    
    #dimensionless solid velocity vD = - d uD/d zD 
    vD = lambda zD,HD:  1  + (np.exp(-HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp( zD) - \
                             (np.exp( HD) - 1)/(np.exp(HD) - np.exp(-HD))* np.exp(-zD)

#Initial condition
mu = mu_max  * np.ones_like(Yc_col)  #Constt viscosity
phi= phi_min * np.ones((Grid.p.N,1)) #+ (phi_max - phi_min)*(Yc(:)/Grid.p.ymax);  #Constt porosity or Decays with depth


#simulation name
simulation_type = 'inst_comp_column' #'lid_driven_cavity_flow_with_no_slip'   #lid_driven_cavity_flow_with_slip or 'no_flow' 
simulation_name = f'{simulation_type}_domain{Gridp.xmax-Gridp.xmin}by{Gridp.ymax-Gridp.ymin}_N{Gridp.Nx}by{Gridp.Ny}'

#building operators
D, Edot, Dp, Gp, I, Ms, Mp = build_stokes_ops_Darcy_Stokes(Grid)
Mud = spdiags((Ms @ (mu*(1-phi))).T,0,np.shape(Edot)[0],np.shape(Edot)[0])
Zd = build_Zd(G,phi,m,mu,Grid.p)
Kd = build_Kd(k0,n,phi,mu_f,Grid.p,Mp)

A  = 2.0 * D @ Mud @ Edot
L  = bmat([[A + Gp @ Zd @ Dp, -(Delta_rho * grav)* Gp], [Dp, -(Delta_rho * grav)* Dp @ Kd @ Gp]],format="csr")
#fs = csr_matrix((Grid.N, 1), dtype=np.float64)
fs = build_RHS(phi,Kd,Grid.p,Mp,Dp,rho_f,rho_s,Gamma,grav)


#Boundary conditions
if 'lid_driven_cavity_flow_with_slip' in simulation_type:
    BC.dof_dir =  np.concatenate((Grid.dof_pene, \
                                  Grid.dof_ymax_vt,\
                                  Grid.dof_pc))
    BC.g = np.transpose([np.concatenate((np.zeros(Grid.N_pene), \
                                         vt*np.ones(len(Grid.dof_ymax_vt)),\
                                         [0.0] ))])
elif 'inst_comp_column' in simulation_type:
    BC.dof_dir =  np.concatenate((Grid.dof_pene, \
                                  Grid.dof_ymax_vt[1:-1],\
                                  Grid.dof_ymin_vt[1:-1],\
                                  np.array([Grid.p.Nf+1])))
    BC.g = np.transpose([np.concatenate((np.zeros(Grid.N_pene), \
                                         np.zeros(len(Grid.dof_ymax_vt[1:-1])),\
                                         np.zeros(len(Grid.dof_ymin_vt[1:-1])),\
                                         np.array([analytical.hD(Grid.p.dy/(2*delta0),HD)*delta0])))])
    
elif 'lid_driven_cavity_flow_with_no_slip' in simulation_type: #did not work
    BC.dof_dir =  np.concatenate((Grid.dof_pene, \
                                  Grid.dof_ymax_vt[1:-1],\
                                  Grid.dof_ymin_vt[1:-1],\
                                  Grid.dof_xmax_vt[1:-1],\
                                  Grid.dof_xmin_vt[1:-1],\
                                  Grid.dof_pc-1))
    BC.g = np.transpose([np.concatenate((np.zeros(Grid.N_pene), \
                                         vt*np.ones(len(Grid.dof_ymax_vt[1:-1])),\
                                         np.zeros(len(Grid.dof_ymin_vt[1:-1])),\
                                         np.zeros(len(Grid.dof_xmax_vt[1:-1])),\
                                         np.zeros(len(Grid.dof_xmin_vt[1:-1])),\
                                        [0.0]))])

else:#no flow
    BC.dof_dir =  np.concatenate((Grid.dof_solid_bnd, \
                                  Grid.dof_pc))
    BC.g        = np.concatenate((np.zeros(Grid.N_solid_bnd), \
                                  [0.0]))  

BC.dof_neu = np.array([])
[B,N,fn] = build_bnd(BC,Grid,I)

#Solving for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N)
v = u[:Grid.p.Nf,:]; h = u[Grid.p.Nf:,:]       #Extracting solid velocity and fluid pressure
[PSI,psi_min,psi_max] = comp_streamfun(v,Gridp)#Solid velocity stream function

p = Delta_rho * grav * (h - Yc_col)           #Overpressure [Pa] 
pf= p - rho_s * grav * Yc_col                 #Fluid pressure [Pa]
ps= pf- G * mu/ phi**m * (Dp @ v)             #Solid pressure [Pa]

#Fluid velocity
#vf = v - spdiags((1./(Mp @ phi)).T,0,Grid.p.Nf,Grid.p.Nf) @ Kd @ (Gp @ p + rho_f * grav * np.vstack([np.zeros((Grid.p.Nfx,1)),np.ones((Grid.p.Nfy,1))]))
vf = v - spdiags((1./(Mp @ phi)).T,0,Grid.p.Nf,Grid.p.Nf) @ Kd @ (Gp @ h)

[PSIf,psif_min,psif_max] = comp_streamfun(vf,Gridp) #Fluid velocity stream function

#Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,7))
fig.suptitle(f'Dim-less depth = {HD}, Porosity = {phic}')

ax1.plot(h[0:Grid.p.Ny],Grid.p.yc,'k.',label=r'Numerical')
ax1.plot(analytical.hD(zDa,HD)*delta0,zDa*delta0,'r-',label=r'Analytical')
ax1.set_ylabel('$z$ [m]')
ax1.set_xlabel('$h$ [m]')
ax1.legend(loc = 'best')

ax2.plot(p[0:Grid.p.Ny],Grid.p.yc,'k.',label=r'N')
ax2.plot(analytical.pD(zDa,HD)*Delta_rho*grav*delta0,zDa*delta0,'r-',label=r'A')
ax2.set_xlabel('Overpressure [Pa]')

ax3.plot(v[Grid.p.Nfx:Grid.p.Nfx+Grid.p.Ny],Grid.p.yc,'k.',label=r'N')
ax3.plot(analytical.vD(zDa,HD)*Kc,zDa*delta0,'r-',label=r'A')
ax3.set_xlabel('Solid velocity [m/s]')

plt.tight_layout()
plt.savefig(f"instant_comp_column_phic{phic}_HD{HD}.pdf")

