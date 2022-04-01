import sys
sys.path.insert(1, '../../HW1-Numerics/Python/')
sys.path.insert(1, '../../HW2-BC_LBVP/Python/')
sys.path.insert(1, '../../HW3-Hetero-Fluxes-NeuBC/Python/')
sys.path.insert(1, '../../HW4-1DMelt-Migration/Python/')
sys.path.insert(1, '../../HW6-2D_operators/Python/')

# import python libraries
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from classfun import *
from build_gridfun2D import build_grid 
from build_opsfun2D_latest import build_ops
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from comp_fluxfun_gen import comp_flux_gen
from comp_mean_matrix import comp_mean
from solve_Helmholtz import solve_Helmholtz
from solve_Poisson import solve_Poisson
from evolve_porosity import evolve_porosity


#Simulation parameters
Param.HD = 25;                 # Dimensionless ice shell thickness    [-]
Param.LD = 25;                 # Dimensionless width of the domain    [-]
Param.DD = 3;                  # Dimensionless thickness of melt lens [-]
Param.phi_melt   = 2e-1;       # Porosity of near surface melt
Param.phi_shell  = 2e-1;       # Porosity of near surface melt
Param.m = 1;                   # compaction viscosity exponent [-]
Param.n = 2;                   # compaction viscosity exponent [-]
Param.tDmax = 4;               # Dimensionless simulation time [-]
Param.theta = .5;              # Crank-Nicholson (implicit)
Param.Nt    = Param.tDmax*50
Param.plot_interval = 50
Param.dtD   = Param.tDmax/Param.Nt

# Build grid and operator
Grid.xmin = 0; Grid.xmax = Param.HD; Grid.Nx = 100
Grid.ymin = 0; Grid.ymax = Param.LD; Grid.Ny = 100
Grid        = build_grid(Grid)
[Xc,Zc] = np.meshgrid(Grid.xc,Grid.yc)
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))
Zc_col  = np.reshape(np.transpose(Zc), (Grid.N,-1))

[D,G,C,I,M] = build_ops(Grid)
Gamma       = np.zeros((Grid.N,1))

## Build boundary conditions

#1. Helmholtz equation 
BC.h.dof_dir   = np.array([])
BC.h.dof_f_dir = np.array([]) 
BC.h.g         = np.array([]) 
BC.h.dof_neu   = np.array([Grid.dof_xmin[0],Grid.dof_xmax[0]])
BC.h.dof_f_neu = np.array([Grid.dof_f_xmin[0],Grid.dof_f_xmax[0]])
BC.h.qb        = np.array([[0],[0]])

[B_h,N_h,fn_h]     = build_bnd(BC.h, Grid, I)

#2. Poisson equation 
BC.u.dof_dir   = np.array([Grid.dof_xmin[0]])
BC.u.dof_f_dir = np.array([Grid.dof_f_xmin[0]]) 
BC.u.g         = np.array([0]) 
BC.u.dof_neu   = Grid.dof_xmax
BC.u.dof_f_neu = Grid.dof_f_xmax
BC.u.qb        = np.transpose([np.zeros_like(Grid.dof_xmax)])

[B_u,N_u,fn_u]     = build_bnd(BC.u, Grid, I)

#3. Porosity evolution equation 
BC.phi.dof_dir  = np.array([])
BC.phi.dof_f_dir= np.array([]) 
BC.phi.g        = np.array([]) 
BC.phi.dof_neu  = np.hstack([Grid.dof_xmin,Grid.dof_xmax])
BC.phi.dof_f_neu= np.hstack([Grid.dof_f_xmin,Grid.dof_f_xmax])
BC.phi.qb       = np.transpose([np.hstack([np.zeros_like(Grid.dof_xmin),np.zeros_like(Grid.dof_xmax)])])

[B_phi,N_phi,fn_phi] = build_bnd(BC.phi, Grid, I)

## Initial condition

#phiD = 1 + 0.1*np.cos(2*np.pi/Param.HD*Zc_col)
phiD = np.ones((Grid.N,1))+1e-2*np.random.randn(Grid.N,1)

phi_D_sol = phiD.copy()
tD  = 0
tD_sol = [tD]

#time loop
for i in range(1,Param.Nt):
    tD = tD + Param.dtD
    ## Solve the equations for one timestep
    #1. Helmholtz equation 
    [hD,pD,qD] = solve_Helmholtz(D,G,I,M,phiD,Param.n,Param.m,Grid,B_h,N_h,fn_h,BC.h,Zc_col,Gamma)
    
    #2. Poisson equation 
    [uD,vD]    = solve_Poisson(D,G,I,phiD,Param.m,pD,Grid,B_u,N_u,fn_u,BC.u)
    
    #3. Porosity evolution equation 
    [phiD,Av]  = evolve_porosity(D,I,phiD,vD,pD,B_phi,N_phi,BC.phi,Grid,Param.phi_shell,Param.theta,.1)
    
    if i % Param.plot_interval ==0:
        phi_D_sol = np.hstack([phi_D_sol,phiD])
        tD_sol.append(tD)
        print(i,tD)



## Plotting the solution
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(20,7))
fig.suptitle('2D Gravity Drainage problem')
plot1 = ax1.contourf(Xc,Zc,np.transpose(phiD.reshape(Grid.Nx,Grid.Ny)))
ax1.set_xlabel(r'$x_D$')
ax1.set_ylabel(r'$z_D$')
ax1.set_title('$\phi_D$')

plot2 = ax2.contourf(Xc,Zc,np.transpose(hD.reshape(Grid.Nx,Grid.Ny)))
ax2.set_xlabel(r'$x_D$')
ax2.set_ylabel(r'$z_D$')
ax2.set_title('$h_D$')

ax3.contourf(Xc,Zc,np.transpose(pD.reshape(Grid.Nx,Grid.Ny)))
ax3.set_xlabel(r'$x_D$')
ax3.set_ylabel(r'$z_D$')
ax3.set_title('$p_D$')

#For flux
Xfx,Xfy = np.meshgrid(Grid.xf,Grid.yc)
Yfx,Yfy = np.meshgrid(Grid.xc,Grid.yf)
qx      = np.transpose(qD[0:Grid.Nfx].reshape(Grid.Nx+1,Grid.Ny))
qy      = np.transpose(qD[Grid.Nfx:Grid.Nf].reshape(Grid.Nx,Grid.Ny+1))

#ax4.contourf(Xfx,Xfy,qx)
ax4.contourf(Yfx,Yfy,qy)
ax4.set_xlabel(r'$x_D$')
ax4.set_ylabel(r'$z_D$')
ax4.set_title('$qDy$')
plt.tight_layout()


plt.figure()
plt.contourf(Xc,Zc,np.transpose(phiD.reshape(Grid.Nx,Grid.Ny)))
plt.xlabel(r'$x_D$')
plt.ylabel(r'$z_D$')
plt.title('$\phi_D$')
plt.colorbar()


plt.figure()
plt.contourf(Xc,Zc,np.transpose(hD.reshape(Grid.Nx,Grid.Ny)))
plt.xlabel(r'$x_D$')
plt.ylabel(r'$z_D$')
plt.title('$h_D$')
plt.colorbar()


plt.figure()
plt.contourf(Xc,Zc,np.transpose(pD.reshape(Grid.Nx,Grid.Ny)))
plt.xlabel(r'$x_D$')
plt.ylabel(r'$z_D$')
plt.title('$p_D$')
plt.colorbar()


plt.figure()
plt.contourf(Yfx,Yfy,qy)
plt.xlabel(r'$x_D$')
plt.ylabel(r'$z_D$')
plt.title('$q_Dy$')
plt.colorbar()

plt.figure()
plt.contourf(Xfx,Xfy,qx)
plt.xlabel(r'$x_D$')
plt.ylabel(r'$z_D$')
plt.title('$q_Dx$')
plt.colorbar()


'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
C_sol = phi_D_sol
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    zarray  = zarray
    plot[0] = plt.contourf(Xc, Zc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = np.min(phi_D_sol), vmax = np.max(phi_D_sol))
    plt.title("t= %0.4f" % t[frame_number],loc = 'center', fontsize=18)
    #plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(0,1)
    
    
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymin,Grid.ymax])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(zarray[:,frame_number])
    mm.set_clim(np.min(phi_D_sol), np.max(phi_D_sol))
    clb = plt.colorbar(mm, pad=0.1)
    clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Zc, np.transpose((phi_D_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(phi_D_sol), vmax = np.max(phi_D_sol))]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymin,Grid.ymax])
#plt.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(phiD)
mm.set_clim(np.min(phi_D_sol), np.max(phi_D_sol))
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\phi_D$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f" % tD_sol[0],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(phi_D_sol[:,:], plot[:],tD_sol[:]), interval=1/fps)
ani.save(f"gravity_drainage_{Grid.Nx}by{Grid.Ny}_tf{Param.tDmax}_phiDsingle.mov", writer='ffmpeg', fps=30)

'''
