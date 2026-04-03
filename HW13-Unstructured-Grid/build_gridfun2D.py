import numpy as np
import matplotlib.pyplot as plt
    
def mean(x):
    x = np.ravel(x)
    y = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]].reshape(-1,1)
    return y

def build_grid(Grid):
    # Author: Mohammad Afzal Shadab
    # Date: 01/27/2020
    
    # This function computes takes in minimal definition of the computational
    # domain and grid and computes all containing all pertinent information 
    # about the grid. 
    # Input:
    # Grid.Lx = length of the domain
    # Grid.dx = cell width
    # Grid.xc = vector of cell center locations
    # Grid.xf = vector of cell face locations
    # Grid.Nfx = number of fluxes in x-direction
    # Grid.dof_xmin = degrees of fredom corrsponding to the cells along the x-min boundary
    # Grid.dof_xmax = degrees of fredom corrsponding to the cells along the x-max boundary
    # Grid.dof_ymin = degrees of fredom corrsponding to the cells along the y-min boundary
    # Grid.dof_ymax = degrees of fredom corrsponding to the cells along the y-max boundary
    
    # Grid.dof_f_xmin = degrees of fredom corrsponding to the faces at the x-min boundary
    # Grid.dof_f_xmax = degrees of fredom corrsponding to the faces at the x-max boundary
    # Grid.dof_f_ymin = degrees of fredom corrsponding to the faces at the y-min boundary
    # Grid.dof_f_ymax = degrees of fredom corrsponding to the faces at the y-max boundary
    # Example call: 
    # >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10; 
    # >> Grid = build_grid(Grid);
    
    # Set up the geometry
    #In x-direction
    if not hasattr(Grid,'xmin'):
        Grid.xmin = 0
        print("Grid.xmin is not defined and has been set to 0.\n")
    if not hasattr(Grid,'xmax'):
        Grid.xmax = 1 
        print("Grid.xmax is not defined and has been set to 1.\n")
    if not hasattr(Grid,'Nx'): 
        Grid.Nx   = 1
        print("Grid.Nx is not defined and has been set to 1.\n")
        
    Grid.Lx = Grid.xmax-Grid.xmin    # domain length in x
    Grid.dx = Grid.Lx/Grid.Nx        # dx of the gridblocks
    
    #In y-direction
    if not hasattr(Grid,'ymin'):
        Grid.ymin = 0
        print("Grid.ymin is not defined and has been set to 0.\n")
    if not hasattr(Grid,'ymax'):
        Grid.ymax = 1 
        print("Grid.ymax is not defined and has been set to 1.\n")
    if not hasattr(Grid,'Ny'): 
        Grid.Ny   = 1
        print("Grid.Ny is not defined and has been set to 1.\n")
        
    Grid.Ly = Grid.ymax-Grid.ymin    # domain length in y
    Grid.dy = Grid.Ly/Grid.Ny        # dy of the gridblocks
    
    # Number for fluxes
    Grid.Nfx =  (Grid.Nx+1)*Grid.Ny
    Grid.Nfy =  Grid.Nx*(Grid.Ny+1)
    Grid.Nf  =  Grid.Nfx + Grid.Nfy  
    
    # x coords of the corners of the domain
    Grid.xdom = [Grid.xmin,Grid.xmin,Grid.xmax,Grid.xmax]
    Grid.ydom = [Grid.ymin,Grid.ymax,Grid.ymin,Grid.ymax]
    
    #Set up mesh for plotting
    #xcoords of the cell centers    
    Grid.xc = np.transpose(np.linspace(Grid.xmin+Grid.dx/2, Grid.xmax-Grid.dx/2, Grid.Nx))
    Grid.yc = np.transpose(np.linspace(Grid.ymin+Grid.dy/2, Grid.ymax-Grid.dy/2, Grid.Ny))

    Grid.xf = np.transpose(np.linspace(Grid.xmin, Grid.xmax, Grid.Nx+1)) # x-coords of gridblock faces
    Grid.yf = np.transpose(np.linspace(Grid.ymin, Grid.ymax, Grid.Ny+1)) # y-coords of gridblock faces

    # Set up dof vectors
    Grid.N = Grid.Nx*Grid.Ny                                             # total number of gridblocks
    Grid.dof   = np.transpose([xi+1 for xi in range(Grid.N)])            # cell centered degree of freedom/gridblock number
    Grid.dof_f = np.transpose([xi+1 for xi in range(Grid.Nf)])           # face degree of freedom/face number

    # Boundary dof's
    # Boundary cells
    # make more efficient by avoidng DOF
    DOF = np.transpose(np.reshape(Grid.dof,(Grid.Nx,Grid.Ny)))

    Grid.dof_xmin = DOF[:,0]
    Grid.dof_xmax = DOF[:,Grid.Nx-1]
    Grid.dof_ymin = np.transpose(DOF[0,:])
    Grid.dof_ymax = np.transpose(DOF[Grid.Ny-1,:])

    # Boundary faces
    DOFx = np.transpose(np.array([list(range(1,Grid.Nfx+1,1))]).reshape((Grid.Nx+1,Grid.Ny)))
    Grid.dof_f_xmin = DOFx[:,0]
    Grid.dof_f_xmax = DOFx[:,Grid.Nx+1-1]

    #Grid.dof_f_xmin = Grid.dof_xmin
    #Grid.dof_f_xmax = np.transpose(np.array([list(range(Grid.Nfx-Grid.Ny+1,Grid.Nfx+1,1))]))

    DOFy = np.transpose(np.reshape(Grid.Nfx + np.array([list(range(1,Grid.Nfy+1,1))]),(Grid.Nx,Grid.Ny+1)))
    Grid.dof_f_ymin = np.transpose(DOFy[0,:])
    Grid.dof_f_ymax = np.transpose(DOFy[Grid.Ny+1-1,:])

    Grid.A  = np.concatenate([np.ones((Grid.Nfx,1))*Grid.dy,np.ones((Grid.Nfy,1))*Grid.dx,[[Grid.dx*Grid.dy]]], axis=0 )
    Grid.V  = np.ones((Grid.N,1))*Grid.dx*Grid.dy
    
    return Grid;



def build_grid_unstructured(Grid):
    # Author: Mohammad Afzal Shadab
    # Date: 04/03/2026
    
    # This function computes takes in minimal definition of the computational
    # domain and grid and computes all containing all pertinent information 
    # about the grid. 
    # Input:
    # Grid.Lx = length of the domain
    # Grid.dx = cell width
    # Grid.scaledx = cell width scaling multiplied to Delta x Nx by 1
    # Grid.xc = vector of cell center locations
    # Grid.xf = vector of cell face locations
    # Grid.Nfx = number of fluxes in x-direction
    # Grid.dof_xmin = degrees of fredom corrsponding to the cells along the x-min boundary
    # Grid.dof_xmax = degrees of fredom corrsponding to the cells along the x-max boundary
    # Grid.dof_ymin = degrees of fredom corrsponding to the cells along the y-min boundary
    # Grid.dof_ymax = degrees of fredom corrsponding to the cells along the y-max boundary
    
    # Grid.dof_f_xmin = degrees of fredom corrsponding to the faces at the x-min boundary
    # Grid.dof_f_xmax = degrees of fredom corrsponding to the faces at the x-max boundary
    # Grid.dof_f_ymin = degrees of fredom corrsponding to the faces at the y-min boundary
    # Grid.dof_f_ymax = degrees of fredom corrsponding to the faces at the y-max boundary
    # Example call: 
    # >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10; 
    # >> Grid = build_grid(Grid);
    
    # Set up the geometry
    #In x-direction
    if not hasattr(Grid,'xmin'):
        Grid.xmin = 0
        print("Grid.xmin is not defined and has been set to 0.\n")
    if not hasattr(Grid,'xmax'):
        Grid.xmax = 1 
        print("Grid.xmax is not defined and has been set to 1.\n")
    if not hasattr(Grid,'Nx'): 
        Grid.Nx   = 1
        print("Grid.Nx is not defined and has been set to 1.\n")
    if not hasattr(Grid,'scale_x'): 
        Grid.scale_x   = np.ones((Grid.Nx,1))
        print("Grid.scale is not defined and has been set to [1,....,1]^T.\n")  
    if not hasattr(Grid,'type'): 
        Grid.type  = 'fixed_max'
        print("Grid.type is not defined and has been set to fixed max the other alternative is fixed cell width 'fixed_wdth' like ParFlow where cell size is dx*scale.\n")   
    
    
    if Grid.type == 'fixed_max':
        Grid.Lx = Grid.xmax-Grid.xmin    # domain length in x
        Grid.dx = Grid.Lx/Grid.Nx        # dx of the gridblocks
        #scaling the grid
        Grid.dx_scaled = Grid.Lx*Grid.scale_x / np.sum(Grid.scale_x)
        Grid.dxf_scaled=  mean(Grid.dx_scaled)   
    
    else:
        Grid.Lx = Grid.xmax-Grid.xmin    # domain length in x
        Grid.dx = Grid.Lx/Grid.Nx        # dx of the gridblocks
        #scaling the grid
        Grid.Lx = Grid.Lx*np.sum(Grid.scale_x)
        Grid.xmax = Grid.xmin + Grid.Lx
        Grid.dx_scaled = Grid.dx*Grid.scale_x
        Grid.dxf_scaled=  mean(Grid.dx_scaled)
    
    #In y-direction
    if not hasattr(Grid,'ymin'):
        Grid.ymin = 0
        print("Grid.ymin is not defined and has been set to 0.\n")
    if not hasattr(Grid,'ymax'):
        Grid.ymax = 1 
        print("Grid.ymax is not defined and has been set to 1.\n")
    if not hasattr(Grid,'Ny'): 
        Grid.Ny   = 1
        print("Grid.Ny is not defined and has been set to 1.\n")
    if not hasattr(Grid,'scale_y'): 
        Grid.scale_y   = np.ones((Grid.Ny,1))
        print("Grid.scale is not defined and has been set to [1,....,1]^T.\n")   

    
    if Grid.type == 'fixed_max':
        Grid.Ly = Grid.ymax-Grid.ymin    # domain length in x
        Grid.dy = Grid.Ly/Grid.Ny        # dx of the gridblocks
        #scaling the grid
        Grid.dy_scaled = Grid.Ly*Grid.scale_y / np.sum(Grid.scale_y)
        Grid.dyf_scaled=  mean(Grid.dy_scaled)   
    else: 
        Grid.Ly = Grid.ymax-Grid.ymin    # domain length in y
        Grid.dy = Grid.Ly/Grid.Ny        # dy of the gridblocks
        #scaling the grid
        Grid.Ly   = Grid.Ly*np.sum(Grid.scale_y)
        Grid.ymax = Grid.ymin + Grid.Ly
        Grid.dy_scaled = Grid.dy*Grid.scale_y
        Grid.dyf_scaled=  mean(Grid.dy_scaled)

    # Number for fluxes
    Grid.Nfx =  (Grid.Nx+1)*Grid.Ny
    Grid.Nfy =  Grid.Nx*(Grid.Ny+1)
    Grid.Nf  =  Grid.Nfx + Grid.Nfy  
    
    # x coords of the corners of the domain
    Grid.xdom = [Grid.xmin,Grid.xmin,Grid.xmax,Grid.xmax]
    Grid.ydom = [Grid.ymin,Grid.ymax,Grid.ymin,Grid.ymax]
    
    #Set up mesh for plotting
    # faces
    Grid.xf = np.r_[Grid.xmin, Grid.xmin + np.cumsum(Grid.dx_scaled[:,0])].reshape(-1,1)
    Grid.yf = np.r_[Grid.ymin, Grid.ymin + np.cumsum(Grid.dy_scaled[:,0])].reshape(-1,1)

    #xcoords of the cell centers    

    # centers
    Grid.xc = 0.5*(Grid.xf[:-1] + Grid.xf[1:])
    Grid.yc = 0.5*(Grid.yf[:-1] + Grid.yf[1:])
    #print(Grid.xc,Grid.yc)

    # Set up dof vectors
    Grid.N = Grid.Nx*Grid.Ny                                             # total number of gridblocks
    Grid.dof   = np.transpose([xi+1 for xi in range(Grid.N)])            # cell centered degree of freedom/gridblock number
    Grid.dof_f = np.transpose([xi+1 for xi in range(Grid.Nf)])           # face degree of freedom/face number

    # Boundary dof's
    # Boundary cells
    # make more efficient by avoidng DOF
    DOF = np.transpose(np.reshape(Grid.dof,(Grid.Nx,Grid.Ny)))

    Grid.dof_xmin = DOF[:,0]
    Grid.dof_xmax = DOF[:,Grid.Nx-1]
    Grid.dof_ymin = np.transpose(DOF[0,:])
    Grid.dof_ymax = np.transpose(DOF[Grid.Ny-1,:])

    # Boundary faces
    DOFx = np.transpose(np.array([list(range(1,Grid.Nfx+1,1))]).reshape((Grid.Nx+1,Grid.Ny)))
    Grid.dof_f_xmin = DOFx[:,0]
    Grid.dof_f_xmax = DOFx[:,Grid.Nx+1-1]

    #Grid.dof_f_xmin = Grid.dof_xmin
    #Grid.dof_f_xmax = np.transpose(np.array([list(range(Grid.Nfx-Grid.Ny+1,Grid.Nfx+1,1))]))

    DOFy = np.transpose(np.reshape(Grid.Nfx + np.array([list(range(1,Grid.Nfy+1,1))]),(Grid.Nx,Grid.Ny+1)))
    Grid.dof_f_ymin = np.transpose(DOFy[0,:])
    Grid.dof_f_ymax = np.transpose(DOFy[Grid.Ny+1-1,:])
    #Grid.A  = np.concatenate([np.ones((Grid.Nfx,1))*Grid.dy,np.ones((Grid.Nfy,1))*Grid.dx,[[Grid.dx*Grid.dy]]], axis=0 )
    Grid.A  = np.concatenate([np.kron(np.ones((Grid.Nx+1,1)),Grid.dy_scaled),np.kron(Grid.dyf_scaled,np.ones((Grid.Ny+1,1))),[[Grid.Lx*Grid.Ly]]], axis=0 )
    Grid.V  = np.kron(Grid.dx_scaled,Grid.dy_scaled)#np.ones((Grid.N,1))*Grid.dx*Grid.dy    
    return Grid;

'''def plot_grid_unstructured(Grid, ms_center=20, ms_xface=30, ms_yface=30, lw=0.8):
    # cell boundary coordinates
    Xv, Yv = np.meshgrid(Grid.xf.ravel(), Grid.yf.ravel())

    fig, ax = plt.subplots()

    # draw all grid lines
    for j in range(Grid.Ny + 1):   # horizontal lines
        ax.plot(Xv[j, :], Yv[j, :], 'k-', lw=lw)

    for i in range(Grid.Nx + 1):   # vertical lines
        ax.plot(Xv[:, i], Yv[:, i], 'k-', lw=lw)

    # cell centers
    Xc, Yc = np.meshgrid(Grid.xc.ravel(), Grid.yc.ravel())
    ax.scatter(Xc, Yc, marker='o', s=ms_center, facecolors='none', edgecolors='b', label='cell centers')

    # x-face centers
    Xx, Yx = np.meshgrid(Grid.xf.ravel(), Grid.yc.ravel())
    ax.scatter(Xx, Yx, marker='x', s=ms_xface, c='r', label='x-face centers')

    # y-face centers
    Xy, Yy = np.meshgrid(Grid.xc.ravel(), Grid.yf.ravel())
    ax.scatter(Xy, Yy, marker='^', s=ms_yface, facecolors='none', edgecolors='g', label='y-face centers')

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()'''
    

def plot_grid_unstructured(Grid, ms_center=50, ms_xface=50, ms_yface=50, lw=0.8):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({'font.size': 22})
    rcParams.update({'font.family': 'Times'})
    dx_off = 0.01 * (Grid.xf.max() - Grid.xf.min())
    dy_off = 0.01 * (Grid.yf.max() - Grid.yf.min())

    Xv, Yv = np.meshgrid(Grid.xf.ravel(), Grid.yf.ravel())
    fig, ax = plt.subplots(figsize=(8,10),dpi=120)

    for j in range(Grid.Ny + 1):
        ax.plot(Xv[j, :], Yv[j, :], 'k-', lw=lw)
    for i in range(Grid.Nx + 1):
        ax.plot(Xv[:, i], Yv[:, i], 'k-', lw=lw)

    # -------- cell centers --------
    Xc, Yc = np.meshgrid(Grid.xc.ravel(), Grid.yc.ravel())
    ax.scatter(Xc, Yc, marker='o', s=ms_center,
               facecolors='none', edgecolors='b', label='cell centers')

    Xc_flat = Xc.T.reshape(-1)
    Yc_flat = Yc.T.reshape(-1)

    for k in range(Grid.N):
        ax.text(Xc_flat[k] + dx_off, Yc_flat[k] + dy_off,
                f'{Grid.dof[k]}',  color='b')

    # -------- x-faces --------
    Xx, Yx = np.meshgrid(Grid.xf.ravel(), Grid.yc.ravel())
    ax.scatter(Xx, Yx, marker='x', s=ms_xface, c='r', label='x-face centers')

    Xx_flat = Xx.T.reshape(-1)
    Yx_flat = Yx.T.reshape(-1)

    for k in range(Grid.Nfx):
        ax.text(Xx_flat[k] + dx_off, Yx_flat[k] + dy_off,
                f'{Grid.dof_f[k]}', color='r')

    # -------- y-faces --------
    Xy, Yy = np.meshgrid(Grid.xc.ravel(), Grid.yf.ravel())
    ax.scatter(Xy, Yy, marker='^', s=ms_yface,
               facecolors='none', edgecolors='g', label='y-face centers')

    Xy_flat = Xy.T.reshape(-1)
    Yy_flat = Yy.T.reshape(-1)

    for k in range(Grid.Nfy):
        ax.text(Xy_flat[k] + dx_off, Yy_flat[k] + dy_off,
                f'{Grid.dof_f[Grid.Nfx + k]}', color='g')

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(frameon=False)
    plt.show()


class Grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx   = []


Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 4; Grid.scale_x = np.transpose([np.linspace(1,4,4)])
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = 3; Grid.scale_y = np.transpose([np.linspace(1,3,3)])

#plot grid structured
Grid = build_grid(Grid)
plot_grid_unstructured(Grid)
print(Grid.V)
print(Grid.A)
#plot grid unstructured
Grid = build_grid_unstructured(Grid)
plot_grid_unstructured(Grid)
print(Grid.V)
print(Grid.A)

