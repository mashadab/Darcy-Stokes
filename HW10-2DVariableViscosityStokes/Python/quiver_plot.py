import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


def quiver_plot(simulation_name, grid,v):

    # x velocity internpolation from faces to cell centers
    Xx, Yx = np.meshgrid(grid.Vx.xc, grid.Vx.yc)
    Vx = np.transpose((v[:grid.p.Nfx,:]).reshape(grid.p.Nx+1,grid.p.Ny))
    interp_Vx = RectBivariateSpline(grid.Vx.yc, grid.Vx.xc, Vx)
    
    # y velocity internpolation from faces to cell centers
    Xy, Yy = np.meshgrid(grid.Vy.xc, grid.Vy.yc)
    Vy = np.transpose((v[grid.p.Nfx:,:]).reshape(grid.p.Nx,grid.p.Ny+1))
    interp_Vy = RectBivariateSpline(grid.Vy.yc, grid.Vy.xc, Vy)
    
    Xc, Yc = np.meshgrid(grid.p.xc, grid.p.yc)
    Vxc = interp_Vx(grid.p.yc, grid.p.xc)
    Vyc = interp_Vy(grid.p.yc, grid.p.xc)
    
    fig = plt.figure(figsize=(15,15) , dpi=100)
    Q  = plt.quiver(Xc,Yc,Vxc,Vyc)
    qk = plt.quiverkey(Q, 0.75, 0.9, 1, r'$1 unit$', labelpos='E',
                       coordinates='figure')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([grid.p.xmin,grid.p.xmax])
    plt.ylim([grid.p.ymin,grid.p.ymax])
    plt.axis('scaled')
    plt.scatter(Xc, Yc, color='r', s=5)
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    plt.show()
    plt.savefig(f'{simulation_name}.pdf')



