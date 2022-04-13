# Stokes solver Grid
# author: Mohammad Afzal Shadab
# email: mashadab@utexas.edu
# date: 02/12/2021
import sys
sys.path.insert(1, '../../HW6-2D_operators/Python/')

from classfun import *
from build_gridfun2D import build_grid

def build_stokes_grid(Gridp):
    
    # pressure grid
    Grid.p = build_grid(Gridp) 
    
    # x-velocity Grid
    Grid.Vx.xmin = Gridp.xmin - 0.5*Gridp.dx
    Grid.Vx.xmax = Gridp.xmax + 0.5*Gridp.dx
    Grid.Vx.Nx   = Gridp.Nx + 1

    Grid.Vx.ymin = Gridp.ymin
    Grid.Vx.ymax = Gridp.ymax
    Grid.Vx.Ny   = Gridp.Ny
    
    Grid.Vx = build_grid(Grid.Vx)

    # y-velocity Grid
    Grid.Vy.xmin = Gridp.xmin
    Grid.Vy.xmax = Gridp.xmax
    Grid.Vy.Nx   = Gridp.Nx

    Grid.Vy.ymin = Gridp.ymin - 0.5*Gridp.dy
    Grid.Vy.ymax = Gridp.ymax + 0.5*Gridp.dy
    Grid.Vy.Ny   = Gridp.Ny + 1
    
    Grid.Vy = build_grid(Grid.Vy)    
    
    # For an array of u, p
    Grid.N  = Grid.p.N + Grid.Vx.N + Grid.Vy.N
    
    # total number of unknowns
    
    ## Boundary dof's
    # Unknown vector is ordered: u = [vx;vy;p]
    
    # Normal velocities on bnd's
    Grid.dof_xmin_vx = Grid.Vx.dof_xmin.copy()
    Grid.dof_xmax_vx = Grid.Vx.dof_xmax.copy()
    Grid.dof_ymin_vy = Grid.p.Nfx+Grid.Vy.dof_ymin
    Grid.dof_ymax_vy = Grid.p.Nfx+Grid.Vy.dof_ymax
    
    # Tangential velocities on bnd's (all)
    Grid.dof_xmin_vy = Grid.p.Nfx+Grid.Vy.dof_xmin
    Grid.dof_xmax_vy = Grid.p.Nfx+Grid.Vy.dof_xmax
    Grid.dof_ymin_vx = Grid.Vx.dof_ymin.copy()
    Grid.dof_ymax_vx = Grid.Vx.dof_ymax.copy()
    
    # excluding extreme faces
    Grid.dof_xmin_vt = Grid.dof_xmin_vy[1:-1] 
    Grid.dof_xmax_vt = Grid.dof_xmax_vy[1:-1]
    Grid.dof_ymin_vt = Grid.dof_ymin_vx[1:-1]
    Grid.dof_ymax_vt = Grid.dof_ymax_vx[1:-1]
    
    # Pressures on bnd's
    Grid.dof_xmin_p = Grid.p.Nf+Grid.p.dof_xmin
    Grid.dof_xmax_p = Grid.p.Nf+Grid.p.dof_xmax
    Grid.dof_ymin_p = Grid.p.Nf+Grid.p.dof_ymin
    Grid.dof_ymax_p = Grid.p.Nf+Grid.p.dof_ymax
    
    # Pressure constraint in center of domain
    Grid.dof_pc = Grid.p.Nf+round(Grid.p.N/2)
    
    # Common useful BC's
    # Penetration - set normal velocities on all boundaries to zero
    Grid.dof_pene = np.hstack([Grid.dof_xmin_vx,Grid.dof_xmax_vx,\
                               Grid.dof_ymin_vy,Grid.dof_ymax_vy])
    Grid.N_pene   = len(Grid.dof_pene)
    
    # Slip - set all tangential velocities on all boundaries to zero
    Grid.dof_slip = np.hstack([Grid.dof_xmin_vt,Grid.dof_xmax_vt,\
                               Grid.dof_ymin_vt,Grid.dof_ymax_vt])
    Grid.N_slip   = len(Grid.dof_slip)
    
    # Solid boundary - no slip and no penetration
    Grid.dof_solid_bnd = np.unique(np.hstack([Grid.dof_pene,Grid.dof_slip]))
    Grid.N_solid_bnd   = len(Grid.dof_solid_bnd);
    
    
    return Grid