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
    
    return Grid;