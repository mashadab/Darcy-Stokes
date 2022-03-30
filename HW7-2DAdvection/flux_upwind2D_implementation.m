clear all, clc

%% Grid and discrete operators
Grid.xmin = 0;  Grid.xmax = 2; Grid.Nx = 10;
Grid.ymin = -1; Grid.ymax = 1; Grid.Ny = 15;
Grid = build_grid2D(Grid); [Xc,Yc] = meshgrid(Grid.xc,Grid.yc);
[D,G,C,I,M] = build_ops2D(Grid);
L = -D*G; fs = zeros(Grid.N,1);
flux = @(h) -G*h;
res = @(h,cell) L(cell,:)*h - fs(cell); 


%% Boundary conditions
BC.dof_dir   = [1; Grid.dof_xmin(2:Grid.Ny); Grid.dof_ymin(2:Grid.Nx)];
BC.dof_f_dir = [Grid.dof_f_xmin;Grid.dof_f_ymin(2:Grid.Nx)];
BC.g         = [.5;ones(Grid.Ny-1,1);zeros(Grid.Nx-1,1)];
BC.dof_neu   = [];
BC.dof_f_neu = [];
BC.qb        = [];
[B,N,fn] = build_bnd(BC,Grid,I);

%% Solve problem and compute fluxes
u = solve_lbvp(L,fs+fn,B,BC.g,N);
v = comp_flux_gen(flux,res,u,Grid,BC);


%% Compute advection operator
A = flux_upwind2D(v,Grid);