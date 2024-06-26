clear all, clc

%% Grid and discrete operators
Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10;
Grid.ymin =-1; Grid.ymax = 1; Grid.Ny = 15;
Grid.zmin =-1; Grid.zmax = 1; Grid.Nz = 20;

Grid = build_grid3D(Grid); [X,Y,Z] = meshgrid(Grid.xc,Grid.yc,Grid.zc);

[D,G,C,I,M] = build_ops3D(Grid);
L = -D*G; fs = zeros(Grid.N,1);
flux = @(h) -G*h;
res = @(h,cell) L(cell,:)*h - fs(cell); 


%% Boundary conditions
BC.dof_dir   = [1; Grid.dof_xmin(2:Grid.Ny); Grid.dof_ymin(2:Grid.Nx)];
BC.dof_f_dir = [Grid.dof_f_xmin(1:Grid.Ny);Grid.dof_f_ymin(2:Grid.Nx)];
BC.g         = [.5;ones(Grid.Ny-1,1);zeros(Grid.Nx-1,1)];
BC.dof_neu   = [];
BC.dof_f_neu = [];
BC.qb        = [];
[B,N,fn] = build_bnd(BC,Grid,I);

%% Solve problem and compute fluxes
u = solve_lbvp(L,fs+fn,B,BC.g,N);
v = comp_flux_gen(flux,res,u,Grid,BC);

%plotting
plotting(Grid,X,Y,Z,u)


%% Compute advection operator
A = flux_upwind3D(v,Grid);



function plotting(Grid,X,Y,Z,soln)

%Plotting
%full 3D slice
figure()
xslice = [(Grid.xmax-Grid.xmin)/4 (Grid.xmax-Grid.xmin)/2 (Grid.xmax-Grid.xmin)*3/4];                               % define the cross sections to view
yslice = [(Grid.ymax-Grid.ymin)/2];
zslice = ([(Grid.zmax-Grid.zmin)/2]);

soln = reshape(soln,Grid.Ny,Grid.Nx,Grid.Nz);

slice(X, Y, Z, soln, xslice, yslice, zslice)    % display the slices
xlabel 'x', ylabel 'y', zlabel 'z'
cb = colorbar;                                  % create and label the colorbar
title 'u'
cb.Label.String = 'u ';
end