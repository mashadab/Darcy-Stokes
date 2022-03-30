%% Simulation paramters
Param.HD = 25;                 % Dimensionless ice shell thickness    [-]
Param.LD  = 25;                % Dimensionless width of the domain    [-]
Param.DD  = 3;                 % Dimensionless thickness of melt lens [-]
Param.phi_melt   = 2e-1;       % Porosity of near surface melt
Param.phi_shell   = 2e-1;      % Porosity of near surface melt
Param.m = 1;                   % compaction viscosity exponent [-]
Param.n = 2;                   % compaction viscosity exponent [-]
Param.tDmax = 4;               % Dimensionless simulation time [-]
Param.theta = .5;               % Crank-Nicholson (implicit)


%% Define Grid and build operators
Grid.xmin = 0; Grid.xmax = Param.LD; Grid.Nx = 100;
Grid.ymin = 0; Grid.ymax = Param.HD; Grid.Ny = 100;

Grid = build_grid2D(Grid);
[D,G,C,I,M] = build_ops2D(Grid);
[Xc,Zc] = meshgrid(Grid.xc,Grid.yc);

%% Define boundary conditions
% 1) Helmholtz eqn.
BC.h.dof_dir   = [];
BC.h.dof_f_dir = [];
BC.h.g         = [];
BC.h.dof_neu   = [];
BC.h.dof_f_neu = [];
BC.h.qb        = [];
[B_h,N_h,fn_h] = build_bnd(BC.h,Grid,I);

% 2) Poisson eqn.
BC.u.dof_dir   = [1];
BC.u.dof_f_dir = [1];
BC.u.g         = [0];
BC.u.dof_neu   = [];
BC.u.dof_f_neu = [];
BC.u.qb        = [];
[B_u,N_u,fn_u] = build_bnd(BC.u,Grid,I);

% 3) Advection eqn.
BC.phi.dof_dir   = [];
BC.phi.dof_f_dir = [];
BC.phi.g         = [];
BC.phi.dof_neu   = [];
BC.phi.dof_f_neu = [];
BC.phi.qb        = [];
[B_phi,N_phi,fn_phi] = build_bnd(BC.phi,Grid,I);

%% Initial condition
phiD = ones(Grid.N,1)+1e-2*randn(Grid.N,1); 

% Solve one timestep
[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phiD,Param.n,Param.m,Grid,B_h,N_h,fn_h,BC.h,Zc(:),0);
[uD,vD]    = solve_Poisson(D,G,I,phiD,Param.m,pD,Grid,B_u,N_u,fn_u,BC.u);
[phiD,Av]  = evolve_porosity(D,I,phiD,vD,pD,B_phi,N_phi,BC.phi,Grid,Param.phi_shell,Param.theta,.1);

% Plot solution
figure
surf(Xc,Zc,reshape(phiD,Grid.Ny,Grid.Nx)), view(2), shading 'interp'
axis tight equal
colorbar
xlabel 'x_D'
ylabel 'z_D'
title 'porosity: \phi_D'