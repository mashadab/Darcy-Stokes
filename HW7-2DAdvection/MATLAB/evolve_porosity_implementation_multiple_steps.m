%% Simulation paramters
% Dimensional
Param.HD = 25;                 % Dimensionless ice shell thickness    [-]
Param.LD  = 50;                % Dimensionless width of the domain    [-]
Param.DD  = 3;                 % Dimensionless thickness of melt lens [-]
Param.phi_melt   = 2e-1;       % Porosity of near surface melt
Param.phi_shell   = 2e-3;      % Porosity of near surface melt
Param.m = 1;                   % compaction viscosity exponent [-]
Param.n = 2;                   % compaction viscosity exponent [-]
Param.tDmax = 4;               % Dimensionless simulation time [-]
Param.theta = .5;               % Crank-Nicholson (implicit)
Param.tDmax = 4;
Param.Nt = Param.tDmax*50;
Param.plot_interval = 50;
Param.dtD = Param.tDmax/Param.Nt;

%% Define Grid and build operators
Grid.xmin = 0; Grid.xmax = Param.LD; Grid.Nx = 200;
Grid.ymin = 0; Grid.ymax = Param.HD; Grid.Ny = 100;

Grid = build_grid(Grid);
[D,G,C,I,M] = build_ops(Grid);
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
% This generates a near surface melt later with a sinusoudal perturbation
% to accelearte the formation of the porosity waves.
phiD = make_ic_class(Grid,Param,D,G,I,Zc);

% Plot initial condition
figure
surf(Xc,Zc,reshape(phiD,Grid.Ny,Grid.Nx)), view(2), shading 'interp'
axis tight equal
colorbar

xlabel 'x_D'
ylabel 'z_D'
title 'Time = 0'

drawnow

% Time stepping loop (the only loop in this entire code!)
tD = 0;
for i = 1:Param.Nt
    tD = tD+Param.dtD;
    % Solve the problem
    [hD,pD,qD] = solve_Helmholtz(D,G,I,M,phiD,Param.n,Param.m,Grid,B_h,N_h,fn_h,BC.h,Zc,0);
    [uD,vD]    = solve_Poisson(D,G,I,phiD,Param.m,pD,Grid,B_u,N_u,fn_u,BC.u);
    [phiD,Av]  = evolve_porosity(D,I,phiD,vD,pD,B_phi,N_phi,BC.phi,Grid,Param.phi_shell,Param.theta,.1);

    % Check for error
    if max(phiD*Param.phi_shell) > 1 || min(phiD*Param.phi_shell) < 0; error('Porosity outside  [0 1].\n'); end
    
    % Plot solution
    if mod(i,Param.plot_interval) == 0;
        fprintf('i = %d: tD= %3.2f;\n',i,tD);
        figure
        surf(Xc,Zc,reshape(phiD,Grid.Ny,Grid.Nx)), view(2), shading 'interp'
        axis tight equal
        colorbar
        ylim([0 Grid.ymax])
        xlabel 'x_D'
        ylabel 'z_D'
        title(['Time = ',num2str(round(tD))])
        drawnow
    end
end