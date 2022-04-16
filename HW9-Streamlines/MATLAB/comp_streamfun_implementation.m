%% Build staggered grids
Gridp.xmin = 0; Gridp.xmax = 1; Gridp.Nx = 50;
Gridp.ymin = 0; Gridp.ymax = 1; Gridp.Ny = 50;
Grid = build_stokes_grid(Gridp);
mu = 1;
%% Build Stokes operators
[D,Edot,Dp,Gp,Z,I] = build_stokes_ops(Grid);
A = 2*mu*D*Edot; % 
L = [A, -Gp;...
     Dp, Z];
fs = spalloc(Grid.N,1,0);

%% Boundary conditions
v_top = tanh(10*(Grid.p.xf-.5));
BC.dof_dir = [Grid.dof_ymax_vt;...  % tangential velocity on the top
              Grid.dof_pene;...     % no penetration on all bnd's
              Grid.dof_pc];         % pressure constraint
BC.dof_f_dir = [];          
BC.g       = [v_top(2:end-1);...       % tangential velocity on the top
              zeros(Grid.N_pene,1);...      % no penetration on all bnd's
              0];                           % pressure constraint
[B,N,fn] = build_bnd(BC,Grid,I);

%% Solve for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N);
v = u(1:Grid.p.Nf); p = u(Grid.p.Nf+1:end);
PSI = comp_streamfun(v,Grid.p);

%% Plot solution
[Xp,Yp] = meshgrid(Grid.x.xc,Grid.y.yc);
figure
subplot 211
plot(Grid.p.xf,v_top)

ylabel('v_{top}')
subplot 212
contour(Xp,Yp,PSI,10,'k')
xlabel 'x'