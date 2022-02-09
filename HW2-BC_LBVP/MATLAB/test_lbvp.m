%% Build grid and operators
Grid.xmin = -1; Grid.xmax = 2.5; Grid.Nx = 100;
Grid = build_grid(Grid);
[D,G,C,I,M]=build_ops(Grid);
L = -D*G;                   % Laplacian operator
fs = spalloc(Grid.N,1,0);   % r.h.s. (zero)

%% Setboundary conditions
BC.dof_dir   = [Grid.dof_xmin;Grid.dof_xmax];     % identify cells on Dirichlet bnd
BC.dof_f_dir = [Grid.dof_f_xmin;Grid.dof_f_xmax]; % identify faces on Dirichlet bnd
BC.dof_neu   = [];     % identify cells on Neumann bnd
BC.dof_f_neu = [];     % identify faces on Neumann bnd
BC.g  = [1;2];         % set bnd value
BC.qb = [];            % set bnd flux
[B,N,fn] = build_bnd(BC,Grid,I);  % Build constraint matrix and basis for its nullspace

%% Solve linear boundary value problem & plot solution
u = solve_lbvp(L,fs+fn,B,BC.g,N);
plot(Grid.xc,u)
xlabel 'x', ylabel 'u'
ylim([0 2])