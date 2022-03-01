% Parameter
n_exp = 3;
m_exp = 1;
HD = 30;

% build grid and ops
Grid.xmin = 0; Grid.xmax = HD; Grid.Nx = 350;
Grid = build_grid(Grid);
[D,G,C,I,M] = build_ops(Grid);
Gamma = spalloc(Grid.N,1,0);
zc = Grid.xc;
zf = Grid.xf;

% build BC's
BC.dof_dir = [];
BC.dof_f_dir = [];
BC.g = [];
BC.dof_neu = [];
BC.dof_f_neu = [];
BC.qb = [];

[B,N,fn] = build_bnd(BC,Grid,I);

% Solve mod. Helmholtz eqn
phi = 1 + 0.1*cos(2*pi/HD*zc);
[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B,N,fn,BC,zc,Gamma);

% Plot solution
subplot 141
plot(phi,zc)
xlabel '\phi_D'
ylabel 'z_D'
xlim([0 1.25])

subplot 142
plot(hD,zc)
xlabel 'h_D'
ylabel 'z_D'
xlim([0 HD])

subplot 143
plot(pD,zc)
xlabel 'p_D'
ylabel 'z_D'
xlim([-1 1])

subplot 144
plot(qD,zf)
xlabel 'q_D'
ylabel 'z_D'
xlim([-1.2 .2])