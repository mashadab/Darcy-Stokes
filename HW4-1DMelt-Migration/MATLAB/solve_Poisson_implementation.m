% Parameters
n_exp = 3;
m_exp = 1;
HD = 30;

% build grid and ops
Grid.xmin = 0; Grid.xmax = HD; Grid.Nx = 35;
Grid = build_grid(Grid);
[D,G,C,I,M] = build_ops(Grid);
Gamma = spalloc(Grid.N,1,0);
zc = Grid.xc;
zf = Grid.xf;

% Porosity field
phi = 1 + 0.1*cos(2*pi/HD*zc);

%% Solve mod. Helmholtz eqn
% build BC's Helmholtz
BC.h.dof_dir = [];
BC.h.dof_f_dir = [];
BC.h.g = [];
BC.h.dof_neu = [];
BC.h.dof_f_neu = [];
BC.h.qb = [];
[B_h,N_h,fn_h] = build_bnd(BC.h,Grid,I);

[hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n_exp,m_exp,Grid,B_h,N_h,fn_h,BC.h,zc,Gamma);

%% Solve Poisson eqn
% build BC's Helmholtz
BC.u.dof_dir = [Grid.dof_xmin];
BC.u.dof_f_dir = [Grid.dof_f_xmin];
BC.u.g = [HD];
BC.u.dof_neu = [];
BC.u.dof_f_neu = [];
BC.u.qb = [];
[B_u,N_u,fn_u] = build_bnd(BC.u,Grid,I);

[uD,vD] = solve_Poisson(D,G,I,phi,m_exp,pD,Grid,B_u,N_u,fn_u,BC.u);

% Plot solution
subplot 141
plot(phi,zc)
xlabel '\phi_D'
ylabel 'z_D'
xlim([0 1.25])

subplot 142
plot(hD,zc), hold on
plot(uD,zc)
xlabel 'h_D and u_D'
ylabel 'z_D'
xlim([0 HD])
legend('h_D','u_D')

subplot 143
plot(pD,zc)
xlabel 'p_D'
ylabel 'z_D'
xlim([-1 1])


subplot 144
plot(qD,zf), hold on
plot(vD,zf)
xlabel 'q_D and v_D'
ylabel 'z_D'
xlim([-1.2 1.2])
legend('q_D','v_D')