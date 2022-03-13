clear 
%% Define physical parameters
n = 3;
m = 1;  % implementation only works for m
phic = 1e-3;

%% Dimensionless parameters
tmax = 10;
Z    = 50;

%% Define Grid and build operators
Grid.xmin = 0;
Grid.xmax = Z;
Grid.Nx   = 300;
Nt = tmax*100;
theta = 1; % forward  Euler (explicit)

Grid = build_grid(Grid);
[D,G,C,I,M] = build_ops(Grid);

%% Define boundary conditions
% 1) mod. Helmholtz eqn.
BC.h.dof_dir   = [];
BC.h.dof_f_dir = [];
BC.h.g         = [];
BC.h.dof_neu   = [];
BC.h.dof_f_neu = [];
BC.h.qb        = [];
[B_h,N_h,fn_h] = build_bnd(BC.h,Grid,I);

% 2) Poisson eqn.
BC.u.dof_dir   = [Grid.dof_xmin];
BC.u.dof_f_dir = [Grid.dof_f_xmin];
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
phi = ones(Grid.N,1);
%phiD = phi;
dt = tmax/Nt;

%% Initialize video
myVideo = VideoWriter('myVideoFile'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

for i = 1:Nt
    % Flow calculations
    [hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n,m,Grid,B_h,N_h,fn_h,BC.h,Grid.xc,zeros(Grid.Nx,1));
    [uD,vD] = solve_Poisson(D,G,I,phi,m,pD,Grid,B_u,N_u,fn_u,BC.u);
    if mod(i,10) == 0
        clf()
        plot_solution(Grid,phi,hD,uD,qD,vD,pD)
        string = "t ="+num2str(i*dt);
        title(string);
        fprintf('i = %d: t= %3.2f V = %3.7f;\n',i,dt*i,sum(phi));
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end
    % Transport calculations
    A = flux_upwind(vD,Grid);
    P = spdiags(pD,0,Grid.Nx,Grid.Nx);    
    L = phic * D * A - P;
    IM = I + dt*(1-theta)*L;
    EX = I - dt*theta*L;
    phi = solve_lbvp(IM , EX * phi, B_phi, BC.phi.g, N_phi); 
    %phi  = phiD; 
    if max(phi*phic) > 1 || min(phi*phic) < 0; error('Porosity outside  [0 1].\n'); end
end

%{
function [] = plot_solution(Grid,phiD,hD,uD,qD,vD,pD)
subplot 141
plot(phi,Grid.xc,'linewidth',2), hold on
% xlim([0 2])
xlabel '\phi_D'
ylabel 'z_D'

subplot 142
plot(hD,Grid.xc,'linewidth',2), hold on
plot(uD,Grid.xc,'linewidth',2)
% xlim(Grid.xmax*[-5 1])
xlabel 'h_D and u_D'
ylabel 'z_D'
legend('h_D','u_D')

subplot 143
plot(qD,Grid.xf,'linewidth',2), hold on
plot(vD,Grid.xf,'linewidth',2)
% xlim(20*[-1 1])
xlabel 'q_D and v_D'
ylabel 'z_D'
legend('q_D','v_D')

subplot 144
plot(pD,Grid.xc,'linewidth',2), hold on
xlim(2*[-1 1])
xlabel 'p_D'
ylabel 'z_D'
drawnow
end
%}
function [] = plot_solution(Grid,phiD,hD,uD,qD,vD,pD)
subplot 141
plot(phiD,Grid.xc,'linewidth',2), hold on
% xlim([0 2])
xlabel '\phi_D'
ylabel 'z_D'

subplot 142
plot(hD,Grid.xc,'linewidth',2), hold on
plot(uD,Grid.xc,'linewidth',2)
% xlim(Grid.xmax*[-5 1])
xlabel 'h_D and u_D'
ylabel 'z_D'
legend('h_D','u_D')

subplot 143
plot(qD,Grid.xf,'linewidth',2), hold on
plot(vD,Grid.xf,'linewidth',2)
% xlim(20*[-1 1])
xlabel 'q_D and v_D'
ylabel 'z_D'
legend('q_D','v_D')

subplot 144
plot(pD,Grid.xc,'linewidth',2), hold on
xlim(2*[-1 1])
xlabel 'p_D'
ylabel 'z_D'
drawnow
end