%Darcy-Stokes at a steady state (h-form)
%Author: Mohammad Afzal Shadab
%Date: April 29, 2024

set(groot,'defaultAxesFontName','Times')
set(groot,'defaultAxesFontSize',20)
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')
set(groot, 'DefaultFigureVisible', 'on');

%Parameters
mu_max = 1e14; %Maximum solid viscosity [Pa.s]
phi_min= 1e-3; phi_max= 0.7; %Minimum and Maximum solid porosities [-]
G      = 1.0; %Coefficient in the bulk viscosity formulation [-]
k0     = 5.6e-11; %Relative permeability [m^2] {Meyer and Hewitt, 2017}
mu_f   = 1e-3;%Viscosity of the fluid [Pa.s]
m      = 1;   %Power law coefficient in compaction viscosity = (G/phi^m * mu_s) [-]
n      = 2;   %Power law coefficient in porosity permeability relationship k = k0*phi^n [-]
rho_s  = 917; %Density of solid [kg/m^3]
rho_f  = 1e3; %Density of fluid [kg/m^3]
Gamma  = 0;   %Rate of melting [kg/m^3-s]
grav   = 9.81;   %Acceleration due to gravity []m/s^2]
Delta_rho = rho_f - rho_s; %Difference in density of the two phases [kg/m^3]

vt     = 0;   %Tangential velocity [m/s]

%Compaction length
phic = phi_min
delta0 = sqrt(k0*phic^n*mu_max/(phic^m*mu_f))
Kc = k0*Delta_rho*grav*phic^n/mu_f


%% Build staggered grids
Gridp.xmin = 0*delta0; Gridp.xmax = 1*delta0; Gridp.Nx = 5;
Gridp.ymin = 0*delta0; Gridp.ymax = 1*delta0; Gridp.Ny = 100;
Grid = build_stokes_grid(Gridp);
[Xc,Yc] = meshgrid(Grid.p.xc,Grid.p.yc);

%mu = mu_max*(Yc(:)/Grid.p.ymax).^n;  %linearly decaying viscosity with depth


%Analytic solution
HD = Grid.p.ymax/delta0;
zDa = linspace(0,HD,1e3);
% coefficients
c1 = @(H) (exp(-H)-1)./(exp(H)-exp(-H));
c2 = @(H) (exp(H)-1)./(exp(H)-exp(-H));
% potentials
hDa = @(z,H) z + c1(H).*exp(z) + c2(H).*exp(-z);
uDa = @(z,H) -z - c1(H).*exp(z) - c2(H).*exp(-z);
% overpressure
pDa = @(z,H) c1(H).*exp(z) + c2(H).*exp(-z);
% flux & velocity
qDa = @(z,H) -1 - c1(H).*exp(z) + c2(H).*exp(-z);
vDa = @(z,H) 1 + c1(H).*exp(z) - c2(H).*exp(-z);


%Initial condition
mu = mu_max*ones(size(Yc(:)));  %This will be a function of temperature later but is constant right now
phi= phi_min * ones(Grid.p.N,1);%+ (phi_max - phi_min)*(Yc(:)/Grid.p.ymax);  %Decays with depth

%% Build Stokes operators
[D,Edot,Dp,Gp,Z,I,Ms,Mp] = build_stokes_ops_Darcy_Stokes(Grid);

%Evaluating different means
%Mud = comp_mean(mu,Ms,-1,Grid.p,1); %Average viscosity
Mud =  spdiags(Ms * (mu .* (1-phi)), 0, length(Edot),length(Edot));
Zd  = build_Zd(G,phi,m,mu,Grid.p);
Kd  = build_Kd(k0,n,phi,mu_f,Grid.p,Mp); 

A = D*2*Mud*Edot;
L = [A + Gp*Zd*Dp,  -Delta_rho*grav*Gp;...
     Dp          ,  -Delta_rho*grav*Dp*Kd*Gp ];
%fs = spalloc(Grid.N,1,0);
fs = build_RHS(phi,Kd,Grid.p,Mp,Dp,rho_f,rho_s,Gamma,grav);

%% Build BC's
BC.dof_dir = [Grid.dof_ymax_vt(2:end-1);...  % tangential velocity on the top
              Grid.dof_pene;...     % no penetration on all bnd's
              Grid.dof_ymin_vt(2:end-1);...   
              Grid.dof_pc_comp_col];         % pressure constraint
BC.dof_f_dir = [];         
BC.g       = [zeros(length(Grid.dof_ymax_vt(2:end-1)),1);...       % tangential velocity on the top
              zeros(Grid.N_pene,1);...      % no penetration on all bnd's
              zeros(length(Grid.dof_ymin_vt(2:end-1)),1);...  
              hDa(Grid.p.dy/(2*delta0),HD)*delta0];                           % pressure constraint
[B,N,fn] = build_bnd(BC,Grid,I);

%% Solve for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N);
v = u(1:Grid.p.Nf); h = u(Grid.p.Nf+1:end); %Solid velocity and overpressure head
PSI = comp_streamfun(v,Grid.p);             %Solid velocity stream function

p  = (Delta_rho * grav) * (h - Yc(:));    %Overpressure [Pa]
pf =  p - rho_s * grav * Yc(:);           %Fluid pressure [Pa]
ps =  pf- G * mu./ phi.^m .* (Dp * v);      %Solid pressure
%Fluid velocity
vf = v - spdiags(1./(Mp * phi),0,Grid.p.Nf,Grid.p.Nf) * Kd * (Gp * p + rho_f * grav * [zeros(Grid.p.Nfx,1);ones(Grid.p.Nfy,1)]);
PSIf = comp_streamfun(vf,Grid.p);            %Fluid velocity stream function



hhh=figure()
set(gcf,'units','points','position',[0,0,3125,1250])
% Enlarge figure to full screen.
set(gcf, 'Position', [50 50 1500 600])
t=sgtitle(sprintf('Dim-less depth = %.1f, Porosity = %0.3f',HD, phic)); t.FontSize = 20;
%% Plotting and post-processing
subplot 131
plot(hDa(zDa,HD)*delta0,zDa*delta0,'linewidth',2), hold on
plot(h(1:Grid.p.Ny),Grid.p.yc,'--','linewidth',2)
xlabel('h [m]','fontsize',22)
ylabel('z [m]','fontsize',22)
legend('analytic','numerical','location','northwest')
set(gca,'fontsize',18)

subplot 132
plot(pDa(zDa,HD)*Delta_rho*grav*delta0,zDa*delta0,'linewidth',2), hold on
plot(p(1:Grid.p.Ny),Grid.p.yc,'--','linewidth',2)
xlabel('Overpressure [Pa]','fontsize',22)
ylabel('z [m]','fontsize',22)
legend('analytic','numerical','location','northeast')
set(gca,'fontsize',18)

subplot 133
plot(vDa(zDa,HD)*Kc,zDa*delta0,'linewidth',2), hold on
plot(v(Grid.p.Nfx+1:Grid.p.Nfx+1+Grid.p.Ny),Grid.p.yf,'--','linewidth',2)
xlabel('Solid velocity [m/s]','fontsize',22)
ylabel('z[m]','fontsize',22)
legend('analytic','numerical','location','northeast')
set(gca,'fontsize',18)
saveas(hhh,sprintf('instant_comp_column_phic%d_HD%d.png',phic,HD));


function Zd = build_Zd(G,phi,m,mu,Grid) %building zeta^*_phi at cell centers
    Zd = (G ./ (phi.^m) - 2/3) .* mu .* (1-phi);
    Zd = spdiags(Zd,0,Grid.N,Grid.N);
end

function Kd = build_Kd(k0,n,phi,mu_f,Grid,Mp) %building Kd at cell faces
    Kd = Mp * ( k0 .* phi.^n ./ mu_f);
    Kd = spdiags(Kd,0,Grid.Nf,Grid.Nf);
end

function F = build_RHS(phi,Kd,Grid,Mp,Dp,rho_f,rho_s,Gamma,grav)
    Delta_rho = rho_f - rho_s;
    %fv at cell faces
    fv = -Delta_rho*grav*(Mp*(1-phi)).* [zeros(Grid.Nfx,1); ones(Grid.Nfy,1)] ;
    
    %fp at cell centers
    fp =-(rho_f - rho_s)/(rho_f * rho_s) * Gamma * ones(Grid.N,1);
    
    F = [fv;fp];
end