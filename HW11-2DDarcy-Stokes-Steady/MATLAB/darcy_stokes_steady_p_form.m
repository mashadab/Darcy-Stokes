%Darcy-Stokes at a steady state
%Author: Mohammad Afzal Shadab
%Date: May 27, 2023

%Parameters
mu_max = 1e14; %Maximum solid viscosity [Pa.s]
phi_min= 0.2; phi_max= 0.7; %Minimum and Maximum solid porosities [-]
G      = 1.0; %Coefficient in the bulk viscosity formulation [-]
k0     = 5.6e-11; %Relative permeability [m^2] {Meyer and Hewitt, 2017}
mu_f   = 1e-3;%Viscosity of the fluid [Pa.s]
m      = 1;   %Power law coefficient in compaction viscosity = (G/phi^m * mu_s) [-]
n      = 2;   %Power law coefficient in porosity permeability relationship k = k0*phi^n [-]
rho_s  = 917; %Density of solid [kg/m^3]
rho_f  = 1e3; %Density of fluid [kg/m^3]
Gamma  = 0;   %Rate of melting [kg/m^3-s]
grav   = 9.81;   %Acceleration due to gravity []m/s^2]

vt     = 1e-5;   %Tangential velocity [m/s]

%% Build staggered grids
Gridp.xmin = 0; Gridp.xmax = 1; Gridp.Nx = 4;
Gridp.ymin = 0; Gridp.ymax = 1; Gridp.Ny = 4;
Grid = build_stokes_grid(Gridp);
[Xc,Yc] = meshgrid(Grid.p.xc,Grid.p.yc);

%mu = mu_max*(Yc(:)/Grid.p.ymax).^n;  %linearly decaying viscosity with depth

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
L = [A + Gp*Zd*Dp,  -Gp;...
     Dp          ,  -Dp*Kd*Gp ];
%fs = spalloc(Grid.N,1,0);
fs = build_RHS(phi,Kd,Grid.p,Mp,Dp,rho_f,rho_s,Gamma,grav);

%% Build BC's
BC.dof_dir = [Grid.dof_ymax_vt(2:end-1);...  % tangential velocity on the top
              Grid.dof_pene;...     % no penetration on all bnd's
              Grid.dof_ymin_vt(2:end-1);...  
              Grid.dof_xmin_vt(2:end-1);...  
              Grid.dof_xmax_vt(2:end-1);...  
              Grid.dof_pc];         % pressure constraint
BC.dof_f_dir = [];         
BC.g       = [vt*ones(length(Grid.dof_ymax_vt(2:end-1)),1);...       % tangential velocity on the top
              zeros(Grid.N_pene,1);...      % no penetration on all bnd's
              zeros(length(Grid.dof_ymin_vt(2:end-1)),1);...  
              zeros(length(Grid.dof_xmin_vt(2:end-1)),1);...  
              zeros(length(Grid.dof_xmax_vt(2:end-1)),1);...  
              0];                           % pressure constraint
[B,N,fn] = build_bnd(BC,Grid,I);

%% Solve for Stokes flow
u = solve_lbvp(L,fs+fn,B,BC.g,N);
v = u(1:Grid.p.Nf); p = u(Grid.p.Nf+1:end); %Solid velocity and fluid pressure
PSI = comp_streamfun(v,Grid.p);             %Solid velocity stream function

ps = p - G * mu./ phi.^m .* (Dp * v);       %Solid pressure
%Fluid velocity
vf = v - spdiags(1./(Mp * phi),0,Grid.p.Nf,Grid.p.Nf) * Kd * (Gp * p + rho_f * grav * [zeros(Grid.p.Nfx,1);ones(Grid.p.Nfy,1)]);
PSIf = comp_streamfun(vf,Grid.p);            %Fluid velocity stream function

%% Plot solution
[Xp,Yp] = meshgrid(Grid.x.xc,Grid.y.yc);
psi_max = max(PSI(:));
figure
contour(Xp,Yp,PSI,100,'k'), hold on
contour(Xp,Yp,PSI,psi_max*[.2:.2:1],'r-')
legend('main circulation','corner eddies','location','south')
axis square
xlabel('x','fontsize',14)
ylabel('z','fontsize',14)
set(gca,'xtick',[0:.2:1],'ytick',[0:.2:1])
xlim([0 1]), ylim([0 1])
ta = annotation('textarrow');
s = ta.FontSize;
ta.FontSize = 12;
ta.Position = [0.4000 .9500 0.2000 0.0000];
ta.VerticalAlignment = 'bot';
ta.HorizontalAlignment = 'right';
text(.45,1.06,'v_{lid}','fontsize',12)

figure
subplot 141
contourf(Xc,Yc,reshape(phi,Grid.p.Ny,Grid.p.Nx),100);
c1 = colorbar;
xlabel('x','fontsize',14)
ylabel('z','fontsize',14)
title 'Porosity';
axis square

subplot 142
contourf(Xc,Yc,reshape(p,Grid.p.Ny,Grid.p.Nx),100);
c2 = colorbar;
xlabel('x','fontsize',14)
ylabel('z','fontsize',14)
title 'Fluid pressure [Pa]';
axis square

subplot 143
contourf(Xc,Yc,reshape(p + rho_s * grav * Yc(:),Grid.p.Ny,Grid.p.Nx),100);
c2 = colorbar;
xlabel('x','fontsize',14)
ylabel('z','fontsize',14)
title 'Overpressure [Pa]';
axis square

subplot 144
contour(Xp,Yp,PSI,50,'k'), hold on
contour(Xp,Yp,PSIf,50,'b--')
legend('solid','fluid','location','southoutside')
axis square
xlabel('x','fontsize',14)
ylabel('z','fontsize',14)
xlim([0 1]), ylim([0 1])
ta = annotation('textarrow');
s = ta.FontSize;
ta.FontSize = 12;
ta.Position = [0.4000 .9500 0.2000 0.0000];
ta.VerticalAlignment = 'bot';
ta.HorizontalAlignment = 'right';
str = strcat('v_{lid}=',num2str(vt),'m/s');
text(.45,1.06,str,'fontsize',12)


function Zd = build_Zd(G,phi,m,mu,Grid) %building zeta^*_phi at cell centers
    Zd = (G ./ (phi.^m) - 2/3) .* mu .* (1-phi);
    Zd = spdiags(Zd,0,Grid.N,Grid.N);
end

function Kd = build_Kd(k0,n,phi,mu_f,Grid,Mp) %building Kd at cell faces
    Kd = Mp * ( k0 .* phi.^n ./ mu_f);
    Kd = spdiags(Kd,0,Grid.Nf,Grid.Nf);
end

function F = build_RHS(phi,Kd,Grid,Mp,Dp,rho_f,rho_s,Gamma,grav)
    %fv at cell faces
    fv = (Mp*(rho_f*phi + rho_s*(1-phi))).*grav.* [zeros(Grid.Nfx,1); ones(Grid.Nfy,1)] ;
    
    %fp at cell centers
    fp =-(rho_f - rho_s)/(rho_f * rho_s) * Gamma + Dp * (Kd * rho_f * grav * [zeros(Grid.Nfx,1); ones(Grid.Nfy,1)]);
    
    F = [fv;fp];
end

