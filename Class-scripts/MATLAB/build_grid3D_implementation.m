%Solving heat equation - D * G * u = fs in 3D
%Spherical source term at the center
%Mohammad Afzal Shadab and Marc Hesse
%Date: 14 March 2022

set_demo_defaults;

%building grid and operators
Grid.xmin = 0; Grid.xmax = 3; Grid.Nx = 50;
Grid.ymin = 0; Grid.ymax = 3; Grid.Ny = 50;
Grid.zmin = 0; Grid.zmax = 3; Grid.Nz = 50;

Grid = build_grid3D(Grid);
[D,G,C,I,M] = build_ops3D(Grid);

%boundary conditions
BC.dof_dir = [Grid.dof_xmin;Grid.dof_xmax];
BC.dof_f_dir= [Grid.dof_f_xmin;Grid.dof_f_xmax];
BC.g = [zeros(length(Grid.dof_xmin),1);zeros(length(Grid.dof_xmax),1)];
BC.dof_neu = [];
BC.dof_f_neu = [];
BC.qb = [];
[B,N,fn] = build_bnd(BC,Grid,I);

L = - D * G;

[X,Y,Z] = meshgrid(Grid.xc,Grid.yc,Grid.zc);
fs = zeros(Grid.N,1);
Xo=(Grid.xmax - Grid.xmin)/2;Yo=(Grid.ymax - Grid.ymin)/2;Zo=(Grid.zmax - Grid.zmin)/2;

fs((X-Xo).^2 + (Y-Yo).^2 + (Z-Zo).^2 < 0.5) = 1.0;  %spherical source term at center

%solving the linear boundary value problem
tic
u = solve_lbvp(L,fs+fn,B,BC.g,N);
toc
%plotting
plotting(Grid,X,Y,Z,u)



function plotting(Grid,X,Y,Z,soln)

%Plotting
%full 3D slice
figure()
xslice = [(Grid.xmax-Grid.xmin)/2];                               % define the cross sections to view
yslice = [(Grid.ymax-Grid.ymin)/2];
zslice = ([(Grid.zmax-Grid.zmin)/2]);

soln = reshape(soln,Grid.Ny,Grid.Nx,Grid.Nz);

slice(X, Y, Z, soln, xslice, yslice, zslice)    % display the slices
xlabel 'x', ylabel 'y', zlabel 'z'
cb = colorbar;                                  % create and label the colorbar
title 'u'
cb.Label.String = 'u ';
%caxis([-120 40])

%1D plots along centerline
figure
title 'Along centerlines'
subplot 311
plot(Grid.xc,soln(Grid.Ny/2,:,Grid.Nz/2),'r','markerfacecolor','w','markersize',6)
xlabel 'x', ylabel 'u'

subplot 312
plot(Grid.yc,soln(:,Grid.Nx/2,Grid.Nz/2),'r','markerfacecolor','w','markersize',6)
xlabel 'y', ylabel 'u'

subplot 313
plot(Grid.zc,reshape(soln(Grid.Ny/2,Grid.Nx/2,:),Grid.Nz,1),'r','markerfacecolor','w','markersize',6)
xlabel 'z', ylabel 'u'

end


%{
disp('Grid.dof_xmin'); Grid.dof_xmin
disp('Grid.dof_xmax');Grid.dof_xmax
disp('Grid.dof_ymin');Grid.dof_ymin
disp('Grid.dof_ymax');Grid.dof_ymax
disp('Grid.dof_zmin');Grid.dof_zmin
disp('Grid.dof_zmax');Grid.dof_zmax
%}