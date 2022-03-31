function [Grid] = build_grid2D(Grid)
% Author: Superman
% Date: X.Y.Z
% Description:
% This function computes takes in minimal definition of the computational
% domain and grid and computes all containing all pertinent information 
% about the grid. 
% Input:
% Grid.xmin = left boundary of the domain
% Grid.xmax = right bondary of the domain
% Grid.Nx   = number of grid cells
% Output: (suggestions)
% Grid.Lx = length of the domain
% Grid.dx = cell width
% Grid.xc = vector of cell center locations
% Grid.xf = vector of cell face locations
% Grid.Nfx = number of fluxes in x-direction
% Grid.dof_xmin = degrees of fredom corrsponding to the cells along the x-min boundary
% Grid.dof_xmax = degrees of fredom corrsponding to the cells along the x-max boundary
% Grid.dof_ymin = degrees of fredom corrsponding to the cells along the y-min boundary
% Grid.dof_ymax = degrees of fredom corrsponding to the cells along the y-max boundary
%
% Grid.dof_f_xmin = degrees of fredom corrsponding to the faces at the x-min boundary
% Grid.dof_f_xmax = degrees of fredom corrsponding to the faces at the x-max boundary
% Grid.dof_f_ymin = degrees of fredom corrsponding to the faces at the y-min boundary
% Grid.dof_f_ymax = degrees of fredom corrsponding to the faces at the y-max boundary
% Grid.psi_x0 = reference location for streamfunction
% Grid.psi_dir = diretion of integration for streamfunction 
% + anything else you might find useful
%
% Example call: 
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10; 
% >> Grid = build_grid(Grid);

%% Set up catesian geometry
if ~isfield(Grid,'geom'); Grid.geom = 'cartesian'; end
if ~isfield(Grid,'xmin'); Grid.xmin = 0;  fprintf('Grid.xmin is not defined and has been set to zero.\n');end
if ~isfield(Grid,'xmax'); Grid.xmax = 10; fprintf('Grid.xmax is not defined and has been set to 10.\n'); end
if ~isfield(Grid,'Nx');   Grid.Nx   = 10; fprintf('Grid.Nx is not defined and has been set to 10.\n');end
Grid.Lx = Grid.xmax - Grid.xmin; % domain length in x
Grid.dx = Grid.Lx/Grid.Nx;       % dx of the gridblocks

if ~isfield(Grid,'ymin'); Grid.ymin = 0; end
if ~isfield(Grid,'ymax'); Grid.ymax = 1; end
if ~isfield(Grid,'Ny');   Grid.Ny   = 1; end
Grid.Ly = Grid.ymax - Grid.ymin; % domain length in y
Grid.dy = Grid.Ly/Grid.Ny;       % dy of the gridblocks

%% Number for fluxes
Grid.Nfx = (Grid.Nx+1)*Grid.Ny;
Grid.Nfy = (Grid.Ny+1)*Grid.Nx;
Grid.Nf  =  Grid.Nfx + Grid.Nfy;

% Set up mesh for plotting
% x, y, z coords of the cell centers      

Grid.xc = [Grid.xmin+Grid.dx/2:Grid.dx:Grid.xmax-Grid.dx/2]';   % x-coords of gridblock centers
Grid.yc = [Grid.ymin+Grid.dy/2:Grid.dy:Grid.ymax-Grid.dy/2]';   % y-coords of gridblock centers
Grid.xf = [Grid.xmin:Grid.dx:Grid.xmax]';                     % x-coords of gridblock faces
Grid.yf = [Grid.ymin:Grid.dy:Grid.ymax]';                     % y-coords of gridblock faces

%% Set up dof vectors
Grid.N =  Grid.Nx*Grid.Ny;% total number of gridblocks
Grid.dof = [1:Grid.N]';    % cell centered degree of freedom/gridblock number
Grid.dof_f = [1:Grid.Nf]'; % face degree of freedom/face number

%% Boundary dof's
Grid.DOF = reshape(Grid.dof,Grid.Ny,Grid.Nx);
Grid.dof_xmin = Grid.DOF(:,1);
Grid.dof_xmax = Grid.DOF(:,end);
Grid.dof_ymin = Grid.DOF(1,:)';
Grid.dof_ymax = Grid.DOF(end,:)';

% Boundary faces
Grid.DOF_Fx = reshape(Grid.dof_f(1:Grid.Nfx),Grid.Ny,Grid.Nx+1);
Grid.dof_f_xmin = Grid.DOF_Fx(:,1);
Grid.dof_f_xmax = Grid.DOF_Fx(:,end);

% note: y-fluxes are shifted by Nfx
Grid.DOF_Fy = reshape(Grid.dof_f(Grid.Nfx+1:Grid.Nf),Grid.Ny+1,Grid.Nx);
Grid.dof_f_ymin = Grid.DOF_Fy(1,:)';
Grid.dof_f_ymax = Grid.DOF_Fy(end,:)'; 

Grid.A = [ones(Grid.Nfx,1)*Grid.dy;...
          ones(Grid.Nfy,1)*Grid.dx;
          Grid.dx*Grid.dy];
Grid.V  = ones(Grid.N,1)*Grid.dx*Grid.dy;