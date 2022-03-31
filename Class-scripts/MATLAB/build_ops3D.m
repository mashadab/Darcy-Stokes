function [D,G,C,I,M]=build_ops3D(Grid)
% author: 
% date: 
% description:
% This function computes the discrete operator matrices on a
% staggered grid using central difference approximations. The
% discrete gradient assumes homogeneous boundary conditions.
% Input:
% Grid = structure containing all pertinent information about the grid.
% Output:
% D = Nx by Nx+1 discrete divergence matrix 
% G = Nx+1 by Nx discrete gradient matrix
% C = discrete curl - not defined in 1D
% I = Nx by Nx identity matrix
% M = Nx+1 by Nx discrete mean matrix
%
% Example call:
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10;
% >> Grid = build_grid(Grid);
% >> [D,G,C,I,M]=build_ops(Grid);

Nx = Grid.Nx; Nfx = Grid.Nfx;
Ny = Grid.Ny; Nfy = Grid.Nfy;
Nz = Grid.Nz; Nfz = Grid.Nfz;

%% 1) Build sparse Divergence operator
%Build Identities
Ix = speye(Nx);
Iy = speye(Ny);
Iz = speye(Nz);

% Build 3D Dx matrix
Dx = spdiags([-ones(Nx,1) ones(Nx,1)]/Grid.dx,[0 1],Nx,Nx+1);
Dx = kron(Iz,kron(Dx,Iy));

% Build 3D Dy matrix
Dy = spdiags([-ones(Ny,1) ones(Ny,1)]/Grid.dy,[0 1],Ny,Ny+1);
Dy = kron(Iz,kron(Ix,Dy));

% Build 3D Dz matrix
Dz = spdiags([-ones(Nz,1) ones(Nz,1)]/Grid.dz,[0 1],Nz,Nz+1);
Dz = kron(Dz,kron(Ix,Iy));

% Full 3D discrete divergence matrix
D = [Dx,Dy,Dz];
                                                                            
%% 2) Build sparse Gradient operator
%  Interior
G = -D';

% Set natural (homogeneous Neumann) boundary conditions
dof_f_bnd = [Grid.dof_f_xmin;Grid.dof_f_xmax; ...
             Grid.dof_f_ymin;Grid.dof_f_ymax; ...
             Grid.dof_f_zmin;Grid.dof_f_zmax]; % all dof's on boundary
G(dof_f_bnd,:) = 0;

% 3) Discrete Curl operator (not defined in 1D)
C = [];

% 4) Sparse Identity 
I = speye(Grid.N);

% 5) Sparse Mean
% Interior
M = G;
M(M~=0) = 0.5;

