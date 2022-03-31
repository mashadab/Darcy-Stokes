function [D,G,C,I,M]=build_ops2D(Grid)
% author: someone
% date: someday
% description:
% This function computes the discrete divergence and gradient matrices on a
% regular staggered grid using central difference approximations. The
% discrete gradient assumes homogeneous boundary conditions.
% Input:
% Grid = structure containing all pertinent information about the grid.
% Output:
% D = discrete divergence matrix
% G = discrete gradient matrix
% I = identity matrix
%
% Example call:
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 4;
% >> Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = 3;
% >> Grid = build_grid(Grid);
% >> [D,G,I]=build_ops(Grid);

% this will help
Nx = Grid.Nx; Ny = Grid.Ny; N = Grid.N;

if (Nx>1) && (Ny>1)  % 2D case
    % 1D divergence matrices
    Dx = spdiags([-ones(Nx,1) ones(Nx,1)]/Grid.dx,[0 1],Nx,Nx+1); % 1D div-matrix in x-dir
    Dy = spdiags([-ones(Ny,1) ones(Ny,1)]/Grid.dy,[0 1],Ny,Ny+1); % 1D div-matrix in y-dir
    Ix = speye(Nx); % 1D Nx-identity
    Iy = speye(Ny); % 1D Ny-identity

    % 2D Tensor-product divergence matrices
    Dx = kron(Dx,Iy);  % 2D div-matrix in x-dir
    Dy = kron(Ix,Dy);  % 2D div-matrix in y-dir

    % Complete 2D divergence
    D = [Dx,Dy];
    % Boundary faces
    dof_f_bnd = [Grid.dof_f_xmin;Grid.dof_f_xmax;Grid.dof_f_ymin;Grid.dof_f_ymax];
elseif (Nx>1) && (Ny==1)
    D = spdiags([-ones(Nx,1) ones(Nx,1)]/Grid.dx,[0 1],Nx,Nx+1);
    dof_f_bnd = [Grid.dof_f_xmin; Grid.dof_f_xmax];   % boundary faces
elseif (Nx==1) && (Ny>1)
    D = spdiags([-ones(Ny,1) ones(Ny,1)]/Grid.dy,[0 1],Ny,Ny+1);
    dof_f_bnd = [Grid.dof_f_ymin; Grid.dof_f_ymax];   % boundary faces
end

%% Gradient
% adjoint relation
G = -D';
% natural bc's
G(dof_f_bnd,:) = 0;

%% Identity
I = speye(N);

%% Curl
C = [];


%% Mean matrix
if (Nx>1) && (Ny==1) % 1D x-direction
    M = spdiags([ones(Nx,1) ones(Nx,1)]/2.0,[-1 0],Nx+1,Nx);  % 1D mean-matrix in x-dir
    M(1,1) = 1; 
    M(Nx+1,Nx) = 1;
elseif (Nx==1) && (Ny>1) % 1D y-direction
    M = spdiags([ones(Ny,1) ones(Ny,1)]/2.0,[-1 0],Ny+1,Ny); % 1D mean-matrix in y-dir
    M(1,1) = 1; 
    M(Ny+1,Ny) = 1;
elseif (Nx>1) && (Ny>1)  % 2D case
    % 1D operators
    Mx = spdiags([ones(Nx,1) ones(Nx,1)]/2.0,[-1 0],Nx+1,Nx); % 1D mean-matrix in x-dir
    Mx(1,1) = 1; 
    Mx(Nx+1,Nx) = 1;
    My = spdiags([ones(Ny,1) ones(Ny,1)]/2.0,[-1 0],Ny+1,Ny); % 1D mean-matrix in y-dir
    My(1,1) = 1; 
    My(Ny+1,Ny) = 1;
    % 2D operators
    Mx = kron(Mx,Iy);                                % 2D mean-matrix in x-dir
    My = kron(Ix,My);                                % 2D mean-matrix in y-dir
    M  = [Mx;My];
    end
