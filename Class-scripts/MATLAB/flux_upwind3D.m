function [A] = flux_upwind3D(q,Grid) % repo
% author: Mohammad Afzal Shadab, Marc Hesse
% date: 29 March 2022
% Description:
% This function computes the upwind flux matrix from the flux vector.
%
% Input:
% q = Nf by 1 flux vector from the flow problem.
% Grid = structure containing all pertinent information about the grid.
%
% Output:
% A = Nf by Nf matrix contining the upwinded fluxes
%
% Example call:
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10;
% >> Grid = build_grid(Grid);
% >> q = ones(Grid.Nf,1);
% >> [A] = flux_upwind(q,Grid);

Nx  = Grid.Nx; Ny = Grid.Ny; Nz = Grid.Nz; N = Grid.N;
Nfx = Grid.Nfx;  % # of x faces
Nfy = Grid.Nfy;  % # of y faces
Nfz = Grid.Nfz;  % # of z faces
Nf  = Grid.Nf;   % # faces

if ((Nx>1) && (Ny==1)) || ((Nx==1) && (Ny>1)) || ((Nx==1) && (Nz>1)) % 1D
    %% One dimensional
    qn = min(q(1:N),0);
    qp = max(q(2:N+1),0);
    A = spdiags([qp,qn],[-1 0],Grid.N+1,Grid.N);
elseif (Nx>1) && (Ny>1) && (Nz>1) % 2D
     Ix = speye(Nx);
     Iy = speye(Ny);
     Iz = speye(Nz);
     
    % x-matrices     
     Axp1 = spdiags(ones(Nx,1),-1,Nx+1,Nx);  % 1D x-poititve
     Axn1 = spdiags(ones(Nx,1), 0,Nx+1,Nx);  % 1D x-negative
     Axp  = kron(Iz,kron(Axp1,Iy));          % 3D x-positive
     Axn  = kron(Iz,kron(Axn1,Iy));          % 3D x-negative
     
     % y-matrices
     Ayp1 = spdiags(ones(Ny,1),-1,Ny+1,Ny);  % 1D y-poititve
     Ayn1 = spdiags(ones(Ny,1), 0,Ny+1,Ny);  % 1D y-negative
     Ayp  = kron(Iz,kron(Ix,Ayp1));          % 3D y-positive
     Ayn  = kron(Iz,kron(Ix,Ayn1));          % 3D y-negative
     
     % z-matrices
     Azp1 = spdiags(ones(Nz,1),-1,Nz+1,Nz);  % 1D z-poititve
     Azn1 = spdiags(ones(Nz,1), 0,Nz+1,Nz);  % 1D z-negative
     Azp  = kron(Azp1,kron(Ix,Iy));          % 3D z-positive
     Azn  = kron(Azn1,kron(Ix,Iy));          % 3D z-negative
     
     % Positive and Negative Matrices
     Ap = [Axp;Ayp;Azp];
     An = [Axn;Ayn;Azn];

     % Diagonal Flux Matrices
     Qp = spdiags(max(q,0),0,Nf,Nf);
     Qn = spdiags(min(q,0),0,Nf,Nf);
    
    % Full 2D Advection Matrix
     A = Qp * Ap + Qn * An;
end