function [A] = flux_upwind(q,Grid) % repo
% author: 
% date: 
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

Nx = Grid.Nx; Ny = Grid.Ny; Nz = Grid.Nz; N = Grid.N;
Nfx = Grid.Nfx;  % # of x faces
Nfy = Grid.Nfy;  % # of y faces
Nf  = Grid.Nf;   % # faces

if ((Nx>1) && (Ny==1)) || ((Nx==1) && (Ny>1)) % 1D
    %% One dimensinal
    qn = min(q(1:N),0);
    qp = max(q(2:N+1),0);
    A = spdiags([qp,qn],[-1 0],Grid.N+1,Grid.N);
elseif (Nx>1) && (Ny>1) % 2D
    % x-matrices
     Iy = speye(Ny);
     Axp1 = spdiags(ones(Nx,1),-1,Nx+1,Nx);  % 1D x-poititve
     Axn1 = spdiags(ones(Nx,1), 0,Nx+1,Nx);   % 1D x-negative
     Axp  = kron(Axp1,Iy);                    % 2D x-positive
     Axn  = kron(Axn1,Iy);                    % 2D x-negative
     
     % y-matrices
     Ix = speye(Nx);
     Ayp1 = spdiags(ones(Ny,1),-1,Ny+1,Ny);  % 1D x-poititve
     Ayn1 = spdiags(ones(Ny,1), 0,Ny+1,Ny);   % 1D x-negative
     Ayp  = kron(Ix,Ayp1);                    % 2D x-positive
     Ayn  = kron(Ix,Ayn1);                    % 2D x-negative
     
     % Positive and Negative Matrices
     Ap = [Axp;Ayp];
     An = [Axn;Ayn];

     % Diagonal Flux Matrices
     Qp = spdiags(max(q,0),0,Nf,Nf);
     Qn = spdiags(min(q,0),0,Nf,Nf);
    
    % Full 2D Advection Matrix
     A = Qp * Ap + Qn * An;
end