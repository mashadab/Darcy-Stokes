function [uD,vD] = solve_Poisson(D,G,I,phi,m,pD,Grid,B,N,fn,BC) % class
% author: you
% date: before it's due

% Input: 
% D = N by Nf discrete divergence operator
% G = Nf by N discrete gradient operator
% I = N by N identity
% phi = Ny by Nx matrix with porosity field
% m = compaction exponent
% pD = N by 1 column vector of dimensionless overpressures
% Grid = structure containing useful info about the grid
% B = Nc by N constraint matrix
% N = N by N-Nc basis for nullspace of constraint matrix
% fn = r.h.s. vector for Neuman BC
% BC = struture with releavant info about problem parameters

% Output:
% uD = N  by 1 column vector of dimensionless solid velocity potentials
% vD = Nf by 1 column vector of dimensionless solid velocities

Phi_m = diag(phi.^m);
L  = - D * G;       % system matrix
fs = Phi_m * pD;   % r.h.s.
flux = @(h) -G * h;
res = @(h,cell) L(cell,:)*h - fs(cell);

uD = solve_lbvp(L,fs+fn,B,BC.g,N);
vD = comp_flux_gen(flux,res,uD,Grid,BC);
end