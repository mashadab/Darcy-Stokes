function [hD,pD,qD] = solve_Helmholtz(D,G,I,M,phi,n,m,Grid,B,N,fn,BC,Zc,Gamma) % class
% author: you
% date: today

% Input: (to be completed)
% D = N by Nf discrete divergence operator
% G = Nf by N discrete gradient operator
% I = N by N identity
% phi = Nx by 1 matrix with porosity field
% n = permeability exponent
% m = compaction exponent
% B = Nc by N constraint matrix
% N = N by N-Nc basis for nullspace of constraint matrix
% fn = r.h.s. vector for Neuman BC
% BC = struture with info for boundary condtions
% Zc = Nx by 1 matrix containing the vertical coordinate of cell centers
% Gamma = N by 1 vector containing melting term

% Output:
% hD = N by 1 column vector of dimensionless overpressure heads
% pD = N by 1 column vector of dimensionless overpressure
% qD = Nf by 1 column vector of dimensionless relative fluid fluxes

% Porosity Matrices
Phi_n = comp_mean(phi.^n,M,-1,Grid,1);
Phi_m = spdiags(phi.^m,0,Grid.N,Grid.N);

%% Solve mod. Helmholtz equations 
L  = -D * Phi_n * G + Phi_m;      % system matrix
fs = Phi_m * Zc + Gamma;     % r.h.s.
flux = @(h) -Phi_n * G * h;
res = @(h,cell) L(cell,:)*h - fs(cell); 

%% Solve boundary value problem
hD = solve_lbvp(L,fs+fn,B,BC.g,N);
qD = comp_flux_gen(flux,res,hD,Grid,BC);
pD = hD - Zc;
end