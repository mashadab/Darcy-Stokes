function [A] = flux_upwind(v,Grid)
% author: Marc Hesse
% date: 15 April 2015, 8 Nov 2017
% Description:
% This function computes the upwind flux matrix from the flux vector.
%
% Input:
% v = Nf by 1 flux vector from the flow problem.
% Grid = structure containing all pertinent information about the grid.
%
% Output:
% A = Nf by Nf matrix contining the upwinded fluxes
%
% Example call:
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10;
% >> Grid = build_grid(Grid);
% >> v = ones(Grid.Nf,1);
% >> [A] = flux_upwind(v,Grid);

Nx = Grid.Nx;  

%% One dimensinal
% too make this work for 1D in y-dir need to replace Nx with N!
vn = min(v(1:Nx),0);
vp = max(v(2:Nx+1),0);
A  = spdiags([vn vp],[0,-1],Nx+1,Nx);
