function [D,Edot,Dp,Gp,Z,I,Ms]=build_stokes_ops_variable_visc(Grid) % repo
% authors: Mohammad Afzal Shadab, Marc Hesse, Evan Carnahan
% date: 23 May 2023

%% Build operators for each variable
[Dp,Gp,~,Ip,Mp] = build_ops2D(Grid.p);
[Dx,Gx,~,Ix,Mx] = build_ops2D(Grid.x);
[Dy,Gy,~,Iy,My] = build_ops2D(Grid.y);

%% Extract x and y components of the velocity operators
Gxx = Gx(1:Grid.x.Nfx,:); Gxy = Gx(Grid.x.Nfx+1:Grid.x.Nf,:);
Gyx = Gy(1:Grid.y.Nfx,:); Gyy = Gy(Grid.y.Nfx+1:Grid.y.Nf,:);

Dxx = Dx(:,1:Grid.x.Nfx); Dxy = Dx(:,Grid.x.Nfx+1:Grid.x.Nf);
Dyx = Dy(:,1:Grid.y.Nfx); Dyy = Dy(:,Grid.y.Nfx+1:Grid.y.Nf);

%Mean operators
Mpx = Mp(1:Grid.p.Nfx,:); Mpy = Mp(Grid.p.Nfx+1:Grid.p.Nf,:);
Mxx = Mx(1:Grid.x.Nfx,:); Mxy = Mx(Grid.x.Nfx+1:Grid.x.Nf,:);
Myx = My(1:Grid.y.Nfx,:); Myy = My(Grid.y.Nfx+1:Grid.y.Nf,:);
Mc  = Mxy * Mpx; %Mean for cross terms 

%Mean operator
Iy = eye(Grid.p.Ny); Ix = eye(Grid.p.Nx); 

Iyy = [Iy(1,:);Iy;Iy(end,:)];
Ixx = [Ix(1,:);Ix;Ix(end,:)];

Iyy = kron(Ix,Iyy); Ixx = kron(Ixx,Iy); %Mean operators are just cell center location

Ms  = [Ixx;Iyy;Mc]; %Mean stokes operator

% Zero blocks
Zxy = spalloc(Grid.x.Nfx,Grid.y.N,0);
Zyx = spalloc(Grid.y.Nfy,Grid.x.N,0);

%% Assemble Stokes operators
% Symmetric derivative
Edot = [Gxx,Zxy; ...
        Zyx,Gyy; ...
      Gxy/2,Gyx/2];

 % Divergence of deviatoric stess tensor
D = [Dxx, Zyx', Dxy; ...
     Zxy',Dyy , Dyx];
 
 % Zero block for the system matrix L
 Z = spalloc(Grid.p.N,Grid.p.N,0);

 % Identity for all dof's
 I = speye(Grid.N);
 
subplot 141
title('Dxx');
spy(Dxx);
subplot 142
title('Dxy');
spy(Dxy);
subplot 143
title('Gxx');
spy(Gxx);
subplot 144
title('Gxy');
spy(Gxy);
 
