function [Grid] = build_stokes_grid(Gridp) % repo
% authors: Marc Hesse, Evan Carnahan
% date: 27 Sept 2019
% Description: This function builds the three stagged grids used for the
% discretization of the variable viscosity Stokes equation. The primary
% grid is the pressure/mass conservation grid. It is a standard staggered
% generated by build_grid.m, with pressures in cell centers and velocity
% components on the associated faces.

%% Build pressure grid
Gridp = build_grid2D(Gridp);

%% Build x-velovity grid
dxh = Gridp.dx/2;
Gridx.xmin = Gridp.xmin-dxh; Gridx.xmax = Gridp.xmax+dxh; Gridx.Nx = Gridp.Nx+1;
Gridx.ymin = Gridp.ymin;     Gridx.ymax = Gridp.ymax;     Gridx.Ny = Gridp.Ny;
Gridx = build_grid2D(Gridx);

%% Build y-velocity grid
 dyh = Gridp.dy/2;
Gridy.xmin = Gridp.xmin;     Gridy.xmax = Gridp.xmax;     Gridy.Nx = Gridp.Nx;
Gridy.ymin = Gridp.ymin-dyh; Gridy.ymax = Gridp.ymax+dyh; Gridy.Ny = Gridp.Ny+1;
Gridy = build_grid2D(Gridy);

Grid.p = Gridp;
Grid.x = Gridx;
Grid.y = Gridy;

%% Helpful quantities
Grid.N = Grid.p.Nf+Grid.p.N;

%% Boundary dof's
% Unknown vector is ordered: u = [vx;vy;p]

% Normal velocities on bnd's
Grid.dof_xmin_vx = Grid.x.dof_xmin;
Grid.dof_xmax_vx = Grid.x.dof_xmax;
Grid.dof_ymin_vy = Grid.p.Nfx+Grid.y.dof_ymin;
Grid.dof_ymax_vy = Grid.p.Nfx+Grid.y.dof_ymax;

% Tangential velocities on bnd's
Grid.dof_xmin_vy = Grid.p.Nfx+Grid.y.dof_xmin;
Grid.dof_xmax_vy = Grid.p.Nfx+Grid.y.dof_xmax;
Grid.dof_ymin_vx = Grid.x.dof_ymin;
Grid.dof_ymax_vx = Grid.x.dof_ymax;
% excluding extreme faces
Grid.dof_xmin_vt = Grid.dof_xmin_vy(2:Grid.p.Ny); 
Grid.dof_xmax_vt = Grid.dof_xmax_vy(2:Grid.p.Ny);
Grid.dof_ymin_vt = Grid.dof_ymin_vx(2:Grid.p.Nx);
Grid.dof_ymax_vt = Grid.dof_ymax_vx(2:Grid.p.Nx);

% Pressures on bnd's
Grid.dof_xmin_p = Grid.p.Nf+Grid.p.dof_xmin;
Grid.dof_xmax_p = Grid.p.Nf+Grid.p.dof_xmax;
Grid.dof_ymin_p = Grid.p.Nf+Grid.p.dof_ymin;
Grid.dof_ymax_p = Grid.p.Nf+Grid.p.dof_ymax;

% Pressure constraint in center of domain
Grid.dof_pc = Grid.p.Nf+round(Grid.p.N/2);
Grid.dof_pc_comp_col = Grid.p.Nf+1;

%% Common useful BC's
% Penetration - set normal velocities on all boundaries to zero
Grid.dof_pene = [Grid.dof_xmin_vx;Grid.dof_xmax_vx;Grid.dof_ymin_vy;Grid.dof_ymax_vy];
Grid.N_pene = length(Grid.dof_pene);

% Slip - set all tangential velocities on all boundaries to zero
Grid.dof_slip = [Grid.dof_xmin_vt;Grid.dof_xmax_vt;Grid.dof_ymin_vt;Grid.dof_ymax_vt];
Grid.N_slip = length(Grid.dof_slip);

% Solid boundary - no slip and no penetration
Grid.dof_solid_bnd = unique([Grid.dof_pene;Grid.dof_slip]);
Grid.N_solid_bnd = length(Grid.dof_solid_bnd);

