function [Grid] = build_grid3D(Grid) % repo (MDOT)
% Author: Marc Hesse
% Date: 09/12/2014, 16 Mar 2018
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
% Grid.dof_zmin = degrees of fredom corrsponding to the cells along the z-min boundary
% Grid.dof_zmax = degrees of fredom corrsponding to the cells along the z-max boundary

% Grid.dof_f_xmin = degrees of fredom corrsponding to the faces at the x-min boundary
% Grid.dof_f_xmax = degrees of fredom corrsponding to the faces at the x-max boundary
% Grid.dof_f_ymin = degrees of fredom corrsponding to the faces at the y-min boundary
% Grid.dof_f_ymax = degrees of fredom corrsponding to the faces at the y-max boundary
% Grid.dof_f_zmin = degrees of fredom corrsponding to the faces at the z-min boundary
% Grid.dof_f_zmax = degrees of fredom corrsponding to the faces at the z-max boundary
% Grid.psi_x0 = reference location for streamfunction
% Grid.psi_dir = diretion of integration for streamfunction 
% + anything else you might find useful
%
% Example call: 
% >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10; 
% >> Grid = build_grid(Grid);

%% Set up cartesian geometry
if ~isfield(Grid,'geom'); Grid.geom = 'cartesian'; end
if ~isfield(Grid,'periodic'); Grid.periodic = 'none'; end
if ~isfield(Grid,'xmin'); Grid.xmin = 0;  end
if ~isfield(Grid,'xmax'); Grid.xmax = 1; end
if ~isfield(Grid,'Nx');   Grid.Nx   = 1; end
Grid.Lx = Grid.xmax-Grid.xmin;    % domain length in x
Grid.dx = Grid.Lx/Grid.Nx;        % dx of the gridblocks

if ~isfield(Grid,'ymin'); Grid.ymin = 0; end
if ~isfield(Grid,'ymax'); Grid.ymax = 1; end
if ~isfield(Grid,'Ny');   Grid.Ny   = 1; end
Grid.Ly = Grid.ymax-Grid.ymin;    % domain length in y
Grid.dy = Grid.Ly/Grid.Ny;        % dy of the gridblocks

if ~isfield(Grid,'zmin'); Grid.zmin = 0; end
if ~isfield(Grid,'zmax'); Grid.zmax = 1; end
if ~isfield(Grid,'Nz');   Grid.Nz   = 1; end
Grid.Lz = Grid.zmax-Grid.zmin;    % domain length in z
Grid.dz = Grid.Lz/Grid.Nz;        % dz of the gridblocks

%% Check input paramters
if Grid.xmin > Grid.xmax; error('xmin > xmax.'); end

%% Number for fluxes
Grid.Nfx = (Grid.Nx+1)*Grid.Ny*Grid.Nz;
Grid.Nfy = Grid.Nx*(Grid.Ny+1)*Grid.Nz;
Grid.Nfz = Grid.Nx*Grid.Ny*(Grid.Nz+1);
Grid.Nf  = Grid.Nfx + Grid.Nfy + Grid.Nfz;


% x, y, z coords of the 12 corners of the domain
Grid.xdom = [Grid.xmin Grid.xmin Grid.xmin Grid.xmin Grid.xmax Grid.xmax Grid.xmax Grid.xmin Grid.xmin Grid.xmin Grid.xmax Grid.xmin; ...
             Grid.xmax Grid.xmin Grid.xmin Grid.xmax Grid.xmax Grid.xmax Grid.xmax Grid.xmin Grid.xmax Grid.xmin Grid.xmax Grid.xmax]; 
Grid.ydom = [Grid.ymin Grid.ymin Grid.ymin Grid.ymax Grid.ymin Grid.ymin Grid.ymax Grid.ymax Grid.ymin Grid.ymin Grid.ymin Grid.ymax;...
             Grid.ymin Grid.ymax Grid.ymin Grid.ymax Grid.ymax Grid.ymin Grid.ymax Grid.ymax Grid.ymin Grid.ymax Grid.ymax Grid.ymax];
Grid.zdom = [Grid.zmin Grid.zmin Grid.zmin Grid.zmin Grid.zmin Grid.zmin Grid.zmin Grid.zmin Grid.zmax Grid.zmax Grid.zmax Grid.zmax;...
             Grid.zmin Grid.zmin Grid.zmax Grid.zmin Grid.zmin Grid.zmax Grid.zmax Grid.zmax Grid.zmax Grid.zmax Grid.zmax Grid.zmax];

% Set up mesh for plotting
% x, y, z coords of the cell centers      
Grid.xc = [Grid.xmin+Grid.dx/2:Grid.dx:Grid.xmax-Grid.dx/2]'; % x-coords of gridblock centers
Grid.yc = [Grid.ymin+Grid.dy/2:Grid.dy:Grid.ymax-Grid.dy/2]'; % y-coords of gridblock centers
Grid.zc = [Grid.zmin+Grid.dz/2:Grid.dz:Grid.zmax-Grid.dz/2]'; % z-coords of gridblock centers
Grid.xf = [Grid.xmin:Grid.dx:Grid.xmax]'; % x-coords of gridblock faces
Grid.yf = [Grid.ymin:Grid.dy:Grid.ymax]'; % y-coords of gridblock faces
Grid.zf = [Grid.zmin:Grid.dz:Grid.zmax]'; % z-coords of gridblock faces

%% Set up dof vectors
Grid.N = Grid.Nx*Grid.Ny*Grid.Nz; % total number of gridblocks
Grid.dof   = [1:Grid.N]';         % cell centered degree of freedom/gridblock number
Grid.dof_f = [1:Grid.Nf]';        % face degree of freedom/face number

%% Boundary dof's
% Boundary cells

Grid.dof_xmin = [];
Grid.dof_xmax = [];
Grid.dof_ymin = [];
Grid.dof_ymax = [];

for i = 1:Grid.N
    if mod(i,Grid.Nx*Grid.Ny)<=Grid.Ny && mod(i,Grid.Nx*Grid.Ny)> 0
        Grid.dof_xmin = [Grid.dof_xmin;i];
    end
    if mod(i - 1,Grid.Nx*Grid.Ny)>=Grid.Nx*Grid.Ny - Grid.Ny
        Grid.dof_xmax = [Grid.dof_xmax;i];
    end
    
    if mod(i,Grid.Ny)==1
        Grid.dof_ymin = [Grid.dof_ymin;i];
    end
    if mod(i,Grid.Ny)==0
        Grid.dof_ymax = [Grid.dof_ymax;i];
    end 
end 

Grid.dof_zmin =[1:Grid.Ny*Grid.Nx]';
Grid.dof_zmax =[Grid.N - Grid.Ny*Grid.Nx+1:Grid.N]';

% Boundary faces
Grid.dof_f_x = Grid.dof_f(1:Grid.Nfx);
Grid.dof_f_x = reshape(Grid.dof_f_x,Grid.Ny,Grid.Nx+1,Grid.Nz);
dummy = Grid.dof_f_x(:,1,:);
Grid.dof_f_xmin = dummy(:);
dummy = Grid.dof_f_x(:,Grid.Nx+1,:);
Grid.dof_f_xmax = dummy(:);

Grid.dof_f_y = Grid.dof_f(Grid.Nfx+1:Grid.Nfx+Grid.Nfy);
Grid.dof_f_y = reshape(Grid.dof_f_y,Grid.Ny+1,Grid.Nx,Grid.Nz);
dummy = Grid.dof_f_y(1,:,:);
Grid.dof_f_ymin = dummy(:);
dummy = Grid.dof_f_y(Grid.Ny+1,:,:);
Grid.dof_f_ymax = dummy(:);


Grid.dof_f_z = Grid.dof_f(Grid.Nf - Grid.Nfz+1:Grid.Nf);
Grid.dof_f_z = reshape(Grid.dof_f_z,Grid.Ny,Grid.Nx,Grid.Nz+1);
dummy = Grid.dof_f_z(:,:,1);
Grid.dof_f_zmin = dummy(:);
dummy = Grid.dof_f_z(:,:,Grid.Nz+1);
Grid.dof_f_zmax = dummy(:);

% Volumes are stored and indexed like unknowns. Areas are stored and indexed like fluxes
Grid.A = [ones(Grid.Nfx,1)*Grid.dy*Grid.dz;...
          ones(Grid.Nfy,1)*Grid.dx*Grid.dz;...
          ones(Grid.Nfz,1)*Grid.dx*Grid.dy;...
          Grid.dx*Grid.dy*Grid.dz];
Grid.V  = ones(Grid.N,1)*Grid.dx*Grid.dy*Grid.dz;


% In preparation for irregular grid, store all cell volumes and face areas
% Volumes are stored and indexed like unknowns. Areas are stored and indexed like fluxes

%{

switch Grid.geom
    case 'cartesian' % 1D and 2D
        Grid.A = [ones(Grid.Nfx,1)*Grid.dy*Grid.dz;...
                  ones(Grid.Nfy,1)*Grid.dx*Grid.dz;
                  Grid.dx*Grid.dy];
        Grid.V  = ones(Grid.N,1)*Grid.dx*Grid.dy*Grid.dz;
    case 'cylindrical_r'
        Grid.A = 2*pi*Grid.xf*Grid.dz;
        Grid.V  = pi*Grid.dz*(Grid.xf(2:Grid.Nx+1).^2-Grid.xf(1:Grid.Nx).^2);
    case 'spherical_r'
        Grid.A = 4*pi*Grid.xf.^2;
        Grid.V  = 4/3*pi*(Grid.xf(2:Grid.Nx+1).^3-Grid.xf(1:Grid.Nx).^3);
    case 'spherical1D'
        Grid.A = 4*pi*Grid.xf.^2;
        Grid.V  = 4*pi*Grid.xc.^2*Grid.dx;%4/3*pi*(Grid.xf(2:Grid.Nx+1).^3-Grid.xf(1:Grid.Nx).^3);
    case 'cylindrical_rz'
%         % assumes: y-dir is radial direction and x-dir is cylinder-axis
%         Grid.A = [repmat(pi*(Grid.yf(2:Grid.Ny+1).^2-Grid.yf(1:Grid.Ny).^2),Grid.Nx+1,1);... % Ax-faces 
%                   repmat(2*pi*Grid.yf*Grid.dz,Grid.Nx,1)]; % Ay-faces
%         Grid.V  = repmat(Grid.A(1:Grid.Nfx),Grid.Nx,1)*Grid.dx;
         % assumes: x-dir is radial direction and y-dir is cylinder-axis
         % x/r-faces:
        x_faces = repmat(2*pi*Grid.xf',Grid.Ny,1)*Grid.dy; 
        y_faces = repmat((pi*(Grid.xf(2:Grid.Nx+1).^2 - Grid.xf(1:Grid.Nx).^2))',Grid.Ny+1,1);
        Grid.A = [x_faces(:);y_faces(:)]; % Ay-faces
        vols = repmat((pi*(Grid.xf(2:Grid.Nx+1).^2 - Grid.xf(1:Grid.Nx).^2)*Grid.dy)',Grid.Ny,1);
        Grid.V  = vols(:);
        % just checking
        if length(x_faces(:)) ~= Grid.Nfx; error('Number of x-face areas inconsistent.'); end
        if length(y_faces(:)) ~= Grid.Nfy; error('Number of y-face areas inconsistent.'); end
        if length(Grid.V) ~= Grid.N; error('Number of cell volumes inconsistent.'); end
    case 'spherical_shell'
        % assumes x = theta = co-lattitude
        if ~isfield(Grid,'R_shell')
            fprintf('Grid.R_shell not initialized. Radius of spherical shell initialized to unity.\n ')
            Grid.R_shell = 1;  
        end
        Acap = 2*pi*Grid.R_shell^2*(1-cos(Grid.xf)); % areas of the sph. caps corresponding to the cell faces
        Aseg = diff(Acap); % areas of the sph. segments of the cells
        Circ = 2*pi*Grid.R_shell*sin(Grid.xf); % circumference of the small circles corresponding to cell faces
        Grid.A = Circ*Grid.dz; % cross-sectional area of the cell faces
        Grid.V = Aseg*Grid.dz; % volume of the cells
    case 'spherical_shell_theta_phi'
        fprintf('spherical_shell_theta_phi:\nNeed to complete volume and area elements\n')
        if ~isfield(Grid,'R_shell')
            fprintf('Grid.R_shell not initialized. Radius of spherical shell initialized to unity.\n ')
            Grid.R_shell = 1;  
        end
        Acap = 2*pi*Grid.R_shell^2*(1-cos(Grid.xf)); % areas of the sph. caps corresponding to the cell faces
        Aseg = diff(Acap); % areas of the sph. segments of the cells
        Aseg_sec = Aseg/Grid.Ny;
        A = repmat(Aseg_sec',Grid.Ny,1);
        Grid.A = A(:);
    otherwise
        error('Unknown grid geometry.')
end

%% Streamfunction
% Set standard integration origin and direction if not specified
if ~isfield(Grid,'psi_x0');  Grid.psi_x0 = 'xmin_ymin'; end
if ~isfield(Grid,'psi_dir'); Grid.psi_dir = 'xy'; end
%}
