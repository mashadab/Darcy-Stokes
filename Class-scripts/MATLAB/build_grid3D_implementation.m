Grid.xmin = 0; Grid.xmax = 3; Grid.Nx = 4;
Grid.ymin = 0; Grid.ymax = 3; Grid.Ny = 3;
Grid.zmin = 0; Grid.zmax = 3; Grid.Nz = 2;

Grid = build_grid3D(Grid);
[D,G,C,I,M] = build_ops3D(Grid);

disp('Grid.dof_xmin'); Grid.dof_xmin
disp('Grid.dof_xmax');Grid.dof_xmax
disp('Grid.dof_ymin');Grid.dof_ymin
disp('Grid.dof_ymax');Grid.dof_ymax
disp('Grid.dof_zmin');Grid.dof_zmin
disp('Grid.dof_zmax');Grid.dof_zmax