[phi_array,phic,tTot_array] = load("output_2D_porosity_wave.mat", "phi_array","tTot_array");

yr2s   = 365.25 * 24 * 60 * 60; %year to second conversion [s/year]

set(groot,'defaultAxesFontName','Times')
set(groot,'defaultAxesFontSize',20)
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')
set(groot, 'DefaultFigureVisible', 'off');


h=figure(4);
set(gcf,'units','points','position',[0,0,700,400])
set(gcf, 'Position', [50 50 700 400])
subplot 131
phi_iter = phi_array(1,:);
phi_iter = reshape(phi_iter,Grid.p.Ny,Grid.p.Nx);
phi_iter(Yc<1000) = 0; %zeroing out the basal increase in porosity ie below 1km

contourf(Xc,Yc,reshape(phi_iter,Grid.p.Ny,Grid.p.Nx),100);
colormap("turbo")
% hold on
% Yc_max = Yc(find(phi_iter==max(phi_iter)));
% Yc_max = max(Yc_max);
% Xc_max = Xc(Yc==Yc_max);
% plot(Yc_max,Xc_max)

xlabel('x-dir [km]','fontsize',14)
ylabel('z-dir [km]','fontsize',14)
axis equal

subplot 133
phi_iter = phi_array(1400,:);
phi_iter = reshape(phi_iter,Grid.p.Ny,Grid.p.Nx);
phi_iter(Yc<1000) = 0; %zeroing out the basal increase in porosity ie below 1km

contourf(Xc/1e3,Yc/1e3,reshape(phi_iter,Grid.p.Ny,Grid.p.Nx),100);
% hold on
% Yc_max = Yc(find(phi_iter==max(phi_iter)));
% Yc_max = max(Yc_max);
% Xc_max = Xc(Yc==Yc_max);
% plot(Xc_max/1e3,Yc_max/1e3,'ro')

c1 = colorbar;
xlabel('x-dir [km]','fontsize',20)
ylabel('z-dir [km]','fontsize',20)
axis equal


subplot 131
phi_iter = phi_array(1,:);
phi_iter = reshape(phi_iter,Grid.p.Ny,Grid.p.Nx);
phi_iter(Yc<1000) = 0; %zeroing out the basal increase in porosity ie below 1km

contourf(Xc/1e3,Yc/1e3,reshape(phi_iter,Grid.p.Ny,Grid.p.Nx),100);
% hold on
% Yc_max = Yc(find(phi_iter==max(phi_iter)));
% Yc_max = max(Yc_max);
% Xc_max = Xc(Yc==Yc_max);
% plot(Xc_max/1e3,Yc_max/1e3,'ro')

xlabel('x-dir [km]','fontsize',20)
ylabel('z-dir [km]','fontsize',20)
axis equal



subplot 132
phi_iter = phi_array(700,:);
phi_iter = reshape(phi_iter,Grid.p.Ny,Grid.p.Nx);
phi_iter(Yc<1000) = 0; %zeroing out the basal increase in porosity ie below 1km

contourf(Xc/1e3,Yc/1e3,reshape(phi_iter,Grid.p.Ny,Grid.p.Nx),100);
% hold on
% Yc_max = Yc(find(phi_iter==max(phi_iter)));
% Yc_max = max(Yc_max);
% Xc_max = Xc(Yc==Yc_max);
% plot(Xc_max/1e3,Yc_max/1e3,'ro')
xlabel('x-dir [km]','fontsize',20)
ylabel('z-dir [km]','fontsize',20)
axis equal

saveas(h,sprintf('2D_porosity_wave1e-3.pdf'));
