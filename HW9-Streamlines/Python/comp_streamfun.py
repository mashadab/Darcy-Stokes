import numpy as np
import scipy.sparse as sp

def comp_streamfun(q,Grid):
    
    # author: Mohammad Afzal Shadab, Marc Hesse
    # date: 
    # Description: The streamfunction of a numerical solution is computed at 
    #              the cell corners given the vector of numerical fluxes across 
    #              the cell faces.
    # Input: q = [qx;qy] Nf by 1 vector of all face fluxes. x-fluxes first.
    #        Grid = structure containing all information about the grid.
    # Output: PSI = Ny+1 by Nx+1 matrix containing the values of the
    #                     streamfunction along the cell corners.
    #         psi_min = minimum of PSI
    #         psi_max = maximum of PSI
    
    # Integrate horizontally along ymin boundary first then vertically into the domain.
    Qymin = np.vstack([0, np.cumsum(q[Grid.dof_f_ymin-1,:]*Grid.A[Grid.dof_f_ymin-1,:],axis=1)])  # Integral of flow into ymin boundary, this is a vector
    print(Qymin)
    print(np.shape(Qymin))
    '''
    Qx = reshape(q(1:Grid.Nfx).*Grid.A(1:Grid.Nfx),Grid.Ny,Grid.Nx+1) # Horizontal fluxes in Ny by Nx+1 matrix
    PSI = cumsum([-Qymin;Qx],1); # integrals into domain with Qymin as initial value
    psi_min = min(PSI(:));  psi_max = max(PSI(:))
    '''
    
    return #PSI,psi_min,psi_max