"""
--------------------------------------------------------------------------------------------------------------
This script simulates a periodically excited dynamical system over a specified time span and computes its 
corresponding Poincaré map.

The Poincaré map is constructed by interpreting each excitation period as a repeating time cycle. In this 
perspective, time can be visualized as folding into a circular (or toroidal) structure, where each period 
begins and ends at the same point. A Poincaré section is defined at a specific position within this cycle, 
capturing the system's state whenever this point in the folded time is reached.

Algorithmically, this is achieved by defining the final integration time based on the period  T  of the 
system's periodic excitation function, rather than specifying it explicitly. The number of integration 
steps is determined by multiplying the number of excitation periods by the number of steps within each 
period (i.e., the number of divisions per excitation period). 

The Poincaré section is defined at a specific division within each excitation period, corresponding to the 
same position in the folded time structure. This is illustrated below, where F is the excitation function. 
The period of excitation is divided into 4 parts and there are 2 excitation periods. Thus, the total number 
of integration steps is N = 4 * 2 = 8. The final integration time, tf, depends on the size of the period (or 
the number of divisions per period). The poincare section, in this case, is defined in the end of every 3rd 
division of each period (marked as X).

  |<------------------- T -------------------->|<------------------- T -------------------->| 
  |<-------->|<-------->|<-------->|<--------->|                ... |<--------->|<--------->| 
     1st div |  2nd div |  3rd div |  4th div  |                    |  3rd div  |  4th div  |
  F          |          |          X           |                    |           X           |
  ^          |          |          X           |                    |           X           |
  |          |          |     /‾‾‾‾X‾‾‾‾‾‾\    |                    |     /‾‾‾‾‾X‾‾‾‾‾\     |   
  |          |          |    /     X       \   |                    |    /      X      \    |
  |          |          |   /      X        \  |                    |   /       X       \   |
  |          |          |  /       X         \ |                    |  /        X        \  |
  |          |          | /        X          \|                    | /         X         \ |
  |\---------------------/---------X-----------\---------------------/----------X----------\---------------> t
  | \                   /          X            \                   /           X           \
  |  \                 /           X             \                 /            X            \
  |   \               /            X              \               /             X             \
  |    \             /             X               \             /              X              \
  |     \___________/              X                \___________/               X               \_________ ...
  |                                X                                            X
                                Poincaré                                     Poincaré
                                section                                      section
  
Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 6 Feb 2025 
--------------------------------------------------------------------------------------------------------------
"""
# ==============================================================================
# Load Libraries
# ==============================================================================
import numpy as np                # famous library for scientific computing   
import matplotlib.pyplot as plt   # famous library for plotting
from include.nldyn import *       # custom library for nonlinear analysis 
from include.odesystems import *  # custom library with ODE systems definitions
# ==============================================================================
# Main Program       
# ==============================================================================
if __name__=='__main__':
    # --------------------------------------------------------------------------
    # Inputs (main program options)                                                                 
    # --------------------------------------------------------------------------
    save_results = True             # Option to save the simulation results 
    file_extension = "csv"          # Extension of files containing results
    plot_results = True             # Option to plot results or not
    save_plot = True                # Option to save plot figures
    fig_extension = "pdf"           # Plot image extension (if saved)
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH
    Omega      = 1.6                      
    gamma      = 0.5                      
    zeta       = 0.025                    
    alpha      = -1.05                    
    beta       = 1.04                     
    chi        = 0.05                      
    kappa      = 0.5                      
    varphi     = 0.05                     
    varepsilon = 0.0                        
    mu         = 0.0        
    p = np.array([Omega, gamma, zeta, alpha, beta, chi, kappa, varphi, 
                  varepsilon, mu])
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    nP      = 1000                     # number of forcing periods
    nDiv    = 1000                     # number of divisions per forcing period
    N       = nP*nDiv                  # number of steps for the integration
    T       = define_period_of_excitation(Omega) # period of excitation
    t0      = 0.0                      # initial time
    dt      = T/nDiv                   # time step 
    tf      = N*dt                     # final time
    N_tran  = int(0.75*N)              # step in which the transient state ends 
    nP_tran = int(0.75*nP)            # period in which the transient state ends   
    IC      = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    int_result = integrate(t0, dt, N, IC, odesystem, p)
    poinc_result = poincare_map_periodic_excitation(int_result, nP, nDiv, nP_tran)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_results == True:
        save_timeseries_data(int_result, odesystem, file_extension, 
                             decimal_precision = 15)
        
        save_poincare_map_data(poinc_result, odesystem, file_extension,
                               decimal_precision = 15)
    # --------------------------------------------------------------------------
    # Data Visualization                                                               
    # --------------------------------------------------------------------------
    if plot_results == True:
        t_span = [N_tran*dt, tf]
        fig, axs = visualize_phase_subspaces(int_result, [[0,1], [0,2], [1,2]], 
                                             t_span = t_span, save = save_plot,
                                             fig_ext = fig_extension)
        fig, axs = visualize_poincare_map(poinc_result, [[0,1], [0,2], [1,2]],
                                          save = save_plot, 
                                          fig_ext = fig_extension, axs = axs)
        
        plt.show()
    