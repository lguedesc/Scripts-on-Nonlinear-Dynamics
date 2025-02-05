"""
--------------------------------------------------------------------------------------------------------------
This script performs the simulation of a dynamical system across a given time span.

Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 5 Feb 2025 
--------------------------------------------------------------------------------------------------------------
"""
# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                # famous library for scientific computing   
from include.nldyn import *       # custom library for nonlinear analysis 
from include.odesystems import *  # custom library with ODE systems definitions
# ==============================================================================
# Main Program       
# ==============================================================================
if __name__=='__main__':
    # --------------------------------------------------------------------------
    # Inputs (main program options)                                                                 
    # --------------------------------------------------------------------------
    save_results = False            # Option to save the simulation results 
    results_file_extension = "csv"  # Extension of the file containing results
    plot_results = True             # Option to plot results or not
    save_plot = True                # Option to save plot figures
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = lorenz
    sigma = 10.0                      
    rho   = 28.0
    beta  = 8.0/3.0                                        
    p = np.array([sigma, rho, beta]) # Define parameter array in the same order 
                                     # defined in system function declaration
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    t0     = 0.0                           # initial time
    dt     = 0.01                          # time step
    tf     = 100.0                         # final time 
    N      = int(tf/dt)                         # number of integration time steps
    IC     = np.array([0.0, 1.0, 1.05])    # initial conditions                              
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    result = integrate(t0, dt, N, IC, odesystem, p)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_results == True:
        save_timeseries_data(result, odesystem, results_file_extension)
    # --------------------------------------------------------------------------
    # Data Visualization                                                               
    # --------------------------------------------------------------------------
    if plot_results == True:
        t_span = [10, 40]
        visualize_timeseries(result, [2, 0, 1], t_span = t_span, save = save_plot)
        visualize_phase_subspaces(result, [[0,1], [0,2], [1,2]], t_span = t_span, save = save_plot)
        plt.show()
    