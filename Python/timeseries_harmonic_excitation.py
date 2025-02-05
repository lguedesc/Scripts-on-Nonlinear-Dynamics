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
    save_results = False             # Option to save the simulation results 
    results_file_extension = "csv"   # Extension of the file containing results
    plot_results = False             # Option to plot results or not
    plot_only_steady_state = False   # Option to plot only the steady state         
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH
    Omega  = 1.6                      
    gamma  = 0.5                      
    zeta   = 0.025                    
    alpha  = -1.05                    
    beta   = 1.04                     
    chi    = 0.05                      
    kappa  = 0.5                      
    varphi = 0.05                     
    ksi_1  = 0.0                        
    ksi_2  = 0.0                      
    p = np.array([Omega, gamma, zeta, alpha, beta, chi, kappa, varphi, ksi_1, ksi_2])
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    nP     = 1000                     # number of forcing periods
    nDiv   = 1000                     # number of divisions per forcing period
    N      = nP*nDiv                  # number of steps for the integration
    t0     = 0.0                      # initial time
    dt     = (2*np.pi/(Omega*nDiv))   # time step
    tf     = nP*2*np.pi/Omega         # final time
    N_tran = int(0.75*N)              # step in which the transient regime ends 
    init_cond = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions                              
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    result = integrate(t0, dt, N, init_cond, odesystem, p)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_results == True:
        save_timeseries_data(result, odesystem, results_file_extension)
    # --------------------------------------------------------------------------
    # Define the data to plot 
    # --------------------------------------------------------------------------
    if plot_only_steady_state == True:
        init_plot = N_tran
    else:
        init_plot = 0
    
    t_plot = result[init_plot:N, 0]
    x0_plot = result[init_plot:N, 1]
    x1_plot = result[init_plot:N, 2]
    x2_plot = result[init_plot:N, 3]
    # --------------------------------------------------------------------------
    # Data Visualization                                                               
    # --------------------------------------------------------------------------
    # Timeseries 
    number_of_plots = result.shape[1] - 1
    fig, axs = plt.subplots(figsize = (10, 5), nrows = number_of_plots, ncols = 1, sharex = True)
    # --------------------------------------------------------------------------
    # Figure 1                                                               
    # --------------------------------------------------------------------------
    plt.close('all')                      
    plt.figure()
    plt.plot(t_plot, x0_plot)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    # --------------------------------------------------------------------------
    # Figure 2                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_plot, x1_plot)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    # --------------------------------------------------------------------------
    # Figure 3                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_plot, x2_plot)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    # --------------------------------------------------------------------------
    # Figure 4                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(x0_plot, x1_plot)
    plt.xlabel("Displacement")
    plt.ylabel("Velocity")
    # --------------------------------------------------------------------------
    plt.show()