# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                            # library for scientific computing   
import matplotlib.pyplot as plt               # library for plotting
from nldyn import *                           # custom library for ode solvers
from odesystems import *                      # custom library for ode systems
# ==============================================================================
# Main Program       
# ==============================================================================
if __name__=='__main__':
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = LR_circuit
    Omega  = 1.0                      
    Amp  = 0.0
    L = 1000
    R = 1000                      
    p = np.array([Omega, Amp, L, R])
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    nP     = 5                     # number of forcing periods
    nDiv   = 1000                     # number of divisions per forcing period
    N      = nP*nDiv                  # number of steps for the integration
    t0     = 0.0                      # initial time
    dt     = (2*np.pi/(Omega*nDiv))   # time step
    tf     = nP*2*np.pi/Omega         # final time
    N_tran = int(0.75*N)              # step in which the transient regime ends 
    init_cond = np.array([1.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False          # Option to save the simulation result 
    plot_results = True
    plot_only_steady_state = False     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    matrix = integrate(t0, dt, N, init_cond, odesystem, p)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        save_timeseries_data(matrix, odesystem) 
    # --------------------------------------------------------------------------
    # Define the data to plot 
    # --------------------------------------------------------------------------
    if plot_results == True:
        if plot_only_steady_state == True:
            init_plot = N_tran
        else:
            init_plot = 0
        
        t_plot = matrix[init_plot:N, 0]
        x0_plot = matrix[init_plot:N, 1]
        # --------------------------------------------------------------------------
        # Figure 1                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.close('all')                      
        plt.plot(t_plot, x0_plot)
        plt.xlabel("Time")
        plt.ylabel("Displacement")
        
plt.show()