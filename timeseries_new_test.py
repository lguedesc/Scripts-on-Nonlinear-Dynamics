# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                            # library for scientific computing   
import matplotlib.pyplot as plt               # library for plotting
from nldyn import *                           # custom library for ode solvers
from odesystems import *                      # custom library for ode systems
from runge_kutta import integrate_new
import time
# ==============================================================================
# Main Program       
# ==============================================================================
if __name__=='__main__':
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH
    Omega  = 1.6                      
    gamma  = 0.5                      
    zeta   = 0.025                    
    alpha  = -1.5                  
    beta   = 1.8                      
    chi    = 0.05
    kappa  = 0.5
    varphi = 0.05
    p = np.array([Omega, gamma, zeta, alpha, beta, chi, kappa, varphi])
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
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False          # Option to save the simulation result 
    plot_results = True
    plot_only_steady_state = False     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    st = time.time()
    matrix_2 = integrate_new(t0, dt, N, init_cond, odesystem, p)
    et = time.time()
    print(f"New done! ({et - st} s)")
    
    st = time.time()
    matrix_1 = integrate(t0, dt, N, init_cond, odesystem, p)
    et = time.time()
    print(f"Old done! ({et - st} s)")
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        save_timeseries_data(matrix_1, odesystem) 
    # --------------------------------------------------------------------------
    # Define the data to plot 
    # --------------------------------------------------------------------------
    if plot_results == True:
        if plot_only_steady_state == True:
            init_plot = N_tran
        else:
            init_plot = 0
        
        t_plot_1 = matrix_1[init_plot:N, 0]
        x0_plot_1 = matrix_1[init_plot:N, 1]
        x1_plot_1 = matrix_1[init_plot:N, 2]
        
        t_plot_2 = matrix_2[init_plot:N, 0]
        x0_plot_2 = matrix_2[init_plot:N, 1]
        x1_plot_2 = matrix_2[init_plot:N, 2]
        # --------------------------------------------------------------------------
        # Figure 1                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.close('all')                      
        plt.plot(t_plot_1, x0_plot_1, label = "Old")
        plt.plot(t_plot_2, x0_plot_2, label = "New")
        plt.xlabel("Time")
        plt.ylabel("Displacement")
        plt.legend()
        # --------------------------------------------------------------------------
        # Figure 2                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.plot(t_plot_1, x1_plot_1, label = "Old")
        plt.plot(t_plot_2, x1_plot_2, label = "New")
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        # --------------------------------------------------------------------------
        # Figure 4                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.plot(x0_plot_1, x1_plot_1, label = "Old")
        plt.plot(x0_plot_2, x1_plot_2, label = "New")
        plt.xlabel("Displacement")
        plt.ylabel("Velocity")
        # --------------------------------------------------------------------------
plt.show()