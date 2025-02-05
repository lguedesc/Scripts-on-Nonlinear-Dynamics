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
    odesystem = bistable_EH
    Omega  = 1.6                      
    gamma  = 0.5                      
    zeta   = 0.025                    
    alpha  = -0.5                  
    beta   = 0.5                      
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
    plot_only_steady_state = True     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    matrix = integrate(t0, dt, N, init_cond, odesystem, p)
    # --------------------------------------------------------------------------
    # Additional Calculations
    # --------------------------------------------------------------------------
    Pe_avg = varphi*(RMS(matrix[:, 3])**2)
    Pm = matrix[:, 2]*gamma*np.sin(Omega*matrix[:, 0])
    Pe = varphi*(matrix[:, 3]**2)
    
    Pm_RMS = RMS(Pm)
    Pe_RMS = RMS(Pe)
    
    Pe_avg_SS = varphi*(RMS(matrix[N_tran:N, 3])**2)
    Pm_SS = matrix[N_tran:N, 2]*gamma*np.sin(Omega*matrix[N_tran:N, 0])
    Pe_SS = varphi*(matrix[N_tran:N, 3]**2)
    
    Pm_RMS_SS = RMS(Pm_SS)
    Pe_RMS_SS = RMS(Pe_SS)
    
    print(" ")
    print(f"alpha = {alpha:.2f} | beta = {beta:.2f}")
    print(f"Overall Average Electrical Output Power      = {Pe_avg:.7f}")
    print(f"Overall RMS Electrical Output Power          = {Pe_RMS:.7f}")
    print(f"Overall RMS Mechanical Input Power           = {Pm_RMS:.7f}")
    print(f"Steady State Average Electrical Output Power = {Pe_avg_SS:.7f}")
    print(f"Steady State RMS Electrical Output Power     = {Pe_RMS_SS:.7f}")
    print(f"Steady State RMS Mechanical Input Power      = {Pm_RMS_SS:.7f}")
    print(" ")
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
        x1_plot = matrix[init_plot:N, 2]
        # --------------------------------------------------------------------------
        # Figure 1                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.close('all')                      
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
        # Figure 4                                                               
        # --------------------------------------------------------------------------
        plt.figure()
        plt.plot(x0_plot, x1_plot)
        plt.xlabel("Displacement")
        plt.ylabel("Velocity")
        # --------------------------------------------------------------------------
plt.show()