# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                            # library for scientific computing   
import matplotlib.pyplot as plt               # library for plotting
from nldyn import *                           # custom library for ode solvers
from odesystems import bistable_EH            # custom library for ode systems
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
    N_tran = int(0.77*N)              # step in which the transient regime ends 
    nP_tran = int(0.77*nP)               
    init_cond = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False          # Option to save the simulation result 
    plot_only_steady_state = True     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    int_result, poinc_result = integrate_and_poincare_map(nP, nDiv, t0, dt, 
                                                          init_cond, odesystem, 
                                                          p, nP_tran)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        file = open("integration_bistable_EH.csv", "w")
        file.write("time x[0] x[1] x[2]\n")
        for i in range(N):
            file.write("%.15f %.15f %.15f %.15f\n" % (int_result[i,0], 
                                                      int_result[i,1], 
                                                      int_result[i,2], 
                                                      int_result[i,3]))
        file.close() 
    
        file_poinc = open("poinc_bitable_EH.csv", "w")
        file_poinc.write("x[0] x[1] x[2]\n")
        for i in range(len(poinc_result)):
            file_poinc.write("%.15f %.15f %.15f %.15f\n" % (poinc_result[i, 0], 
                                                            poinc_result[i, 1],
                                                            poinc_result[i, 2], 
                                                            poinc_result[i, 3]))
        file_poinc.close()
    # --------------------------------------------------------------------------
    # Define the data to plot 
    # --------------------------------------------------------------------------
    if plot_only_steady_state == True:
        init_plot = N_tran
    else:
        init_plot = 0
    
    t_plot = int_result[init_plot:N, 0]
    x0_plot = int_result[init_plot:N, 1]
    x1_plot = int_result[init_plot:N, 2]
    x2_plot = int_result[init_plot:N, 3]
    # --------------------------------------------------------------------------
    # Figure 1                                                               
    # --------------------------------------------------------------------------
    plt.close('all')                      
    plt.figure()
    plt.plot(t_plot, x0_plot, zorder = 0)
    plt.scatter(poinc_result[:,0], poinc_result[:,1], color = "black", zorder = 1)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    # --------------------------------------------------------------------------
    # Figure 2                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_plot, x1_plot, zorder = 0)
    plt.scatter(poinc_result[:,0], poinc_result[:,2], color = "black", zorder = 1)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    # --------------------------------------------------------------------------
    # Figure 3                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_plot, x2_plot, zorder = 0)
    plt.scatter(poinc_result[:,0], poinc_result[:,3], color = "black", zorder = 1)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    # --------------------------------------------------------------------------
    # Figure 4                                                               
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(x0_plot, x1_plot, zorder = 0)
    plt.scatter(poinc_result[:,1], poinc_result[:,2], color = "black", zorder = 1)
    plt.xlabel("Displacement")
    plt.ylabel("Velocity")
    # --------------------------------------------------------------------------
    plt.show()