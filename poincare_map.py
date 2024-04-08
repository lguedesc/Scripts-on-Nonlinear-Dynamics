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
    alpha  = -1.39                    
    beta   = 1.16                     
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
    N_tran = int(0.75*N)              # step in which the transient state ends 
    nP_tran = int(0.75*nP)            # period in which the transient state ends   
    init_cond = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = True          # Option to save the simulation result 
    plot_only_steady_state = True     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    int_result, poinc_result = integrate_and_poincare_map(nP, nDiv, t0, dt, 
                                                          init_cond, odesystem, 
                                                          p, nP_tran)
    # --------------------------------------------------------------------------
    # Perform Additional calculations with the result of the integration
    # --------------------------------------------------------------------------
    overall_x0_rms = RMS(int_result[:, 1])     # Overall RMS of the displacement
    overall_x1_rms = RMS(int_result[:, 2])     # Overall RMS of the velocity
    overall_x2_rms = RMS(int_result[:, 3])     # Overall RMS of the voltage
    overall_Pout = varphi*(overall_x2_rms**2)  # Overall Electrical output power
    
    x0_rms = RMS(int_result[N_tran:, 1])       # RMS of the steady state displacement
    x1_rms = RMS(int_result[N_tran:, 2])       # RMS of the steady state velocity
    x2_rms = RMS(int_result[N_tran:, 3])       # RMS of the steady state voltage
    Pout = varphi*(x2_rms**2)                  # Electrical output power of the steady state
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
        
        file_additional = open("additional_calculations.txt", "w")
        file_additional.write(f"Overall x0_rms = {overall_x0_rms:.15f}\n")
        file_additional.write(f"Overall x1_rms = {overall_x1_rms:.15f}\n")
        file_additional.write(f"Overall x2_rms = {overall_x2_rms:.15f}\n")
        file_additional.write(f"x0_rms         = {x0_rms:.15f}\n")
        file_additional.write(f"x1_rms         = {x1_rms:.15f}\n")
        file_additional.write(f"x2_rms         = {x2_rms:.15f}\n")
        file_additional.write(f"Overall Pout   = {overall_Pout:.15f}\n")
        file_additional.write(f"Pout           = {Pout:.15f}\n")
    # --------------------------------------------------------------------------
    # Show the solution                              
    # --------------------------------------------------------------------------
    print("Result of additional calculations:")
    print(f"Overall x0_rms = {overall_x0_rms:.15f}")
    print(f"Overall x1_rms = {overall_x1_rms:.15f}")
    print(f"Overall x2_rms = {overall_x2_rms:.15f}")
    print(f"x0_rms         = {x0_rms:.15f}")
    print(f"x1_rms         = {x1_rms:.15f}")
    print(f"x2_rms         = {x2_rms:.15f}")
    print(f"Overall Pout   = {overall_Pout:.15f}")
    print(f"Pout           = {Pout:.15f}")
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