# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                            # library for scientific computing   
import matplotlib.pyplot as plt               # library for plotting
from runge_kutta import *                     # custom library for ode solvers
from odesystems import bistable_EH            # custom library for ode systems
# ============================================================================== 
# Define Functions                                
# ============================================================================== 
def integrate(t0, dt, n, x0, func, p):
    # This function apply the steps of integration for the simulation
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # t0:   initial time
    # x0:   vector containing the initial conditions (initial state variables)
    # n:    number of integration steps
    # func: system of first order ODEs to be solved
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------
    # Create array with the size of data full of zeros
    data = np.zeros((n + 1, len(x0) + 1))
    # Assign the first row of data with the initial conditions and initial time
    data[0, 0] = t0
    data[0, 1:] = x0
    # Integrate over the number of steps
    for i in range(n):
        x = rk4(func, data[i, 1:], dt, data[i, 0], p)
        # Update time
        data[i + 1, 0] = data[i, 0] + dt        
        # Update the value of state variables
        data[i + 1, 1:] = x
        
    return data
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
    N_tran = int(0.75*N)              # step in which the transient regime ends 
    init_cond = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False          # Option to save the simulation result 
    plot_only_steady_state = True     # Option to plot only the steady state                                  
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    matrix = integrate(t0, dt, N, init_cond, odesystem, p)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        arq = open("output_bistable_EH.txt", "w")
        arq.write("time x[0] x[1] x[2]\n")
        for i in range(N):
            arq.write("%f %f %f %f\n"%(matrix[i,0], matrix[i,1], matrix[i,2], 
                                       matrix[i,3]))
        arq.close() 
    # --------------------------------------------------------------------------
    # Define the data to plot 
    # --------------------------------------------------------------------------
    if plot_only_steady_state == True:
        init_plot = N_tran
    else:
        init_plot = 0
    
    t_plot = matrix[init_plot:N, 0]
    x0_plot = matrix[init_plot:N, 1]
    x1_plot = matrix[init_plot:N, 2]
    x2_plot = matrix[init_plot:N, 3]
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