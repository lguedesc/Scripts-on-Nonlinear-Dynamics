# ============================================================================== 
# Load Libraries
# ============================================================================== 
import numpy as np                            # library for scientific computing   
import matplotlib.pyplot as plt               # library for plotting
from runge_kutta import *                     # custom library for ode solvers
from odesystems import bistable_EH            # custom library for ode systems
# ==============================================================================
# Main Program       
# ==============================================================================
if __name__=='__main__':
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH
    Omega  = 0.0 # Control parameter                      
    gamma  = 0.5                      
    zeta   = 0.025                    
    alpha  = -1.0                    
    beta   = 1.0                     
    chi    = 0.05                      
    kappa  = 0.5                      
    varphi = 0.05                     
    ksi_1  = 0.0                        
    ksi_2  = 0.0            
    p = np.array([Omega, gamma, zeta, alpha, beta, chi, kappa, varphi, ksi_1, ksi_2])              
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    tol       = 0.001                    # numerical tolerance for loop
    Omega_i   = 0.01                     # initial forcing frequency
    Omega_f   = 0.14 + tol               # final forcing frequency
    Omega_inc = 5e-4                     # forcing frequency increment
    nP        = 200                      # number of forcing periods
    nDiv      = 1000                     # number of divisions per period
    N         = nP*nDiv                  # number of steps for the integration
    t0        = 0.0                      # initial time
    transient = int(0.75*nP)             # step the transient regime ends 
    init_cond = np.array([-1.0, 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False             # Option to save the simulation result                                  
    #========================================================================#
    # Solution                                                               #
    #========================================================================#    
    result = []                         # Vector that poincaré map and control parameter
    x = init_cond                       # set x as initial conditions
    OMEGA = np.arange(Omega_i, Omega_f, Omega_inc)   # Excitation Frequency Vector
    # Loop through the values of control parameters (in this case, Omega)    
    for Omg in OMEGA: 
        # Update control parameter
        p[0] = Omg
        # Reset variables
        t = t0                          # time
        dt = (2*np.pi/(Omg*nDiv))       # time step        
        x = init_cond                   # Comment if you want to vary initial condition based on the evolution of the system
        print(f'Omg = {Omg}')           # Print to Monitor Progress
        # Solution of each time series
        for i in range(nP):
            for k in range(nDiv):
                x = rk4(odesystem, x, dt, t, p)             # Call Runge-Kutta 
                # If is greater than Transient Regime
                if i > transient:                              
                    if k == 1:                              # Poincaré Section
                        result.append([Omg]+list(x))        # Poincaré Map
                
                t = t + dt                                  # Increase Time 
        
        result_bifurc = np.array(result)                    # Stores All Results
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        arq = open("output_bifurcation.csv", "w")
        arq.write("Var x_poinc xdot_poinc v_poinc\n")
        for i in range(len(result_bifurc)):
            arq.write("%f %f %f %f\n" % (result_bifurc[i,0], result_bifurc[i,1], 
                                         result_bifurc[i,2], result_bifurc[i,3]))
        arq.close() 
    # --------------------------------------------------------------------------
    # Create figure and plot data
    # --------------------------------------------------------------------------
    plt.close('all')                                        # close all figures
    fig = plt.figure(1, layout = "constrained")
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(result_bifurc[:, 0], result_bifurc[:, 1], s = 1, color = 'black')
    ax.set_xlabel(r'$f_0$')
    ax.set_ylabel(r'$x$ (Poincaré Map)')
    ax.set_title(r'Bifurcation Diagram')
    
    plt.show()