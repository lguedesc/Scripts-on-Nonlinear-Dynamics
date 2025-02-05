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
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH
    Omega  = 0.0                # Control parameter                      
    gamma  = 0.3                      
    zeta   = 0.025                    
    alpha  = -0.7                    
    beta   = 0.25                     
    chi    = 0.05                      
    kappa  = 0.5                      
    varphi = 0.05                     
    ksi_1  = 0.0                        
    ksi_2  = 0.0            
    p = np.array([Omega, gamma, zeta, alpha, beta, chi, kappa, varphi, ksi_1, ksi_2])              
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    tol          = 0.001                          # numerical tolerance for loop
    Cpar_i       = 0.01                           # initial forcing frequency
    Cpar_f       = 2.0                            # final forcing frequency
    bifurc_steps = 500                            # number of bifurcation steps
    nP           = 1000                           # number of forcing periods
    nDiv         = 1000                           # number of divisions per period
    N            = nP*nDiv                        # number of steps for the integration
    t0           = 0.0                            # initial time
    transient    = int(0.75*nP)                   # step the transient regime ends 
    init_cond    = np.array([-1.0, 0.0, 0.0])     # initial conditions
    cpar_indx = 0                                 # index of the control parameter (in this case Omega = p[0], so the index is 0)
    # choose bifurcation mode: "reset_IC" to reset initial conditions in every 
    # step, or "follow_attractor" to use as initial condition the final condition 
    # of the previous step
    bifurc_mode = "reset_IC"    
    # --------------------------------------------------------------------------
    # Inputs (program options)                                                                 
    # --------------------------------------------------------------------------
    save_output_file = False             # Option to save the simulation result                                  
    #========================================================================#
    # Solution                                                               #
    #========================================================================#    
    n, bifurc_result = bifurcation_diagram(cpar_indx, Cpar_i, Cpar_f, 
                                           bifurc_steps, nP, nDiv, transient, 
                                           t0, init_cond, p, odesystem, 
                                           bifurc_mode = bifurc_mode)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_output_file == True:
        file = open("output_bifurcation.csv", "w")
        file.write("Cpar x[0]_poinc x[1]_poinc x[2]_poinc\n")
        for i in range(len(bifurc_result)):
            file.write("%f %f %f %f\n" % (bifurc_result[i,0], bifurc_result[i,1], 
                                         bifurc_result[i,2], bifurc_result[i,3]))
        file.close() 
    # --------------------------------------------------------------------------
    # Create figure and plot data
    # --------------------------------------------------------------------------
    plt.close('all')                                        # close all figures
    fig = plt.figure(1, layout = "constrained")
    ax = fig.add_subplot(1,1,1)
    ax.scatter(bifurc_result[:, 0], bifurc_result[:, 1], s = 1, color = 'black')
    ax.set_xlabel(r'$Control Parameter$')
    ax.set_ylabel(r'$x$ (Poincaré Map)')
    ax.set_title(r'Bifurcation Diagram')
    
    plt.show()