"""
--------------------------------------------------------------------------------------------------------------
This script simulates a dynamical system over a given time span. In this specific case, the integration's 
final time is not explicitly defined. Instead, it is determined by the period of a periodic excitation 
function that the system is subjected to.

This approach allows the definition of an excitation period T. An example of such a periodic excitation 
function is shown below, where f_e represents the excitation function and t denotes time:

                        T                                             T   
  |<------------------------------------------>|<------------------------------------------>| 
                                               |                                            |
  f_e                                          |                                            |
  ^                                            |                                            |
  |                           /‾‾‾‾‾‾‾‾‾‾‾\    |                          /‾‾‾‾‾‾‾‾‾‾‾\     |   
  |                          /             \   |                         /             \    |
  |                         /               \  |                        /               \   |
  |                        /                 \ |                       /                 \  |
  |                       /                   \|                      /                   \ |
  |\---------------------/---------------------\---------------------/---------------------\--------> t
  | \                   /                       \                   /                       \
  |  \                 /                         \                 /                         \
  |   \               /                           \               /                           \
  |    \             /                             \             /                             \
  |     \___________/                               \___________/                               \___ ...
  |

The number of integration steps is determined by multiplying the number of excitation periods by the number 
of steps within each period (i.e., the number of divisions per excitation period). This is illustrated below, 
where a period is divided into 4 parts and there are 2 excitation periods. Thus, the total number of 
integration steps is N = 4 * 2 = 8. The final integration time, tf, depends on the size of the period (or the
number of divisions per period).

                        T                                             T   
  |<-------->|<-------->|<-------->|<--------->|<------------------------------------------>| 
     1st div |  2nd div |  3rd div |  4th div  |                                            |
  f_e        |          |          |           |                                            |
  ^          |          |          |           |                                            |
  |          |          |     /‾‾‾‾|‾‾‾‾‾‾\    |                          /‾‾‾‾‾‾‾‾‾‾‾\     |   
  |          |          |    /     |       \   |                         /             \    |
  |          |          |   /      |        \  |                        /               \   |
  |          |          |  /       |         \ |                       /                 \  |
  |          |          | /        |          \|                      /                   \ |
  |\---------------------/---------------------\---------------------/---------------------\--------> t
  | \                   /                       \                   /                       
  |  \                 /                         \                 /                         
  |   \               /                           \               /                           
  |    \             /                             \             /                             
  |     \___________/                               \___________/                               
  |

Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 5 Feb 2025 
--------------------------------------------------------------------------------------------------------------
"""

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