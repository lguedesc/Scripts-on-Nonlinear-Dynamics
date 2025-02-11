"""
--------------------------------------------------------------------------------------------------------------
This script simulates a dynamical system over a given time span. In this specific case, the integration's 
final time is not explicitly defined. Instead, it is determined by the period, T, of a periodic excitation 
function that the system is subjected to.

An example of such a periodic excitation function is shown below, where F represents the excitation function 
and t denotes time:
                                                                        
  |<------------------- T -------------------->|<------------------- T -------------------->| 
                                               |                                            |
  F                                            |                                            |
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

  |<------------------- T -------------------->|<------------------- T -------------------->| 
  |<-------->|<-------->|<-------->|<--------->|                                            | 
     1st div |  2nd div |  3rd div |  4th div  |                                            |
  F          |          |          |           |                                            |
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

For a single frequency excitation, the period T is defind as T = 2*pi/omega, where omega is the angular
frequency (in rad/s), given by omega = 2*pi/T = 2*pi*f, with f representing the ordinary frequency (in Hz).
For a multi-frequency excitation, T is defined as a common period between all the excitation frequencies
that compose the excitation signal. The common period is found using the least common multiple (LCM) between 
all the excitation frequencies. The function that returns the period T is defined in the nldyn.py file.

This method of defining the time span based on the excitation characteristics is particularly useful for 
determining the Poincaré map in systems with periodic excitation.

Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 6 Feb 2025 
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
    save_results = False            # Option to save the simulation results 
    results_file_extension = "csv"  # Extension of the file containing results
    plot_results = True             # Option to plot results or not     
    save_plot = True                # Option to save plot image
    fig_extension = "pdf"           # Plot image extension (if saved)
    # --------------------------------------------------------------------------
    # Input (system constant parameters)                                                                 
    # --------------------------------------------------------------------------
    odesystem = bistable_EH_multi_freq
    Omega_1    = 1.6                      
    gamma      = 0.5                      
    zeta       = 0.025                    
    alpha      = -1.05                    
    beta       = 1.04                     
    chi        = 0.05                      
    kappa      = 0.5                      
    varphi     = 0.05                     
    varepsilon = 0.0                        
    mu         = 0.0
    Omega_2    = 0.5                      
    p = np.array([Omega_1, gamma, zeta, alpha, beta, chi, kappa, varphi, 
                  varepsilon, mu, Omega_2])
    # --------------------------------------------------------------------------
    # Input (simulation parameters)                                                                 
    # --------------------------------------------------------------------------
    Omegas = [Omega_1, Omega_2]                 # excitation frequencies
    nP     = 2000                               # number of excitation periods
    nDiv   = 1000                               # divisions per period
    N      = nP*nDiv                            # number of integration steps 
    T      = define_period_of_excitation(Omegas) # period of excitation
    t0     = 0.0                                # initial time
    dt     = T/nDiv                             # time step (T/nDiv)
    tf     = N*dt                               # final time (N*dt)
    N_tran = int(0.75*N)                        # step when transient state ends
    IC     = np.array([np.sqrt(-alpha/beta), 0.0, 0.0]) # initial conditions
    # --------------------------------------------------------------------------
    # Solution                              
    # --------------------------------------------------------------------------
    result = integrate(t0, dt, N, IC, odesystem, p)
    # --------------------------------------------------------------------------
    # Save the solution                              
    # --------------------------------------------------------------------------
    if save_results == True:
        save_timeseries_data(result, odesystem, results_file_extension)
    # --------------------------------------------------------------------------
    # Data Visualization                                                               
    # --------------------------------------------------------------------------
    if plot_results == True:
        t_span = [N_tran*dt, tf]
        #t_span = None
        visualize_timeseries(result, [0, 1, 2], t_span = t_span, 
                             save = save_plot, fig_ext = fig_extension)
        visualize_phase_subspaces(result, [[0,1], [0,2], [1,2]], 
                                  t_span = t_span, save = save_plot,
                                  fig_ext = fig_extension)
        plt.show()