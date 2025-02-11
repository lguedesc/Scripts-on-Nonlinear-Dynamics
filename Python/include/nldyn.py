"""
--------------------------------------------------------------------------------------------------------------
This file contains the implementation of the functions related to the analysis and result visualization of
dynamical systems simulations using a nonlinear dynamics perspective.

Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 5 Feb 2025 
--------------------------------------------------------------------------------------------------------------
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt   # famous library for plotting
from include.solvers import *
# ------------------------------------------------------------------------------
# Solution methods and nonlinear dynamics tools
# ------------------------------------------------------------------------------
def integrate(t0: float, 
              dt: float, 
              n: int, 
              x0: NDArray[np.float64], 
              func: Callable[[NDArray[np.float64], float, NDArray[np.float64]], 
                             NDArray[np.float64]], 
              p: NDArray[np.float64]) -> NDArray[np.float64]:
    # --------------------------------------------------------------------------
    # This function apply the steps of integration for the simulation
    # --------------------------------------------------------------------------
    # Description of function arguments:
    # t0:   initial time
    # x0:   vector containing the initial conditions (initial state variables)
    # n:    number of integration steps
    # func: system of first order ODEs to be solved
    # p:    array containing the values of constant parameters of the system
    # --------------------------------------------------------------------------
    # Returns:
    # The solution of the numerical integration in array format (time + state
    # variables)
    # --------------------------------------------------------------------------
    # Create array with the size of data full of zeros
    sol = np.zeros((n + 1, len(x0) + 1))
    # Assign the first sol row with initial conditions and time
    sol[0, 0] = t0
    sol[0, 1:] = x0
    # Integrate over the number of steps
    for i in range(n):
        # Solve for state variables
        sol[i + 1, 1:] = rk4(func, sol[i, 1:], dt, sol[i, 0], p)
        # Update time
        sol[i + 1, 0] = sol[i, 0] + dt        
        
    return sol

def define_period_of_excitation(angular_freqs: list) -> float:
    # --------------------------------------------------------------------------
    # This function determines the period of excitation of a periodic excitation
    # function based on its defined frequencies. 
    # --------------------------------------------------------------------------
    # Description of function arguments:
    # angular_freqs: list with all frequencies composing the excitation function
    # --------------------------------------------------------------------------
    # Returns
    # The period of excitation, T
    # --------------------------------------------------------------------------
    # Transform freq_list into array even if it is a scalar
    angular_freqs = np.atleast_1d(angular_freqs)
    # Check how many frequencies the signal is composed of
    if len(angular_freqs) == 1: # For single frequency
        # Define fundamental period of excitation
        T = 2*np.pi/angular_freqs[0]
    elif len(angular_freqs) > 1: # For multiple frequency components
        # Find the number of decimal places needed for dynamic scaling to 
        # frequencies to integer numbers
        decimal_places = -np.log10(angular_freqs.min())
        # Determine precision scaling to a power of 10 to fully eliminate 
        # decimals
        precision = 10 ** np.ceil(decimal_places)
        # Create scaled angular freqs array and convert to int
        angular_freqs_scaled = (angular_freqs * precision).astype(int)
        # Least common multiple (LCM) to find the common period of excitation
        lcm_freq = np.lcm.reduce(angular_freqs_scaled)
        T = 2*np.pi/(float(lcm_freq)/precision)
    else:
        print("Error: Invalid 'angular_freq_list' argument in 'define_period_of_excitation()' function.")
        exit()
        
    return T

def poincare_map_periodic_excitation(sol: NDArray[np.float64], 
                                     nP: int, 
                                     nDiv: int, 
                                     P_transient_ends: int, 
                                     section: int = 1) -> NDArray[np.float64]:
    # --------------------------------------------------------------------------
    # This function extracts the Poincaré map of the solution obtained by a 
    # fixed-step solver of a periodically excited dynamical system
    # --------------------------------------------------------------------------
    # Description of function arguments:
    # sol:              array with system solution (time + state variables)
    # nP:               number of excitation periods
    # nDiV:             number of steps per excitation period
    # P_transient_ends: excitation period when the transient state ends
    # section:          Poincaré section (value between 0 and nDiv)
    # --------------------------------------------------------------------------
    # Returns:
    # Array containing the points of the Poincaré map (time + state variables)
    # --------------------------------------------------------------------------
    # Determine the poincaré 
    poinc_map_size = nP - P_transient_ends
    poinc_result = np.zeros((poinc_map_size, sol.shape[1]))
    start_index_solution = P_transient_ends*nDiv
    # Start looping through the solution starting at index where transient ends    
    for j in range(poinc_map_size):
        for k in range(nDiv):
            # Determine Poincaré section (where in a single period the values
            # of the Poincaré map will be selected)
            if k == section:
                # Save the result as part of the Poincaré map
                poinc_result[j, :] = sol[start_index_solution + j*nDiv + k, :]
                
    return poinc_result

def poincare_map_autonomous(result, section = 1):
    # S is the location of section
    S = result[-1, -1] - section
    # Monitor if signal of S changes
    
    # 
    dx = -S
"""    
def bifurcation_diagram(Cpar_index, Cpar_i, Cpar_f, bifurc_steps, nP, nDiv, nP_transient_end, t0, init_cond, p, 
                        odesystem, bifurc_mode = "follow_attractor"):
    # This function constructs the bifurcation diagram of the ODE system 
    # --------------------------------------------------------------------------
    # Description of function arguments:
    # Cpar_index:       the index of the parameter, p, used as control parameter
    # Cpar_i:           initial control parameter
    # Cpar_f:           final control parameter
    # bifurc_steps:     number of bifurcation steps of the control parameter
    # nP:               number of external forcing periods
    # nDiv:             number of steps per forcing period
    # nP_transient_end: forcing period that the transient state ends
    # t0:               initial time
    # init_cont:        array with initial conditions of the state variables
    # p:                array with the contant parameters of the ODE system
    # odesystem:        1st order ODE system to be solved
    # bifurc_mode:      determine how to handle the initial conditions at
    #                   each step of the bifurcation ("follow attractor" to
    #                   use the last value of the integration result as the
    #                   initial condition for the next bifurcation step; 
    #                   "reset_IC" to reset the initial condition at every
    #                   bifurcation step) 
    # --------------------------------------------------------------------------
    # OBSERVATION:      p[0] must be the excitation frequency used in the 
    #                   1st order ODE system. It only works properly that way
    #                   because of the time step that depends on p[0].
    # --------------------------------------------------------------------------
    # Declare array that holds the value of all bifurcation steps
    Cpar = np.linspace(Cpar_i, Cpar_f, bifurc_steps, endpoint=True) 
    # Determine size of each poincare section to and use it to allocate an array for the bifurcation result 
    poincare_size = nP - nP_transient_end
    bifurc_result = np.zeros(((poincare_size)*len(Cpar), len(init_cond) + 1))
    # Monitor of the row of the bifurc_result matrix
    n = 0
    # Length of state variables vector
    len_x = len(init_cond)
    # Loop through the values of control parameters (in this case, Omega)    
    for cp in Cpar:
        print(f"Cpar = {cp}") 
        # Update control parameter
        p[Cpar_index] = cp
        # Reset time step (if Omega varies)
        dt = (2*np.pi/(p[0]*nDiv))            
        # Use proper initial conditions depending on the bifurc_mode
        if (bifurc_mode == "reset_IC") or (cp == Cpar[0]):
            x0 = init_cond
        elif bifurc_mode == "follow_attractor":
            x0 = int_result[nP*nDiv + 1, :]
        else:
            print("bifurc_mode must be 'reset_IC' or 'follow_attractor'.")
            sys.exit(1)
        # Solution of each time series and poincaré map
        int_result, poinc_result = integrate_and_poincare_map(nP, nDiv, t0, dt, x0, odesystem, p, nP_transient_end)
        # Save the values of the bifurcation diagram and the correspondent value of control parameter
        for i in range(poincare_size):
            # Save control parameter
            bifurc_result[n, 0] = cp
            # Save poincaré map results
            bifurc_result[n, 1:len_x+1] = poinc_result[i, 1:len_x+1]
            # go to the next row of the bifurc_result matrix
            n += 1        
        ##################################################
        # TO INSERT ADDITIONAL CALCULATIONS MUST BE HERE #    
        ##################################################
        
    # Return the row monitor and the result of the bifurcation
    return n, bifurc_result
"""
# ------------------------------------------------------------------------------
# Save results functions
# ------------------------------------------------------------------------------
def save_timeseries_data(sol: NDArray[np.float64], 
                         odesystem: Callable[[NDArray[np.float64], float, 
                                              NDArray[np.float64]], 
                                             NDArray[np.float64]], 
                         extension: str = "csv", 
                         decimal_precision: int = 10):
    # Determine the number of state variables
    nvars = sol.shape[1] - 1 # Subtract 1 to account for the time column
    # Open the file
    file = open(f"timeseries_{odesystem.__name__}.{extension}", "w")
    # Write the header
    header = " ".join(["time"] + [f"x[{i}]" for i in range(nvars)]) + "\n"
    file.write(header)     
    # Write the data
    for row in sol:
        line = " ".join([f"{value:.{decimal_precision}f}" for value in row]) + "\n"
        file.write(line)
    
    file.close()
    
def save_poincare_map_data(results: NDArray[np.float64], 
                           odesystem: Callable[[NDArray[np.float64], float, 
                                                NDArray[np.float64]], 
                                               NDArray[np.float64]], 
                           extension: str = "csv", 
                           decimal_precision: int = 10):
    # Determine the number of state variables
    nvars = results.shape[1] - 1 # Subtract 1 to account for the time column
    # Open the file
    file = open(f"poincare_map_{odesystem.__name__}.{extension}", "w")
    # Write the header
    header = " ".join(["time"] + [f"x[{i}]" for i in range(nvars)]) + "\n"
    file.write(header)     
    # Write the data
    for row in results:
        line = " ".join([f"{value:.{decimal_precision}f}" for value in row]) + "\n"
        file.write(line)
    
    file.close()
# ------------------------------------------------------------------------------
# Data visualization functions
# ------------------------------------------------------------------------------
def get_tspan_indexes(sol, t_span):
    if t_span is None:
        # Full interval
        ti_index = 0
        tf_index = -1
    else:
        # Find the closest indexes to the given time_interal
        ti_index = np.abs(sol[:, 0] - t_span[0]).argmin()
        tf_index = np.abs(sol[:, 0] - t_span[1]).argmin()
        
    return ti_index, tf_index

def visualize_timeseries(result, variables, t_span = None, color = "red", save = False, fig_ext = "pdf", 
                         fig_quality = 300, fig_name = "timeseries"):
    # Determine number of plots based on the number of variables
    nplots = len(variables)
    # Determine the time interval of the plot
    ti_index, tf_index = get_tspan_indexes(result, t_span)
    # Create figure and axes
    fig, axs = plt.subplots(figsize = (10, 2*nplots), nrows = nplots, 
                            ncols = 1, sharex = True, layout = "constrained")
    # Plot results    
    axs = np.atleast_1d(axs)
    variables = np.atleast_1d(variables)
    for i, n in zip(variables, range(nplots)):
        axs[n].plot(result[ti_index:tf_index, 0], result[ti_index:tf_index, i+1], color = color)
        axs[n].set_ylabel(f"x[{i}]")
    
    axs[-1].set_xlabel(f"t")
    # Save figure if requested
    if save == True:
        fig.savefig(f"{fig_name}.{fig_ext}", dpi = fig_quality)
    
    return fig, axs
        
def visualize_phase_subspaces(result, variable_sets, t_span = None, color = "blue", save = False, 
                              fig_ext = "pdf", fig_quality = 300, fig_name = "phase_subspaces"):    
    # Get number of plots based on the number of variable sets
    nplots = len(variable_sets)
    # Determine the time interval of the plot
    ti_index, tf_index = get_tspan_indexes(result, t_span)
    # Create figure and axes
    fig, axs = plt.subplots(figsize = (3*nplots, 2), nrows = 1, ncols = nplots,
                            layout = "constrained")
    # Ensure axs is an array
    axs = np.atleast_1d(axs)
    # Plot results 
    for i in range(len(variable_sets)):
        axs[i].plot(result[ti_index:tf_index, variable_sets[i][0] + 1], 
                    result[ti_index:tf_index, variable_sets[i][1] + 1], color = color)
        axs[i].set_ylabel(f"x[{variable_sets[i][1]}]")
        axs[i].set_xlabel(f"x[{variable_sets[i][0]}]")
        # Save figure if requested
    if save == True:
        fig.savefig(f"{fig_name}.{fig_ext}", dpi = fig_quality)
    
    return fig, axs

def visualize_poincare_map(result, variable_sets, color = "black", axs = None, save = False, 
                           fig_ext = "pdf", fig_quality = 300, fig_name = "poincare"):
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Get number of plots based on the number of variable sets
    nplots = len(variable_sets)
    # define a step variable fig
    fig = None
    if axs is None:
        # Create figure and axes
        fig, axs = plt.subplots(figsize = (3*nplots, 2), nrows = 1, 
                                ncols = nplots, layout = "constrained")
    # Ensure axs in an array
    axs = np.atleast_1d(axs)
    # Plot results
    for i in range(len(variable_sets)):
        axs[i].scatter(result[:, variable_sets[i][0] + 1], 
                       result[:, variable_sets[i][1] + 1], color = color,
                       zorder = 10)
        axs[i].set_ylabel(f"x[{variable_sets[i][1]}]")
        axs[i].set_xlabel(f"x[{variable_sets[i][0]}]")
    # Save figure
    if save == True:
        if fig == None: 
            fig = axs[0].get_figure()
        fig.savefig(f"{fig_name}.{fig_ext}", dpi = fig_quality)
    
    return fig, axs
# ------------------------------------------------------------------------------
# Other functions
# ------------------------------------------------------------------------------
def RMS(array):
    # Compute the square value of all cells in the array
    squared_values = (array**2)
    # Sum all the square_values
    s = sum(squared_values)
    # Compute the RMS value
    RMS_value = np.sqrt(s/len(array))
    # Return the RMS value
    return RMS_value
