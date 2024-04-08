import numpy as np
import sys

def rk4(func, x, dt, t, p):
    # This function applies the fourth order Runge-Kutta method.
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # func: ODE system that we want to solve
    # x:    vector containing the current value of the state variables
    # dt:   step size
    # t:    current time
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------ 
    # Compute slopes
    k1 = func(x, t, p)
    k2 = func(x + (k1*dt/2.0), t + (dt/2.0), p)
    k3 = func(x + (k2*dt/2.0), t + (dt/2.0), p)
    k4 = func(x + k3*dt, t + dt, p)
    # Return average of slopes    
    return x + dt*(k1 + 2.0*(k2 + k3) + k4)/6.0

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
    result = np.zeros((n + 1, len(x0) + 1))
    # Assign the first row of result with the initial conditions and initial time
    result[0, 0] = t0
    result[0, 1:] = x0
    # Integrate over the number of steps
    for i in range(n):
        # Solve for state variables
        result[i + 1, 1:] = rk4(func, result[i, 1:], dt, result[i, 0], p)
        # Update time
        result[i + 1, 0] = result[i, 0] + dt        
        
    return result

def poincare_map(result, integration_result, current_P, current_Div, current_step, pp, nP_trans_end, section = 1):
    # This function extracts the Poincaré map while the solution of the
    # simulation is being performed
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # result:             matrix that contains the result of the poincaré map
    # integration_result: matrix that contains the result of the integration
    # current_P:          current forcing period of analysis
    # current_Div:        current division of the forcing period
    # current_step:       current_step in the analysis
    # pp:                 monitor of the position in the result vector
    # nP_trans_end:       forcing period that the transient state ends
    # section:            Poincaré section (it must be a value between 0 and nDiv)
    # ------------------------------------------------------------------------
    # If the current period is greater or equal the nP_trans_end   
    if (current_P >= nP_trans_end):  
        # Determine poincaré Section (where in a single period the values of 
        # the poincaré map will be selected)
        if (current_Div == section):
            # Put the result of the integration in the section into the result
            # of the Poincaré Map              
            result[pp, :] = integration_result[current_step, :]
            # Advance one position in the Poincaré result matrix
            pp += 1
    
    # Returns the index of the poincaré result matrix, and the result matrix            
    return pp, result

def integrate_and_poincare_map(nP, nDiv, t0, dt, x0, func, p, nP_transient_end):
    # This function solves the system of ODEs and extracts the Poincaré map while 
    # the simulation is being performed
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # nP:                 number of forcing periods
    # nDiv:               number of divisions per forcing period
    # t0:                 initial time
    # dt:                 time step
    # x0:                 array with initial conditions of the state variables
    # func:               1st order ODE system to be solved
    # p:                  array with the contant parameters of the ODE system
    # nP_trans_end:       forcing period that the transient state ends
    # ------------------------------------------------------------------------
    # Allocate arrays to store result of integration and poincaré map
    int_result = np.zeros((nP*nDiv + 1, len(x0) + 1))
    poinc_result = np.zeros((nP - nP_transient_end, len(x0) + 1))
    # Assign the first row of int_result with the initial conditions and initial time
    int_result[0, 0] = t0
    int_result[0, 1:] = x0
    # Integrate over the number of steps, but splitting between number of periods (nP)
    # and divisions per period (nDiv)
    i = 0
    pp = 0;
    for j in range(nP):
        for k in range(nDiv):
            # Solve for state variables
            int_result[i + 1, 1:] = rk4(func, int_result[i, 1:], dt, int_result[i, 0], p)
            pp, poinc_result = poincare_map(poinc_result, int_result, j, k, i, pp, nP_transient_end)
            # Update time
            int_result[i + 1, 0] = int_result[i, 0] + dt
            # Update counter
            i += 1

    return int_result, poinc_result

def bifurcation_diagram(Cpar_index, Cpar_i, Cpar_f, bifurc_steps, nP, nDiv, nP_transient_end, t0, init_cond, p, odesystem, bifurc_mode = "follow_attractor"):
    # This function constructs the bifurcation diagram of the ODE system 
    # ------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------
    # OBSERVATION:      p[0] must be the excitation frequency used in the 
    #                   1st order ODE system. It only works properly that way
    #                   because of the time step that depends on p[0].
    # ------------------------------------------------------------------------
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
        
def RMS(array):
    # Compute the square value of all cells in the array
    squared_values = (array**2)
    # Sum all the square_values
    s = sum(squared_values)
    # Compute the RMS value
    RMS_value = np.sqrt(s/len(array))
    # Return the RMS value
    return RMS_value