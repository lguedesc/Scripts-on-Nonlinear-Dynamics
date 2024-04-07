import numpy as np

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

def poincare_map(result, integration_result, current_P, current_Div, current_step, pp, nP_trans_end):
    # Poincaré map
    if (current_P >= nP_trans_end):  
        if (current_Div == 1):   # Poincaré Section             
            result[pp, :] = integration_result[current_step, :]
            pp += 1
            
    return pp, result

def integrate_and_poincare_map(nP, nDiv, t0, dt, x0, func, p, nP_transient_end):
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