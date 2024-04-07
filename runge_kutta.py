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
