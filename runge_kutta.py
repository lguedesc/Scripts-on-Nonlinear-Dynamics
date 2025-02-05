import numpy as np

# Fonte 0 (generalização por tabela): https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods
# Fonte 1 (implementação scipy): https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/rk.py 
# Fonte 2 (butcher tableau): https://en.wikipedia.org/wiki/Runge–Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
# Fonte 3 (DP tablau method): https://en.wikipedia.org/wiki/Dormand–Prince_method
# Fonte 4 (Octave implementation): https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_45_dorpri.m
# Fonte 5 (Algoritmo): https://numerary.readthedocs.io/en/latest/dormand-prince-method.html    
    
def rk4_old(func, x, dt, t, p):
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

def rk_fixed_step(fun, x, dt, t, p, A, B, C):
    # Number of stages
    stages = len(B)
    # Create array for slopes (number of slopes/stages, dimension of the system)
    k = np.zeros((stages, len(x)))
    # Compute slopes
    for i in range(stages):
        ti = t + C[i]*dt    # time increment
        xi = x + dt * np.dot(A[i, :i], k[:i]) # Explicit RK: only lower triangular part is used
        k[i] = fun(xi, ti, p)
    # Return average of slopes
    return x + dt*np.dot(B, k)
    
def rk4_tab():
    # Time fraction array
    C = np.array([0.0, 1.0/2.0, 1.0/2.0, 1.0])
    # Coefficients for intermediate stages
    A = np.array([[    0.0,     0.0, 0.0, 0.0],
                  [1.0/2.0,     0.0, 0.0, 0.0],
                  [    0.0, 1.0/2.0, 0.0, 0.0],
                  [    0.0,     0.0, 1.0, 0.0]])
    # Define weigths for the fourth order solution
    B = np.array([1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0])
    
    return A, B, C

def integrate_new(t0, dt, n, x0, func, p, method = "RK4"):
    # Define method
    if method == "RK4":
        A, B, C = rk4_tab()
    else:
        raise ValueError(f"Unsuported Method: {method}")
    # Create array with the size of data full of zeros
    result = np.zeros((n + 1, len(x0) + 1))
    # Assign the first row of result with the initial conditions and initial time
    result[0, 0] = t0
    result[0, 1:] = x0
    # Integrate over the number of steps
    for i in range(n):
        # Solve for state variables
        result[i + 1, 1:] = rk_fixed_step(func, result[i, 1:], dt, result[i, 0], p, A, B, C)
        # Update time
        result[i + 1, 0] = result[i, 0] + dt        
        
    return result

def rkdp54(fun, x, dt, t, p):
    # Time fraction array
    C = np.array([0.0, 1.0/5.0, 3.0/1.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0])
    # Coefficients for intermediate stages
    A = np.array([
        [           0.0,             0.0,            0.0,          0.0,             0.0,       0.0],
        [       1.0/5.0,             0.0,            0.0,          0.0,             0.0,       0.0],
        [      3.0/40.0,        9.0/40.0,            0.0,          0.0,             0.0,       0.0],
        [     44.0/45.0,      -56.0/15.0,       32.0/9.0,          0.0,             0.0,       0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,             0.0,       0.0],
        [ 9017.0/3168.0,     -355.0/33.0, 46732.0/5247.0,   49.0/176.0, -5103.0/18656.0,       0.0],        
        [    35.0/384.0,             0.0,   500.0/1113.0,  125.0/192.0,  -2187.0/6784.0, 11.0/84.0]
    ])
    # The weights for the fifth order solution, in this special case, are presented in the last row of A.
    # Define weigths for the fourth order solution used to estimate errors
    B = np.array(
        [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]
    )
    # Compute slopes
    slopes = 7
    k = np.zeros(slopes,1)
    
    k[0] = fun(x, t, p)
#    for i in range(1, slopes):
#        k[i] = 
    