import numpy as np

def bistable_EH(x, t, p):
    # This function defines the ODE system (the bistable energy harvester)
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # x:    vector containing the current value of the state variables
    # t:    current time
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------
    # Parameters:
    # p[0]: Omega  (excitation frequency)
    # p[1]: gamma  (excitation amplitude)
    # p[2]: zeta   (mechanical dissipation)
    # p[3]: alpha  (first restitution coefficient)
    # p[4]: beta   (second restitution coefficient)
    # p[5]: chi    (electromechanical coupling in the mechanical ODE)
    # p[6]: kappa  (electromechanical coupling in the electrical ODE)
    # p[7]: varphi (electrical conductance -> 1/resistance)
    # p[8]: ksi_1  (first nonlinear coupling)
    # p[9]: ksi_2  (second nonlinear coupling)
    # ------------------------------------------------------------------------
    # State Variables:
    # x[0]: displacement
    # x[1]: velocity
    # x[2]: voltage
    # ------------------------------------------------------------------------    
    # Create an array full of zeros
    f = np.zeros(len(x))       
    # Define system of first order ODEs
    f[0] = x[1]
    f[1] = p[1]*np.sin(p[0]*t) - 2.0*p[2]*x[1] - p[3]*x[0] - p[4]*(x[0]**3) + p[5]*(1 + p[8]*x[0] + p[9]*(x[0]**2))*x[2]
    f[2] = -p[6]*(1 + p[8]*x[0] + p[9]*(x[0]**2))*x[1] - p[7]*x[2]
    
    return f