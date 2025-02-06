"""
--------------------------------------------------------------------------------------------------------------
All the dynamical system definitions are provided in this file.

Author: Luã G Costa [https://github.com/lguedesc]
Created: 7 Apr 2024
Last Update: 6 Feb 2025 
--------------------------------------------------------------------------------------------------------------
"""

import numpy as np

def lorenz(x, t, p):
    # This function defines the ODE system (classocal Lorenz atmospheric model)
    # Type: autonomous
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # x:    vector containing the current value of the state variables
    # t:    current time
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------
    # Parameters:
    # p[0]: sigma (contant proportional to Prandtl number)
    # p[1]: rho   (constant proportional to Rayleigh number)
    # p[2]: beta  (contant proportional to layer dimensions)
    # ------------------------------------------------------------------------
    # State Variables:
    # x[0]: rate of convection
    # x[1]: horizontal temperature variation
    # x[2]: vertical temperature variation
    # ------------------------------------------------------------------------
    # Create an array full of zeros
    f = np.zeros(len(x))
    # Define system of 1st order ODEs
    f[0] = p[0]*(x[1] - x[0])
    f[1] = x[0]*(p[1] - x[2]) - x[1]
    f[2] = x[0]*x[1] - p[2]*x[2]
    
    return f

def bistable_EH(x, t, p):
    # This function defines the ODE system (the bistable energy harvester)
    # Type: non-autonomous
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # x:    vector containing the current value of the state variables
    # t:    current time
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------
    # Parameters:
    # p[0]: Omega      (excitation frequency)
    # p[1]: gamma      (excitation amplitude)
    # p[2]: zeta       (mechanical dissipation)
    # p[3]: alpha      (first restitution coefficient)
    # p[4]: beta       (second restitution coefficient)
    # p[5]: chi        (electromechanical coupling in the mechanical ODE)
    # p[6]: kappa      (electromechanical coupling in the electrical ODE)
    # p[7]: varphi     (electrical conductance -> 1/resistance)
    # p[8]: varepsilon (first nonlinear coupling)
    # p[9]: mu         (second nonlinear coupling)
    # ------------------------------------------------------------------------
    # State Variables:
    # x[0]: displacement
    # x[1]: velocity
    # x[2]: voltage
    # ------------------------------------------------------------------------    
    # Create an array full of zeros
    f = np.zeros(len(x))
    # Define base excitation
    xb = -p[1]*np.sin(p[0]*t)     
    # Define system of first order ODEs
    f[0] = x[1]
    f[1] = - xb - 2.0*p[2]*x[1] - p[3]*x[0] - p[4]*(x[0]**3) + p[5]*(1 + p[8]*np.abs(x[0]) + p[9]*(x[0]**2))*x[2]
    f[2] = -p[6]*(1 + p[8]*np.abs(x[0]) + p[9]*(x[0]**2))*x[1] - p[7]*x[2]
    
    return f

def bistable_EH_multi_freq(x, t, p):
    # This function defines the ODE system (the bistable energy harvester)
    # Type: non-autonomous
    # ------------------------------------------------------------------------
    # Description of function arguments:
    # x:    vector containing the current value of the state variables
    # t:    current time
    # p:    array containing the values of constant parameters of the system
    # ------------------------------------------------------------------------
    # Parameters:
    # p[0]: Omega      (excitation frequency)
    # p[1]: gamma      (excitation amplitude)
    # p[2]: zeta       (mechanical dissipation)
    # p[3]: alpha      (first restitution coefficient)
    # p[4]: beta       (second restitution coefficient)
    # p[5]: chi        (electromechanical coupling in the mechanical ODE)
    # p[6]: kappa      (electromechanical coupling in the electrical ODE)
    # p[7]: varphi     (electrical conductance -> 1/resistance)
    # p[8]: varepsilon (first nonlinear coupling)
    # p[9]: mu         (second nonlinear coupling)
    # p[10]: Omega_2   
    # ------------------------------------------------------------------------
    # State Variables:
    # x[0]: displacement
    # x[1]: velocity
    # x[2]: voltage
    # ------------------------------------------------------------------------    
    # Create an array full of zeros
    f = np.zeros(len(x))
    # Define base excitation
    xb = -p[1]*np.sin(p[0]*t) - p[1]*np.cos(p[10] * t)     
    # Define system of first order ODEs
    f[0] = x[1]
    f[1] = - xb - 2.0*p[2]*x[1] - p[3]*x[0] - p[4]*(x[0]**3) + p[5]*(1 + p[8]*np.abs(x[0]) + p[9]*(x[0]**2))*x[2]
    f[2] = -p[6]*(1 + p[8]*np.abs(x[0]) + p[9]*(x[0]**2))*x[1] - p[7]*x[2]
    
    return f