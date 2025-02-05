import numpy as np

def duffing(x, t, p):
    # Create array full of zeros
    f = np.zeros(len(x))
    # Define system of first order ODEs
    f[0] = x[1]
    f[1] = p[1]*np.sin(p[0]*t) - p[2]*x[1] - p[3]*x[0] - p[4]*(x[0]**3)
    
    return f
    
def bistable_EH(x, t, p): 
    # Create an array full of zeros
    f = np.zeros(len(x))       
    # Define system of first order ODEs
    f[0] = x[1]
    f[1] = p[1]*np.sin(p[0]*t) - 2.0*p[2]*x[1] - p[3]*x[0] - p[4]*(x[0]**3) + p[5]*x[2]
    f[2] = -p[6]*x[1] - p[7]*x[2]
    
    return f

def LR_circuit(x, t, p):
    # Create an array full of zeros
    f = np.zeros(len(x))       
    # Define system of first order ODEs
    f[0] = (1/p[2])*(p[0]*np.sin(p[1]*t) - p[3]*x[0])
    
    return f