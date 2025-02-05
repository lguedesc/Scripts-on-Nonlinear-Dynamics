import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_0_1(data):
    # Variable names are according to "Gottwald and Melbourne (2009) - On the Implementation of the 0-1 Test for Chaos"
    # Convert pandas dataframe to numpy array (exclude time column)
    phi = data.iloc[:, 1:].to_numpy() # Observation (dynamics of each state variable of the system)
    # Compute translation variables
    N = len(phi)                            # Get length of the data
    c = 1                                   # Define c (must be within 0 and pi)  
    p_c = np.zeros(N)       # Create an empty numpy array of same shape of phi for the horizontal translation variable
    q_c = np.zeros(N)       # Create an empty numpy array of same shape of phi for the vertical translation variable
    for n in range(N):
        j = np.arange(n) # Create a vector of j
        p_c[n] = np.sum(phi[j, 0]*np.cos(j*c))
        q_c[n] = np.sum(phi[j, 0]*np.sin(j*c))
        print(n)
    return p_c, q_c
    
# Input
filename = "timeseries_bistable_EH.csv"
save = False
show = True

# Read Data
df = pd.read_csv(filename, sep= " ")
# Define steady state data
N = len(df)
N_trans = int(0.75*N)
steady_state_df = df.iloc[N_trans:]
# Call 0-1 test
p, q = test_0_1(steady_state_df)
# Plot
plt.plot(p, q)
plt.show()