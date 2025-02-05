import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdb
def regressionChaos(data,a,b):
        return a+(b*data)
    
def chaos01Test(data,c=np.pi/2,ncut=152,fs = 1526):
    
    if fs:
        ts = np.arange(0, len(data)/fs, 1/fs)
    else:
        ts = np.arange(0,len(data))
    #ts = np.arange(0,len(data))
    N = len(data)
    p = np.zeros((N,))
    q = np.zeros((N,))
    #A little silly I realize, 
    p[0] = data[0]*np.cos(c)
    q[0] = data[0]*np.sin(c)
    M = np.zeros((ncut,))
    
    for ck in range(N-1):
        p[ck+1] = p[ck] + (data[ck]*np.cos(c*ts[ck]))
        q[ck+1] = q[ck] + (data[ck]*np.sin(c*ts[ck]))
    #Get mean-squared displacement
    
    for jk in range(ncut):
        curSum = []
        for bc in range(N):
            if bc+jk >= len(data-1):
                break
            else:
                
                curVal = np.power((p[bc+jk]-p[bc]),2) + np.power((q[bc+jk]-q[bc]),2)
                curSum.append(curVal)
        
        M[jk] = np.mean(curSum)
    #Get oscillation term
    
    expectedValData = np.mean(data)
    Vosc = np.zeros((ncut,))
    for tk in range(ncut):
        Vosc[tk] = np.power(expectedValData,2)*((1-np.cos(tk/fs*c))/(1-np.cos(c)))
    D = M - Vosc
    a = 1.1
    
    Dtilde = D-(a*np.min(D))
    Dtilde = Dtilde[1:len(Dtilde)]
    [Kc, pcov] = curve_fit(regressionChaos,np.log(np.arange(1,ncut)),np.log(Dtilde+0.0001))
    
    return Kc[1]

def estimateChaos(data,pltFlag = 0):
    carray = np.arange(0.1,2*np.pi,0.1)
    Kest = np.zeros(len(carray),)
    for ck in range(len(carray)):
        Kest[ck] = chaos01Test(data,c=carray[ck])
    if pltFlag == 1:
        plt.plot(carray,Kest)
        plt.show()
    return np.median(Kest)

def logisticMap(mu,x0,n):
    x = np.zeros((n,))
    x[0] = x0
    for ck in np.arange(1,n-1):
        try:
            x[ck] = mu*x[ck-1]*(1-x[ck-1])
        except:
            pdb.set_trace()
    return x


N = 50000
mu_range = np.arange(2.5,4,0.001)
regular = logisticMap(2.8,0.5,N)
chaos = logisticMap(3.7,0.5,N)

Knorm = estimateChaos(regular,pltFlag=1)
Kchaos = estimateChaos(chaos,pltFlag=1)
pdb.set_trace()