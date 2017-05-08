import numpy as np
import pandas as pd

def envelope(m, A=0.2, m0=22., tau=1.):
    return A*np.exp(-(m0-m)/tau)

def make_random_y(m, **kwargs):
    n = len(m)
    env = envelope(m, **kwargs)
    y = np.random.randn(n)*env
    return y
    
def simulate_data(N, A=0.2, m0=22., tau=1., mag_lo=15, mag_hi=25):
    m = np.random.random(N)*(mag_hi - mag_lo) + mag_lo
    y = make_random_y(m, A=A, m0=m0, tau=tau)
    
    u = np.random.random(N)
    label = np.array(['A']*N)
    label[u < 0.5] = 'B'
    
    return pd.DataFrame({'x':m, 'y':y, 'label':label})