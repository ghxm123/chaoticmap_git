import numpy as np
import matplotlib.pyplot as plt

def duffing_fun(state, t):
    # https://github.com/andyj1/chaotic-duffing-oscillator/blob/master/src/duffing.py
    # parameters
    delta = 0.3 #damping constant
    alpha = -1 #linear stiffness
    beta = 1 #second damping constant
    gamma = 0.5 #amplitude
    omega = 1.2 #frequency 
    
    x , v = state
    dx = v
    dv = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t) 
    return np.array([dx,dv])
    