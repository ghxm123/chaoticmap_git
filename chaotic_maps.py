import numpy as np

def logistic_fun(t, state, r=2):
    x = state
    x_n = r * x * (1-x)
    return np.array([x_n])

def henon_fun(t, state, a=1.4, b=0.3):
    x, y = state
    x_n = 1 - a*x**2 + y
    y_n = b*x
    return np.array([x_n,y_n])
def chua_fun(t, state, alpha=15.395, beta=28):
    # https://stackoverflow.com/questions/61127919/chuas-circuit-using-python-and-graphing
    R = -1.143
    C_2 = -0.714
    x, y, z = state
    # electrical response of the nonlinear resistor
    f_x = C_2*x + 0.5*(R-C_2)*(abs(x+1)-abs(x-1))
    dx = alpha*(y-x-f_x)
    dy = x - y + z
    dz = -beta * y
    return np.array([dx,dy,dz])

def duffing_fun(t, state, alpha=-1, beta=1, delta=0.3, gamma=0.5, omega=1.2 ):
    # https://github.com/andyj1/chaotic-duffing-oscillator/blob/master/src/duffing.py
    x , v = state
    dx = v
    dv = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t) 
    return np.array([dx,dv])


def lorenz_fun(t, state, sigma=10, beta=2.67, rho=28):    
    x, y, z = state
    # print(sigma,beta,rho)
    dx =  sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx,dy,dz])                   
                    
def L96(t, x, N=5, F=8):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        
    return d


# t = 0
# tf = 100
# h = 0.01
def rossler_fun(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = - y - z
    dy =  x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])  

def vanderpol_fun(t, state, miu=1):
    x, y = state
    dx = y
    dy = miu*(1 - x**2)*y - x
    return np.array([dx, dy])                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                