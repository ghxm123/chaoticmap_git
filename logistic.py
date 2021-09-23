import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def logistic_fun(t, state, r=2):
    x = state
    x_n = r * x * (1-x)
    return x_n
sol = solve_ivp(logistic_fun, (0,20), [8])