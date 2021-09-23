import numpy as np
import matplotlib.pyplot as plt

def henon_fun(state, a, b):
    x, y = state
    x_n = 1 - a*x**2 + y
    y_n = b*x
    return np.array([x_n,y_n])
    