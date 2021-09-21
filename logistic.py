import numpy as np
import matplotlib.pyplot as plt

def Logistic_fun(state, r):
    x = state
    x_n = r * x * (1-x)
    return x_n
    