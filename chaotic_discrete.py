import numpy as np

import matplotlib.pyplot as plt



def logistic_fun(state, r=2):
    x = state
    x_n = r * x * (1-x)
    return np.array(x_n)


def henon_fun(state, a=1.4, b=0.3):
    x, y = state
    x_n = 1 - a*x**2 + y
    y_n = b*x
    return np.array([x_n,y_n])


def get_points(funname, state0, ns, nf, *args):
    x = state0
    X = [x]
    for i in range(nf):
        xn = chaotic_dict[funname](x, *args)
        X.append(xn)
        x = xn
    return X[ns:]    

chaotic_dict = {'Logistic':logistic_fun, 'Henon':henon_fun}

# funname = 'Logistic'
# state0 = 0.5
# ns, nf = 0, 30
# args = [r for r in np.arange(0,4) ]
# for r in args:
#     p = get_points(funname, state0, ns, nf, r)
#     n = np.arange(ns, nf+1)
#     plt.plot(n, p)
# plt.legend([str(r) for r in args], title='rate', loc='right')
# plt.title(f'{funname} Time series')

# funname = 'Henon'
# state0 = [0, 0]
# ns, nf = 0, 1000
# args = [(1.4,0.3), (1.3,0.2)]
# for a,b in args:
#     p = get_points(funname, state0, ns, nf, a, b)
#     x, y = np.vstack(p).T
#     plt.scatter(x, y, s = 1, alpha=0.5)
# plt.title(f'{funname} Phase portrait')

