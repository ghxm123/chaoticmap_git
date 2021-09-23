import numpy as np
from scipy.integrate import solve_ivp,odeint
import matplotlib.pyplot as plt
import matplotlib.animation as ani
# from mpl_toolkits.mplot3d import Axes3D
# https://scipython.com/blog/the-lorenz-attra

WIDTH, HEIGHT, DPI = 1000, 750, 100

# Lorenz paramters and initial conditions.

u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points.
tmax, n = 50, 5000
t = np.linspace(0, tmax, n)

def lorenz_fun(t, state):
    sigma = 10
    beta = 8/3
    rho = 28
    """The Lorenz equations."""
    x, y, z = state
    dx =  sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx,dy,dz])

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz_fun, (0, tmax), (u0, v0, w0),t_eval=t)
x, y, z = soln.y

# sol = odeint(Lorenz_fun,  (u0, v0, w0), t, tfirst=True)
# x, y, z = sol[:,0],sol[:,1],sol[:,2]

# Plot the Lorenz attractor using a Matplotlib 3D projection.
fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI)) #figsize=(WIDTH/DPI, HEIGHT/DPI)
ax = fig.gca(projection='3d')
ax.set_xlim((-30,30))
ax.set_ylim((-30,30))
ax.set_zlim((0,50))

# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# ax.plot(x, y, z)

def an(i):
    ax.plot(x[:i], y[:i], z[:i], color='g', alpha=0.5, linewidth=0.5)
animator = ani.FuncAnimation(fig, an,range(1,len(x),100), interval = 2)
animator.save('C:/Users/x/Desktop/m.gif')
















