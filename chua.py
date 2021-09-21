import numpy as np
from scipy.integrate import odeint,solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Chua_fun(t,state):
    # https://stackoverflow.com/questions/61127919/chuas-circuit-using-python-and-graphing
    # parameters
    alpha = 15.395
    beta = 28
    R = -1.143
    C_2 = -0.714
    x, y, z = state
    # electrical response of the nonlinear resistor
    f_x = C_2*x + 0.5*(R-C_2)*(abs(x+1)-abs(x-1))
    dx = alpha*(y-x-f_x)
    dy = x - y + z
    dz = -beta * y
    return np.array([dx,dy,dz], float)




# time discretization
t_0 = 0
dt = 1e-3
t_final = 300
t = np.arange(t_0, t_final, dt)

# initial conditions
u0 = [0.1,0,0]
# integrate ode system
sol = odeint(Chua_fun, u0, t, tfirst=True)
soln = solve_ivp(Chua_fun, [t_0, t_final], u0, dense_output=True)
x, y, z = soln.sol(t)
x2, y2, z2 = soln.y

# 3d-plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.plot(sol[:,0],sol[:,1],sol[:,2])
# ax.plot(x, y, z)
# ax.plot(x2, y2, z2)