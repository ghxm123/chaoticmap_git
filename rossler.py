import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint
# 数值方法
# https://medium.com/codex/python-and-physics-lorenz-and-rossler-systems-65735791f5a2
a = 0.2
b = 0.2
c = 5.7
t = 0
tf = 100
h = 0.01
def Rossler_fun(t, state):
    x, y, z = state
    dx = - y - z
    dy =  x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

soln = solve_ivp(Rossler_fun, (0, tf), (0.1, 0.1, 0.1),t_eval=np.arange(0,tf,h))
x, y, z = soln.y

fig = plt.figure() #figsize=(WIDTH/DPI, HEIGHT/DPI)
ax = fig.gca(projection='3d')
ax.plot(x, y, z)


# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# ax.plot(x, y, z)
# time = np.array([])
# x = np.array([])
# y = np.array([])
# z = np.array([])
# r = np.array([0.1, 0.1, 0.1])
# while (t <= tf ):
    
#         time = np.append(time, t)
#         z = np.append(z, r[2])
#         y = np.append(y, r[1])
#         x = np.append(x, r[0])
        
#         k1 = h*derivative(r,t)
#         k2 = h*derivative(r+k1/2,t+h/2)
#         k3 = h*derivative(r+k2/2,t+h/2)
#         k4 = h*derivative(r+k3,t+h)
#         r += (k1+2*k2+2*k3+k4)/6
        
#         t = t + h
# fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15, 5))
# ax1.plot(x, y)
# ax1.set_title("X & Y")
# ax2.plot(x, z)
# ax2.set_title("X & Z")
# ax3.plot(y, z)
# ax3.set_title("Y & Z")
# plt.show()