import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as ani
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# https://scipython.com/blog/the-lorenz-attra


def lorenz_fun(t, state, sigma=10, beta=8/3, rho=28):    
    x, y, z = state
    # print(sigma,beta,rho)
    dx =  sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx,dy,dz])

def rk4(fun, state0, ts, tf, h):
    t = ts
    y = state0
    T = [t]
    Y = [y]
    while t < tf:
        K1 = fun(t, y)
        K2 = fun(t+h/2, y+h*K1/2)
        K3 = fun(t+h/2, y+h*K2/2)
        K4 = fun(t+h, y+h*K3)
        t = t+h
        y = y + h*(K1+2*K2+2*K3+K4)/6  
        T.append(t)
        Y.append(y)
    return np.hstack(T), np.vstack(Y).T
        
    
# Integrate the Lorenz equations.
def ode_solver(fun, state0, ts, tf, h, method='solve_ivp', args=None):
    if args is not None:
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        
    if method == 'solve_ivp' :
        sol = solve_ivp(fun, (ts,tf), state0, t_eval=np.arange(ts,tf,h))
        return sol.t, sol.y
    elif method == 'rk4':
        sol = rk4(fun, state0, ts, tf, h)
        return sol

# sol = ode_solver(lorenz_fun, (0, 1, 1.05), ts=80, tf=100, h=0.01, method='solve_ivp')    
# t, (x, y, z) = sol[0], sol[1]
# funname = 'Lorenz'

# sol_rk = ode_solver(lorenz_fun, (0, 1, 1.05), ts=50, tf=100, h=0.1, method='rk4') 
# t2, (x2, y2, z2) = sol_rk[0], sol_rk[1] 

def time_plot(funname, t, y):
    plt.plot(t, y)
    plt.title(f'{funname} Time series')
    plt.xlabel('t')
    plt.ylabel('x')
    # fig = plt.figure(dpi=200)
# time_plot(funname,t,y)

def phase_plot(funname, x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)
    ax.set_title(f'{funname} Phase portrait')
# phase_plot(funname,x,y,z)

def time_phase_plot(funname, t, x, y, z):
    fig = plt.figure(dpi=200) #figsize=(WIDTH/DPI, HEIGHT/DPI)
    # gs = plt.GridSpec(1, 2)
    plt.title('Lorenz System: sigma=10,beta=8/3,rho=28',y=0.9,fontsize=13)
    # plt.title('Lorenz System: sigma=10,beta=8/3,rho=28',y =-0.1,pad=20,fontsize=12)
    
    plt.axis('off')
    # fig.subplots_adjust(left=0, right=1, bottom=1, top=2)
    # ax1 = fig.gca(projection='3d')
        
    # ax1 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(121)
    ax1.plot(t, x) 
    ax1.set_aspect(0.3)
    # ax.set_aspect((max(t)-min(t))/(max(x)-min(x)))
    ax1.set_title(f'{funname} Time series', y=-0.3)
    
    # ax2 = fig.add_subplot(gs[0,1], projection='3d')
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot(x, y, z) 
    ax2.set_title(f'{funname} Phase portrait', y=-0.2)

# time_phase_plot(funname, t, x, y, z)

def phase_animation(funname, t, x, y, z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    ax.set_xlim((-30,30))
    ax.set_ylim((-30,30))
    ax.set_zlim((0,50))
    ax.set_title(f'{funname} Phase animation')
    
    def an(i):
        ax.plot(x[:i], y[:i], z[:i], color='g', alpha=0.5, linewidth=0.5)
    animator = ani.FuncAnimation(fig, an,range(1,len(x),100), interval = 2)
    animator.save('C:/Users/x/Desktop/m.gif')

# phase_animation(funname, t, x, y, z)

if __name__=='main':
    funname = 'Lorenz'
    state0 = (0, 1, 1.05)
    args = None # sigma=10,beta=8/3,rho=28
    sol = ode_solver(lorenz_fun, state0, ts=80, tf=100, h=0.01, method='solve_ivp')    
    t, (x, y, z) = sol[0], sol[1]  
    # time_plot(funname,t,y)
    # phase_plot(funname,x,y,z)
    # time_phase_plot(funname, t, x, y, z)
    # phase_animation(funname, t, x, y, z)














