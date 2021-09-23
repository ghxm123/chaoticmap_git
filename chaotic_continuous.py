import numpy as np
from scipy.integrate import solve_ivp
from chaotic_plots import *
from chaotic_maps import *
# from chaotic_maps import logistic_fun, chua_fun, duffing_fun, henon_fun, lorenz_fun, L96, rossler_fun, vanderpol_fun
chaotic_dict = {
            "Chua's Circuit":chua_fun, 
            'Duffing':duffing_fun,
            'Lorenz':lorenz_fun, 
            'Lorenz 96':L96, 
            'Rossler':rossler_fun, 
            'Van der Pol':vanderpol_fun
            }

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
        
def ode_solver(fun, state0, ts, tf, h, method='solve_ivp', args=None):
    if args is not None:
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        
    if method == 'solve_ivp' :
        sol = solve_ivp(fun, (ts,tf), state0, t_eval=np.arange(ts,tf,h))
        return sol.t, sol.y
    elif method == 'rk4':
        sol = rk4(fun, state0, ts, tf, h)
        return sol
    



# if __name__ == 'main':
    
# funname = "Chua's Circuit"
# method = 'solve_ivp'
# state0 = [0.1, 0, 0]
# args = (15.395, 28) # alpha=15.395, beta=28
# soln = ode_solver(chaotic_dict[funname], state0, ts=80, tf=100, h=0.01, method=method)    
# t, X = soln[0], soln[1]
# time_plot(funname, t, X[0])
# phase_plot(funname,X)
# title = f"{funname}: alpha={args[0]},beta={args[1]}"
# time_phase_plot_3d(funname, t, X, title)

# funname = 'Lorenz'
# method = 'solve_ivp'
# state0 = [0, 1, 1.05]
# args = (10, 2.67, 28) # sigma=10,beta=8/3,rho=28
# soln = ode_solver(chaotic_dict[funname], state0, ts=60, tf=100, h=0.01, method=method)   
# skip = 1000 
# t, X = soln[0][skip:], soln[1][:,skip:]
# # time_plot(funname,t,X[0])
# # phase_plot(funname,X)
# title = f"{funname}: alpha={args[0]},beta={args[1]},rho={args[2]}"
# f = time_phase_plot_3d(funname, t, X, title)
# # f.savefig(f'{funname}.png')
# # phase_animation(funname, t, X)

# funname = "Duffing"
# method = 'solve_ivp'
# state0 = [0, 0]
# args = (-1, 1, 0.3, 0.5, 1.2) # alpha=-1, beta=1, delta=0.3, gamma=0.5, omega=1.2
# soln = ode_solver(chaotic_dict[funname], state0, ts=0, tf=100, h=0.01, method=method)    
# t, X = soln[0], soln[1]
# time_plot(funname, t, X[0])
# phase_plot(funname, X)
# title = f"{funname}: alpha={args[0]},beta={args[1]},delta={args[2]}"
# time_phase_plot_2d(funname, t, X, title)

# funname = "Lorenz 96"
# method = 'rk4'
# state0 = [1.01, 1, 1, 1, 1]
# args = (5, 8) # N=5, F=8
# soln = ode_solver(chaotic_dict[funname], state0, ts=80, tf=100, h=0.01, method=method)    
# t, X = soln[0], soln[1][:3]
# time_plot(funname, t, X[0])
# phase_plot(funname, X)
# title = f"{funname}: N={args[0]},F={args[1]}"
# time_phase_plot_3d(funname, t, X, title)

# funname = "Rossler"
# method = 'solve_ivp'
# state0 = [0.1, 0.1, 0.1]
# args = (0.2, 0.2, 5.7) # a=0.2, b=0.2, c=5.7
# soln = ode_solver(chaotic_dict[funname], state0, ts=50, tf=100, h=0.01, method=method)    
# t, X = soln[0], soln[1]
# time_plot(funname, t, X[0])
# phase_plot(funname, X)
# title = f"{funname}: a={args[0]},b={args[1]},c={args[2]}"
# time_phase_plot_3d(funname, t, X, title)

# funname = "Van der Pol"
# method = 'solve_ivp'
# state0 = [1, 0]
# args = (1,) # miu=1
# soln = ode_solver(chaotic_dict[funname], state0, ts=50, tf=100, h=0.01, method=method)    
# t, X = soln[0], soln[1]
# time_plot(funname, t, X[0])
# phase_plot(funname, X)
# title = f"{funname}: miu={args[0]}"
# time_phase_plot_2d(funname, t, X, title)











