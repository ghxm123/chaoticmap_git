import numpy as np
import matplotlib.pyplot as plt

# def ChuasCircuit_fun():
#     dx = a*(y-x-f(x))
    

# def Duffing_fun():

def RK2(f,u,times,subdiv = 1):
 uout = np.zeros((len(times),)+u.shape)
 uout[0] = u;
 for k in range(len(times)-1):
     t = times[k]
     h = (times[k+1]-times[k])/subdiv
     for j in range(subdiv):
        k1 = f(u,t)*h
        k2 = f(u+0.5*k1, t+0.5*h)*h
        u, t = u+k2, t+h
     uout[k+1]=u
 return uout

def plotphase(A,B,C,D):
     def derivs(u,t): y,z = u; return np.array([ z, -A*y**3 + B*y - C*z + D*np.sin(t) ])
     N=60
     u0 = np.array([0.0, 0.0])
     t  = np.arange(0,300,2*np.pi/N); 
     u  = RK2(derivs, u0, t, subdiv = 10)
     plt.plot(u[:-2*N,0],u[:-2*N,1],'.--y', u[-2*N:,0],u[-2*N:,1], '.-b', lw=0.5, ms=2);
     plt.plot(u[::N,0],u[::N,1],'rs', ms=4); plt.grid(); plt.show()

plotphase(0.25, 1.0, 0.1, 1.0)