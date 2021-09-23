import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def time_plot(funname, t, x):    
    fig = plt.figure()
    ax1 = fig.gca()
    ax1.plot(t, x) 
    aspect = 0.5*np.ptp(t)/np.ptp(x)
    ax1.set_aspect(aspect)
    ax1.set_title(f'{funname} Time series')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    return fig
    
def phase_plot(funname, X):
    fig = plt.figure()
    if X.shape[0] == 3:
      x, y, z = X
      ax2 = fig.gca(projection='3d')
      ax2.plot(x, y, z)
      ax2.set_title(f'{funname} Phase portrait')
    elif X.shape[0] == 2:
      x, y = X
      ax2 = fig.gca()
      ax2.plot(x, y)
      ax2.set_title(f'{funname} Phase portrait')
    return fig
    
def time_phase_plot_3d(funname, t, X, title):
    fig = plt.figure(dpi=100) #figsize=(WIDTH/DPI, HEIGHT/DPI)
    plt.title(title, y=0.9, fontsize=13)
    plt.axis('off')
    
    ax1 = fig.add_subplot(121)
    ax1.plot(t, X[0]) 
    aspect = 0.5*np.ptp(t)/np.ptp(X[0])
    ax1.set_aspect(aspect)
    ax1.set_title(f'{funname} Time series', y=-0.5)
    x, y, z = X
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x, y, z)
    ax2.set_title(f'{funname} Phase portrait', y=-0.2)
    return fig

def time_phase_plot_2d(funname, t, X, title):
    fig = plt.figure(figsize=(9,3)) #figsize=(WIDTH/DPI, HEIGHT/DPI)
    gs = plt.GridSpec(1, 3)
    plt.title(title, fontsize=13)
    plt.axis('off')

    ax1 = fig.add_subplot(gs[0,0:2])
    ax1.plot(t, X[0]) 
    ax1.set_title(f'{funname} Time series', y=-0.3)
    
    x, y = X
    ax2 = fig.add_subplot(gs[0,2])
    ax2.plot(y, x) 
    ax2.set_title(f'{funname} Phase portrait', y=-0.3)
    plt.tight_layout()
    return fig
# time_phase_plot(funname, t, x, y, z)

def phase_animation(funname, t, X):
    x, y, z = X
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