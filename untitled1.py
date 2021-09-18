# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:21:58 2020

@author: feng
"""

###洛伦兹系统分岔图、相位图、时间历程图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer  # 需要导入的模块


#洛伦兹方程
def Lorenz_fun(x0, rho):
    sigma, bata = 10, 8/3.
    x, y, z = x0[0], x0[1], x0[2]
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - bata * z
    
    x1 = np.array([dx,dy,dz]) #如果这里是列表的话会造成下面循环出现问题：can't multiply sequence by non-int of type 'float'
    return x1

#龙格库塔方程
def Rk4(x0, h, rho):    #洛伦兹方程组是自治的
    K1 = Lorenz_fun(x0, rho)
    K2 = Lorenz_fun(x0 + h * K1/2., rho)
    K3 = Lorenz_fun(x0 + h * K2/2., rho)
    K4 = Lorenz_fun(x0 + h * K3, rho)
    x_n = x0 + h * (K1 + 2.* K2 + 2.*K3 + K4)/6.   
    return x_n


 #洛伦兹系统的分岔图 （截面法）
def Bifurcation_diagram(x0, h):
    X, R = [], []            #存储满足条件的x值和参数值
    eps = 0.01
    rho = np.arange(0.1, 400, 0.01)
    for _,r in enumerate(rho): 
        for k in range(1000):#去瞬态
            x_n = Rk4(x0, h, r)
            x0 = x_n
            
        for n in range(3000):
            x_n = Rk4(x0, h, r)
            x, y = x_n[0], x_n[1]
            if (x < y + eps) & ( x > y - eps):#计算机中没有完全相同的两个数，如果直接写成相等的话图中只有一个点
                R.append(r) 
                X.append(x)
            x0 = x_n    
                
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
    plt.tick_params(labelsize=13) #刻度字体大小13        
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False   # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号             
    plt.scatter(R,X,color='g',s=1,linewidth=0.5)
    plt.xlabel('$\\rho$',fontsize=18)
    plt.ylabel('x',fontsize=18)    
    plt.title("Lorenz System's Bifurcation diagram(截面法)" )
    plt.show()


def Bifurcation(x0, h):#极值法
    Xmin, Xmax, R, R1 = [], [], [],[] #存储满足条件的x值和参数值
    rho = np.arange(1, 400, 0.1)
    for _,r in enumerate(rho): 
        
        for k in range(100):#去瞬态
            x_n = Rk4(x0, h, r)
            x0 = x_n
            
        for n in range(1000): #计算极值
            x_n = Rk4(x0, h, r) #x_n的变量分别为x，y，z
            
            if (x_n[0] < x0[0]) & (x_n[0] < Rk4(x_n, h, r)[0]):  #x的极小值                
                Xmin.append(x_n[0])
                R.append(r)
                        
                
            if (x_n[0] > x0[0]) & (x_n[0] > Rk4(x_n, h, r)[0]):  #x的极大值
                R1.append(r)   
                Xmax.append(x_n[0])
                
            x0 = x_n   
                    
    ax=plt.figure(1)     
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
    plt.tick_params(labelsize=13) #刻度字体大小13
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    plt.plot(R,Xmin,linewidth=0.5)
    plt.xlabel('$\\rho$',fontsize=16)
    plt.ylabel('x',fontsize=16)    
    # plt.xlim(0, 40, 5)
    # plt.ylim(-20, 60)
    plt.title("Lorenz System's Bifurcation diagram(极小值法)" )
    
    ax=plt.figure(2)      
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
    plt.tick_params(labelsize=13) #刻度字体大小13        
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False   # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号             
    plt.scatter(R1,Xmax,s=1,linewidth=0.5)
    plt.xlabel('$\\rho$',fontsize=16)
    plt.ylabel('x',fontsize=16)    
    plt.title("Lorenz System's Bifurcation diagram(极大值法)"  )
    plt.show()    

    
#洛伦兹系统的相位图
def Phase_diagram(x0, h, r = 28):
    X, Y, Z = [],[],[]
    # for k in range(100):
    #     x_n = Rk4(x0, h, r)
    #     x0 = x_n
    for n in range(3000):
        x_n = Rk4(x0, h, r)
        x, y, z = x_n[0], x_n[1], x_n[2]
        X.append(x)
        Y.append(y)
        Z.append(z)
        x0 = x_n
        
    ax = plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(4);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(4);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(4);####设置上部坐标轴的粗细
    plt.tick_params(labelsize=16) #刻度字体大小13    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False   # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号   
    fig = plt.figure()
    #创建3d图形的两种方式
    # ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d') 
    ax = fig.gca(projection="3d")
    plt.title("Lorenz map's Phase diagram" ,fontsize=18)
    ax.plot(X,Y,Z,linewidth=0.8)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    # plt.zlabel('z',fontsize=16)
    plt.show()


def Time_diagram(t0,x0,h,r):
    X,T=[],[]
    # for k in range(100):
    #     x_n=Rk4(x0,h,r)
    #     x0=x_n
    for k in range(10000):
        t = t0+ k * h
        x_n = Rk4(x0, h, r)
        x = x_n[0]
        X.append(x)
        T.append(t)
        x0 = x_n
        
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False   # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号    
    plt.scatter(T,X,s=1,linewidth=1)
    plt.xlabel('t')
    plt.ylabel('x')    
    plt.title("Lorenz's Time diagram" )
    plt.show()
 
    
tic = timer()
# Bifurcation_diagram([1., 1., 0], 0.01)   #截面法
# Bifurcation([1.2,0.9, 0.3], 0.01)        #极值法
Phase_diagram([1.0, 0., 0.5], 0.01, r=28)
# Time_diagram(0,[1.0, 0., 0.5], 0.01, r=28) 
   
toc = timer()
print("程序运行时间：" + str(toc - tic)+'s')












