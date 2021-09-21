# https://blog.csdn.net/suzyu12345/article/details/78338091
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation  # 动图的核心函数
# import seaborn as sns  # 美化图形的一个绘图包

# sns.set_style("whitegrid")  # 设置图形主图

# # 创建画布
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)

# # 画出一个维持不变（不会被重画）的散点图和一开始的那条直线。
# x = np.arange(0, 20, 0.1)
# ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
# line, = ax.plot(x, x - 5, 'r-', linewidth=2)

# def update(i):
#     label = 'timestep {0}'.format(i)
#     print(label)
#     # 更新直线和x轴（用一个新的x轴的标签）。
#     # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
#     line.set_ydata(x - 5 + i)  # 这里是重点，更新y轴的数据
#     ax.set_xlabel(label)    # 这里是重点，更新x轴的标签
#     return line, ax

# # FuncAnimation 会在每一帧都调用“update” 函数。
# # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
# anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)


# from matplotlib import pyplot as plt
# from matplotlib import animation
# import numpy as np
# import seaborn as sns
# sns.set_style("whitegrid")


# def randn_point():
#     # 产生随机散点图的x和y数据
#     x=np.random.randint(1,100,3)
#     y=np.random.randint(1,2,3)
#     return x,y

# # 创建画布，包含2个子图
# fig = plt.figure(figsize=(15, 10))
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)

# # 先绘制初始图形，每个子图包含1个正弦波和三个点的散点图
# x = np.arange(0, 2*np.pi, 0.01)

# line1, = ax1.plot(x, np.sin(x)) # 正弦波
# x1,y1=randn_point()
# sca1 = ax1.scatter(x1,y1)   # 散点图

# line2, = ax2.plot(x, np.cos(x))  # 余弦波
# x2,y2=randn_point()
# sca2 = ax2.scatter(x2,y2)   # 散点图

# def init():
#     # 构造开始帧函数init
#     # 改变y轴数据，x轴不需要改
#     line1.set_ydata(np.sin(x))
#     line1.set_ydata(np.cos(x))
#     # 改变散点图数据
#     x1, y1 = randn_point()
#     x2, y2 = randn_point()
#     data1 = [[x,y] for x,y in zip(x1,y1)]
#     data2 = [[x, y] for x, y in zip(x2, y2)]
#     sca1.set_offsets(data1)  # 散点图
#     sca2.set_offsets(data2)  # 散点图
#     label = 'timestep {0}'.format(0)
#     ax1.set_xlabel(label)
#     return line1,line2,sca1,sca2,ax1  # 注意返回值，我们要更新的就是这些数据

# def animate(i):
#     # 接着，构造自定义动画函数animate，用来更新每一帧上各个x对应的y坐标值，参数表示第i帧
#     # plt.cla() 这个函数很有用，先记着它
#     line1.set_ydata(np.sin(x + i/10.0))
#     line2.set_ydata(np.cos(x + i / 10.0))
#     x1, y1 = randn_point()
#     x2, y2 = randn_point()
#     data1 = [[x,y] for x,y in zip(x1,y1)]
#     data2 = [[x, y] for x, y in zip(x2, y2)]
#     sca1.set_offsets(data1)  # 散点图
#     sca2.set_offsets(data2)  # 散点图
#     label = 'timestep {0}'.format(i)
#     ax1.set_xlabel(label)
#     return line1,line2,sca1,sca2,ax1


# # 接下来，我们调用FuncAnimation函数生成动画。参数说明：
# # fig 进行动画绘制的figure
# # func 自定义动画函数，即传入刚定义的函数animate
# # frames 动画长度，一次循环包含的帧数
# # init_func 自定义开始帧，即传入刚定义的函数init
# # interval 更新频率，以ms计
# # blit 选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显示动画

# ani = animation.FuncAnimation(fig=fig,
#                               func=animate,
#                               frames=100,
#                               init_func=init,
#                               interval=20,
#                               blit=True)
# plt.show()

# #https://www.bilibili.com/read/cv11339746
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio

# # 生成40个取值在30-40的数
# y = np.random.randint(30, 40, size=(10)) 

# filenames = []
# num = 0
# for i in y:
#     num += 1
#     # 绘制40张折线图
#     plt.plot(y[:num])
#     plt.ylim(20, 50)

#     # 保存图片文件
#     filename = f'{num}.png'
#     filenames.append(filename)
#     plt.savefig(filename)
#     plt.close()

# # 生成gif
# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# # 删除40张折线图
# for filename in set(filenames):
#     os.remove(filename) 

# # https://zhuanlan.zhihu.com/p/139084960
# import matplotlib.animation as ani
# import matplotlib.pyplot as plt
# import pandas as pd
# url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
# df = pd.read_csv(url, delimiter=',', header='infer')
# df_interest = df.loc[
#     df['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany'])
#     & df['Province/State'].isna()]
# df_interest.rename(
#     index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
# df1 = df_interest.transpose()
# df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
# df1 = df1.loc[(df1 != 0).any(1)]
# df1.index = pd.to_datetime(df1.index)

# color = ['red', 'green', 'blue', 'orange']
# fig = plt.figure()
# plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
# plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
# plt.ylabel('No of Deaths')
# plt.xlabel('Dates')

# def buildmebarchart(i=int):
#     plt.legend(df1.columns)
#     p = plt.plot(df1[:i].index, df1[:i].values) #note it only returns the dataset, up to the point i
#     for i in range(0,4):
#         p[i].set_color(color[i]) #set the colour of each curve
# animator = ani.FuncAnimation(fig, buildmebarchart, interval = 100)
# animator.save('C:/Users/x/Desktop/myfirstAnimation.gif')

