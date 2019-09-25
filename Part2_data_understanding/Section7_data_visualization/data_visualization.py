from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
#data.hist()         #直方图
#data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)    #密度图
#data.plot(kind='box', subplots=True, layout=(3,3), sharex=False)    #箱线图

# #相关矩阵图
# correlations = data.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, 9, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)

scatter_matrix(data)

plt.show()