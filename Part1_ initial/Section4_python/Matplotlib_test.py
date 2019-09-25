import matplotlib.pyplot as plt
import numpy as np

myarray = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# #初始化绘图
# plt.plot(myarray)
# #设定x轴和y轴
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# #绘图
# plt.show()

myarray1 = myarray
myarray2 = myarray1*10+1
#初始化绘图
plt.scatter(myarray1, myarray2)
#设定x轴和y轴
plt.xlabel('x axis')
plt.ylabel('y axis')
#绘图
plt.show()