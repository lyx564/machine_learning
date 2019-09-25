import numpy as np
#创建数组
# myarray = np.array([1, 2, 3])
# print(myarray)
# print(myarray.shape)

myarray = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# print(myarray)
# print(myarray.shape)
# print('第一行：%s' % myarray[0])
# print('最后一行：%s' % myarray[-1])
# print('访问整列（第3列）数据：%s' % myarray[:, 2])
# print('访问指定行（第2行）和列（第3列）的数据：%s' % myarray[1, 2])
#
myarray1 = myarray
myarray2 = myarray*10+1
print(myarray2)
print(myarray1 + myarray2)
print(myarray1 * myarray2)

