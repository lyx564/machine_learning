from csv import reader
import numpy as np
from numpy import loadtxt
from pandas import read_csv

#印第安人糖尿病数据集
filename = 'pima_data.csv'

# #使用标准的Python类库导入CSV数据
# with open(filename, 'rt') as raw_data:
#     readers = reader(raw_data, delimiter=',')
#     x = list(readers)
#     data = np.array(x).astype('float')
#     print(data.shape)

# #使用NumPy导入CSV数据
# with open(filename, 'rt') as raw_data:
#     data = loadtxt(raw_data, delimiter=',')
#     print(data.shape)

#使用Pandas导入CSV数据
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)