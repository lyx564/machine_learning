from pandas import read_csv
from pandas import set_option
#印第安人糖尿病数据集
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# print(data.shape)       #数据维度
# print()
# print(data.dtypes)      #数据类型
# print()
# peek = data.head(10)
# print(peek)             #前10条数据
# print()
# set_option('display.width', 100)
# set_option('precision', 4)
# print(data.describe())      #数据描述

#print(data.groupby('class').size())        #数据根据class分组的情况

# set_option('display.width', 100)
# set_option('precision', 2)
# print(data.corr(method='pearson'))          #显示数据属性的相关性

print(data.skew())  #计算数据的高斯偏离