from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# #调整数据尺度

# #指定将数据调整为0-1的数
# transformer = MinMaxScaler(feature_range=(0, 1))
# #数据转换
# newX = transfromer.fit_transform(X)

# #将数据进行正态化处理
# transformer = StandardScaler().fit(X)
# #数据转换
# newX = transfromer.transform(X)

# #将数据进行标准化处理（每一行的距离为1）
# transformer = Normalizer().fit(X)
# #数据转换
# newX = transfromer.transform(X)

#将数据进行二值处理（设定阀值，大于阀值为1，小于阀值为0）
transformer = Binarizer(threshold=0.0).fit(X)
#数据转换
newX = transformer.transform(X)

#设置数据的打印格式
set_printoptions(precision=3)
print(newX)
