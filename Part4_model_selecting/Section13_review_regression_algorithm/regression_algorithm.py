from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")
#导入数据
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names=names, delim_whitespace=True)
#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:13]
Y = array[:, 13]
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)

# #线性算法
# #线性回归算法
# model = LinearRegression()

# #岭回归算法
# model = Ridge()

# #套索回归算法
# model = Lasso()

# #弹性网络回归算法
# model = ElasticNet()

#非线性算法

# #K邻近
# model = KNeighborsRegressor()

# #分类和回归树
# model = DecisionTreeRegressor()

#支持向量机
model = SVR()

scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('Regression: %.3f' % result.mean())