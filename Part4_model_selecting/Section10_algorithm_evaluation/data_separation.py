from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
import warnings
warnings.filterwarnings("ignore")
#导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# #分离训练数据集和评估数据集
# test_size = 0.33
# seed = 4
# X_train, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# model = LogisticRegression()
# model.fit(X_train, Y_training)
# result = model.score(X_test, Y_test)
# print("算法评估结果：%.3f%%" % (result*100))

# #K折交叉验证分离
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# result = cross_val_score(model, X, Y, cv=kfold)
# print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100))

# #弃一交叉验证分离
# loocv = LeaveOneOut()
# model = LogisticRegression()
# result = cross_val_score(model, X, Y, cv=loocv)
# print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100))

#重复随机分离（重复进行K折交叉验证）
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100))