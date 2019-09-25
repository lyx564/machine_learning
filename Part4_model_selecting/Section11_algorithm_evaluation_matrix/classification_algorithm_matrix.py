from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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


# #分类准确度
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# result = cross_val_score(model, X, Y, cv=kfold)
# print("算法评估结果准确度：%.3f (%.3f)" % (result.mean(), result.std()))

# #对数损失函数
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# scoring = 'neg_log_loss'
# result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print("Logloss %.3f (%.3f)" % (result.mean(), result.std()))

# #AUC图
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# scoring = 'roc_auc'
# result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print('AUC %.3F (%.3F)' % (result.mean(), result.std()))

# #混淆矩阵
# test_size = 0.33
# seed = 4
# X_train, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# model = LogisticRegression()
# model.fit(X_train, Y_training)
# predicted = model.predict(X_test)
# matrix = confusion_matrix(Y_test, predicted)
# classes = ['0', '1']
# dataframe = pd.DataFrame(data=matrix,
#                          index=classes,
#                          columns=classes)
# print(dataframe)

#分类报告
test_size = 0.33
seed = 4
X_train, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_training)
predicted = model.predict(X_test)
report = classification_report(Y_test , predicted)
print(report)