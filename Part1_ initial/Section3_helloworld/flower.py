from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

filename = 'iris.data.csv'
names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

##显示数据维度
#print('数据维度：行 %s, 列 %s' % dataset.shape)

##查看数据前10行
#print(dataset.head(10))

##统计描述数据信息
#print(dataset.describe())

##分类分布情况
#print(dataset.groupby('class').size())

# #箱线图
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# #直方图
# dataset.hist()
# pyplot.show()

# #散点矩阵图
# scatter_matrix(dataset)
# pyplot.show()

#分离数据集
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=validation_size,
                     random_state=seed)
# #算法审查
# #获取模型
# models = {}
# models['LR'] = LogisticRegression()             #线性回归
# models['LDA'] = LinearDiscriminantAnalysis()    #线性判别分析
# models['KNN'] = KNeighborsClassifier()          #K近邻
# models['CART'] = DecisionTreeClassifier()       #分类与回归树
# models['NB'] = GaussianNB()                     #贝叶斯分类器
# models['SVM'] = SVC()                           #支持向量机
# #评估算法
# results = []
# for key in models:
#     kfold = KFold(n_splits=10, random_state=seed)
#     cv_results = cross_val_score(models[key], X_train,
#                                  Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     print('%s: %f (%f)' %(key, cv_results.mean(),
#                           cv_results.std()))
#
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(models.keys())
# pyplot.show()

#使用评估数据集评估算法
svm = SVC()          #获取模型
svm.fit(X=X_train, y=Y_train)           #训练模型
predictions = svm.predict(X_validation)             #预测
print(accuracy_score(Y_validation, predictions))        #计算准确率
print(confusion_matrix(Y_validation, predictions))      #混淆矩阵
print(classification_report(Y_validation, predictions))     #报告，精确率，召回率，F1值




