from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)

# #线性算法
# #逻辑回归模型
# model = LogisticRegression()

# #线性判别分析
# model = LinearDiscriminantAnalysis()

# #非线性算法
# #K近邻算法
# model = KNeighborsClassifier()

# #贝叶斯分类器
# model = GaussianNB()

# #分类与回归树
# model = DecisionTreeClassifier()

#支持向量机
model = SVC()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())