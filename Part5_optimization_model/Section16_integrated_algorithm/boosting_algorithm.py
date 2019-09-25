from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

#导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
#将数据分为输入数据和输出数据
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)


# #AdaBoost
# num_tree = 100
# model = AdaBoostClassifier(n_estimators=num_tree, random_state=seed)

# #随机梯度提升
# num_tree = 100
# model = GradientBoostingClassifier(n_estimators=num_tree, random_state=seed)

#投票算法
cart = DecisionTreeClassifier()
models = []
model_logistic = LogisticRegression()
models.append(('logistic', model_logistic))
models_cart = DecisionTreeClassifier()
models.append(('cart', models_cart))
model_svc = SVC()
models.append(('svm', model_svc))
ensemble_model = VotingClassifier(estimators=models)

result = cross_val_score(ensemble_model, X, Y, cv=kfold)
print(result.mean())