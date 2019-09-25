from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
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

# #装袋决策树
# cart = DecisionTreeClassifier()
# num_tree = 100
# model = BaggingClassifier(base_estimator=cart, n_estimators=num_tree, random_state=seed)

# #随机森林
# num_tree = 100
# max_features = 3
# model = RandomForestClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)

#极端随机树
num_tree = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)

result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

