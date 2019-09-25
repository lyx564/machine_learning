from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
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

# #数据准备和生成模型的Pipeline
# steps = []
# steps.append(('Standardize', StandardScaler()))
# steps.append(('lda', LinearDiscriminantAnalysis()))

#特征选择和生成模型的Pipeline
#生成FeatureUnion
features = []
features.append(('pca',PCA()))
features.append(('select_best',SelectKBest(k=6)))
#生成Pipeline
steps = []
steps.append(('feature_union', FeatureUnion(features)))
steps.append(('logistic', LogisticRegression()))
model = Pipeline(steps)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())
