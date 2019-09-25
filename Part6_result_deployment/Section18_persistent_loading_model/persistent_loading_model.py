from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from pickle import dump
# from pickle import load
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
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
test_size = 0.33
seed = 4
X_train, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#训练模型
model = LogisticRegression()
model.fit(X_train, Y_training)

# #通过pickle序列化和反序列化保存加载模型
# model_file = 'finalized_model.sav'

#通过joblib序列化和反序列化保存加载模型
model_file = 'finalized_model_joblib.sav'

with open(model_file, 'wb') as model_f:
    #模型序列化
    dump(model, model_f)

#加载模型
with open(model_file, 'rb') as  model_f:
    #模型反序列化
    loaded_model = load(model_f)
    result = loaded_model.score(X_test, Y_test)
    print("算法评估结果：%.3f%%" % (result*100))



