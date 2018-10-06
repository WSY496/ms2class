# import some library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import lightgbm as lgb
import itertools
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# read data
df = pd.read_csv('../data/feature_selection_positive.csv', index_col=0)

# divide objective and target
objective = df.Subclass
le = preprocessing.LabelEncoder()
objective = le.fit_transform(objective)
features = df.drop('Subclass', axis=1)

# train test split
random_state=np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    objective,
    test_size=0.2
)

# # Use large max_bin (may be slower)
# # Use small learning_rate with large num_iterations
# # Use large num_leaves (may cause over-fitting)
# # Try dart

# # initial parameters on LGBMClassifier
# # boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, 
# # subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, 
# # min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, 
# # colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, 
# # silent=True, importance_type='split', **kwargs

params = {
    'num_leaves': [31, 50, 100],
    'max_bin':[255],
    'max_depth': [100, 200, -1],
    'min_child_samples': [10, 20, 30, 40, 50],
    'boosting': ['gbdt']
}

gbm = lgb.LGBMClassifier(
    objective='multiclass',
    device = 'gpu',
    gpu_device_id=1,
    n_jobs=-1 
)

clf = GridSearchCV(
    gbm,
    params,
    verbose=2,
    cv=3,
    n_jobs=-1
)

clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
# pickle.dump(gbm, open('../model/LGBM_best_params_fs.sav', 'wb'))

# accuracy
t = clf.best_estimater_
