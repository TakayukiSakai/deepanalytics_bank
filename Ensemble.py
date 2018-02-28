
# coding: utf-8

# ## 複数の分類器で平均

# In[1]:

import os
import datetime
import multiprocessing
import numpy as np
import pandas as pd
from scipy import interp
from patsy import dmatrices
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


# In[2]:

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# In[3]:

def getDayOfYear(month, day): # ex. month = "mar", day = 15
    return int(datetime.datetime.strptime("{} {}".format(month, day), '%b %d').date().strftime('%j'))

train_df['dayofyear'] = np.vectorize(getDayOfYear)(train_df['month'], train_df['day'])
train_df['dayofweek'] = train_df['dayofyear'] % 7

test_df['dayofyear'] = np.vectorize(getDayOfYear)(test_df['month'], test_df['day'])
test_df['dayofweek'] = test_df['dayofyear'] % 7


# In[4]:

def getPDayOfWeek(pdays):
    return -1 if pdays == -1 else pdays % 7

train_df['pdayofweek'] = np.vectorize(getPDayOfWeek)(train_df['pdays'])
test_df['pdayofweek'] = np.vectorize(getPDayOfWeek)(test_df['pdays'])


# In[5]:

# 質的変数をダミー変数化
y_, X_ = dmatrices('y ~ age + job + marital + education + default + balance + housing + loan + contact + day + month + dayofyear + dayofweek + duration + campaign + pdays + pdayofweek + previous + poutcome', data=train_df, return_type='dataframe')
X_train = X_.values
y_train = y_.y.values

id_, X_ = dmatrices('id ~ age + job + marital + education + default + balance + housing + loan + contact + day + month + dayofyear + dayofweek + duration + campaign + pdays + pdayofweek + previous + poutcome', data=test_df, return_type='dataframe')
X_test = X_.values
id_test = id_.id.values.astype(int)


# In[6]:

def fitKFold(base_classifiers, X, y, n_folds=6):
    classifiers = []
    cv = StratifiedKFold(y, n_folds=n_folds)

    print("start fitting")
    for base_classifier in base_classifiers:
        print('fitting: ({})'.format(type(base_classifier).__name__))
        for i, (train, test) in enumerate(cv):
            print(i, end=': ')
            classifier = clone(base_classifier)
            classifier.fit(X[train], y[train])
            classifiers.append(classifier)

            y_scores = classifier.predict_proba(X[test])[:, 1]
            auc = roc_auc_score(y[test], y_scores)
            print(auc)
    print("done fitting")

    return classifiers

def predict(classifiers, X):
    print("start prediction")
    probas_list = []
    for classifier in classifiers:
        probas_list.append(classifier.predict_proba(X)[:, 1])
    probas_ = np.array(probas_list).mean(axis=0)
    print("done prediction")

    return probas_

def create_submission(id, y):
    if not os.path.isdir('subm'):
        os.mkdir('subm')

    d = { 'id': id, 'y': y }

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')

    pd.DataFrame(data=d).to_csv(sub_file, header=False, index=False)

# In[8]:

gbdt_model = GradientBoostingClassifier(n_estimators=500)
rf_model = RandomForestClassifier(n_estimators=2000, n_jobs=multiprocessing.cpu_count())

base_models = [gbdt_model, rf_model]

models = fitKFold(base_models, X_train, y_train)
y_test = predict(models, X_test)

create_submission(id_test, y_test)

