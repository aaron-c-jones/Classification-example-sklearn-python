"""
Author: Aaron Jones
Date: 2017-12-09
"""

### Loading Libraries ----------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

execfile('classification_algorithm_testing_functions.py')

### Loading Data From UC Irvine Machine Learning Repository - Blood Transfusion Data Set ----------
blood = pd.read_table('http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
                      , sep = ','
                      , header = 0
                      , names = ['Recency', 'Frequency', 'Monetary', 'Time', 'March2007'])

### Plots ----------
colormap = plt.cm.RdBu
plt.figure(figsize = (14,12))
plt.title('Pearson Correlation of Features', y = 1.05, size = 15)
sns.heatmap(blood.astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True)

pairs = sns.pairplot(
    blood[[u'March2007', u'Recency', u'Frequency', u'Monetary', u'Time']],
    hue = 'March2007', palette = 'seismic', size = 1.2, diag_kind = 'kde', diag_kws = dict(shade = True), plot_kws = dict(s = 10)
)
pairs.set(xticklabels = [])

### Define Major Values ----------
SEED = 0
NFOLDS = 5

X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std = dataPrep(blood, 'March2007', SEED)

### Results Tables ----------
train_stats = []
test_stats = []

### Dummy ----------
dummyGrid = {'strategy': ['most_frequent', 'stratified', 'uniform'],
             'random_state': [SEED]}

model_dummy, y_train_dummy, y_test_dummy = modelFit(DummyClassifier(),
                                                    parameters=dummyGrid, whichXTrain=X_train, whichXTest=X_test, whichYTrain=y_train)

dummy_train_stats = modelScoring(y_train, y_train_dummy, 'Training', 'Dummy', printing = False, table = train_stats)
dummy_test_stats = modelScoring(y_test, y_test_dummy, 'Testing', 'Dummy', printing = False, table = test_stats)

### Gaussian Naive Bayes ----------
model_gnb, y_train_gnb, y_test_gnb = modelFit(GaussianNB(),
                                              parameters=None, whichXTrain=X_train, whichXTest=X_test, whichYTrain=y_train)

gnb_train_stats = modelScoring(y_train, y_train_gnb, 'Training', 'Gaussian Naive Bayes', printing = False, table = train_stats)
gnb_test_stats = modelScoring(y_test, y_test_gnb, 'Testing', 'Gaussian Naive Bayes', printing = False, table = test_stats)

### Logistic Regression ----------
lrGrid = {'penalty': ['l1', 'l2'],
          'C': np.linspace(0.01, 10000, 20),
          'random_state': [SEED]}

model_lr, y_train_lr, y_test_lr = modelFit(LogisticRegression(),
                                           parameters=lrGrid, whichXTrain=X_train_std, whichXTest=X_test_std, whichYTrain=y_train)

lr_train_stats = modelScoring(y_train, y_train_lr, 'Training', 'Logistic Regression', printing = False, table = train_stats)
lr_test_stats = modelScoring(y_test, y_test_lr, 'Testing', 'Logistic Regression', printing = False, table = test_stats)

### K Nearest Neighbors ----------
knnGrid = {'n_neighbors': np.linspace(1, 10, 10).astype(int),
           'weights': ['uniform', 'distance'],
           'p': np.linspace(1, 6, 6).astype(int),
           'algorithm': ['auto']}

model_knn, y_train_knn, y_test_knn = modelFit(KNeighborsClassifier(),
                                              parameters=knnGrid, whichXTrain=X_train_minmax, whichXTest=X_test_minmax, whichYTrain=y_train)

knn_train_stats = modelScoring(y_train, y_train_knn, 'Training', 'K Nearest Neighbors', printing = False, table = train_stats)
knn_test_stats = modelScoring(y_test, y_test_knn, 'Testing', 'K Nearest Neighbors', printing = False, table = test_stats)

### Support Vector Machines ----------
svcGrid = {'C': np.linspace(0.01, 10000, 20),
           'kernel': ['linear', 'rbf'],
           'class_weight': ['balanced', None],
           'random_state': [SEED]}

model_svc, y_train_svc, y_test_svc = modelFit(SVC(),
                                              parameters=svcGrid, whichXTrain=X_train_std, whichXTest=X_test_std, whichYTrain=y_train)

svc_train_stats = modelScoring(y_train, y_train_svc, 'Training', 'Support Vector Machine', printing = False, table = train_stats)
svc_test_stats = modelScoring(y_test, y_test_svc, 'Testing', 'Support Vector Machine', printing = False, table = test_stats)

### Random Forest ----------
rfGrid = {'n_estimators': [1000],
          'max_features': np.linspace(2, X_train.shape[1], 8).astype(int),
          'max_depth': np.linspace(2, 8, 7).astype(int),
          'class_weight': ['balanced', None],
          'bootstrap': [True],
          'oob_score': [True],
          'random_state': [SEED],
          'n_jobs': [-1]}

model_rf, y_train_rf, y_test_rf = modelFit(RandomForestClassifier(),
                                           parameters=rfGrid, whichXTrain=X_train, whichXTest=X_test, whichYTrain=y_train)

rf_train_stats = modelScoring(y_train, y_train_rf, 'Training', 'Random Forest', printing = False, table = train_stats)
rf_test_stats = modelScoring(y_test, y_test_rf, 'Testing', 'Random Forest', printing = False, table = test_stats)

### Gradient Boosting ----------
gbGrid = {'learning_rate': [0.01],
          'n_estimators': [500],
          'max_depth': np.linspace(2, 8, 7).astype(int),
          'max_features': np.linspace(2, X_train.shape[1], 8).astype(int),
          'random_state': [SEED]}

model_gb, y_train_gb, y_test_gb = modelFit(GradientBoostingClassifier(),
                                           parameters=gbGrid, whichXTrain=X_train, whichXTest=X_test, whichYTrain=y_train)

gb_train_stats = modelScoring(y_train, y_train_gb, 'Training', 'Gradient Boosting', printing = False, table = train_stats)
gb_test_stats = modelScoring(y_test, y_test_gb, 'Testing', 'Gradient Boosting', printing = False, table = test_stats)

### Neural Network ----------
model_nn, y_train_nn, y_test_nn = modelFit(MLPClassifier(max_iter=100000, random_state=SEED),
                                           parameters=None, whichXTrain=X_train_std, whichXTest=X_test_std, whichYTrain=y_train)

nn_train_stats = modelScoring(y_train, y_train_nn, 'Training', 'Neural Network', printing = False, table = train_stats)
nn_test_stats = modelScoring(y_test, y_test_nn, 'Testing', 'Neural Network', printing = False, table = test_stats)

### Results Data Frame ----------
train_fitness_stats = pd.DataFrame(train_stats,
                                   columns=['Algorithm', 'Accuracy', 'Sensitivity', 'Specificity',
                                            'Positive Predictive Value', 'Negative Predictive Value',
                                            'F1 Score', 'ROC AUC Score'])
test_fitness_stats = pd.DataFrame(test_stats,
                                  columns=['Algorithm', 'Accuracy', 'Sensitivity', 'Specificity',
                                           'Positive Predictive Value', 'Negative Predictive Value',
                                           'F1 Score', 'ROC AUC Score'])

print train_fitness_stats
print test_fitness_stats