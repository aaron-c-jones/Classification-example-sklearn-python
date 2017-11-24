"""
Author: Aaron Jones
Date: 2017-11-24
"""

### Loading Libraries ----------
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

### Loading Data From UC Irvine Machine Learning Repository - Blood Transfusion Data Set ----------
blood = pd.read_table('http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
                      , sep = ','
                      , header = 0
                      , names = ['Recency', 'Frequency', 'Monetary', 'Time', 'March2007'])

print blood.head()
print blood.shape
print blood.March2007.value_counts()

### Splitting The Data ----------
X = blood.drop('March2007', axis = 1)
y = blood.March2007
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, stratify = y)

### Scaling Data ----------
minmax = MinMaxScaler()
X_train_scaled = minmax.fit_transform(X_train)
X_test_scaled = minmax.transform(X_test)

### Functions ----------
def cmat(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

def accuracy(y_true, y_pred):
    matrix = cmat(y_true, y_pred)
    #Accuary = (TP + TN) / (TP + TN + FN + FP)
    acc = (matrix[0, 0] + matrix[1, 1]) / float(matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])
    return acc

def sensitivity(y_true, y_pred):
    matrix = cmat(y_true, y_pred)
    #Sensitivity = TP / (FN + TP)
    sens = matrix[1, 1] / float(matrix[1, 0] + matrix[1, 1])
    return sens

def specificity(y_true, y_pred):
    matrix = cmat(y_true, y_pred)
    #Specificity = TN / (TN + FP)
    spec = matrix[0, 0] / float(matrix[0, 0] + matrix[0, 1])
    return spec

def positive_predictive_value(y_true, y_pred):
    matrix = cmat(y_true, y_pred)
    #Positive Predictive Value = TP / (FP + TP)
    ppv = matrix[1, 1] / float(matrix[0, 1] + matrix[1, 1])
    return ppv

def negative_predictive_value(y_true, y_pred):
    matrix = cmat(y_true, y_pred)
    #Negative Predictive Value = TN / (TN + FN)
    npv = matrix[0, 0] / float(matrix[0, 0] + matrix[1, 0])
    return npv

def model_results(model, which, X1 = X_train_scaled, X2 = X_test_scaled, y1 = y_train, y2 = y_test):
    #roc curve, precision recall curve

    train_score = model.score(X1, y1)
    test_score = model.score(X2, y2)

    y_train_predicted = model.predict(X1)
    y_test_predicted = model.predict(X2)

    confusion_matrix_train = cmat(y1, y_train_predicted)
    confusion_matrix_test = cmat(y2, y_test_predicted)

    results_dict = {'Algorithm': which
        , 'Accuracy (Train, Test)': (round(accuracy(y1, y_train_predicted), 4),
                                     round(accuracy(y2, y_test_predicted), 4))
        , 'Sensitivity (Train, Test)': (round(sensitivity(y1, y_train_predicted), 4),
                                        round(sensitivity(y2, y_test_predicted), 4))
        , 'Specificity (Train, Test)': (round(specificity(y1, y_train_predicted), 4),
                                        round(specificity(y2, y_test_predicted), 4))
        , 'Positive Predictive Value (Train, Test)': (round(positive_predictive_value(y1, y_train_predicted), 4),
                                                      round(positive_predictive_value(y2, y_test_predicted), 4))
        , 'Negative Predictive Value (Train, Test)': (round(negative_predictive_value(y1, y_train_predicted), 4),
                                                      round(negative_predictive_value(y2, y_test_predicted), 4))}

    return train_score, test_score, confusion_matrix_train, confusion_matrix_test, results_dict

### Dummy Classifier - Most Frequent ----------
dum_mod = DummyClassifier(random_state = 0)
dum_grid = {'strategy': ['most_frequent', 'stratified']}
dum = GridSearchCV(dum_mod, dum_grid, scoring = 'accuracy')
dum.fit(X_train_scaled, y_train)

dum_train_score, dum_test_score, dum_train_confusion_matrix, dum_test_confusion_matrix, dum_results_dict = model_results(dum, 'Dummy')

print 'Mean Training Set Accuracy (based on 5 fold) - Dummy: {}'.format(dum_train_score)
print 'Test Set Accuracy - Dummy: {}'.format(dum_test_score)

### Naive Bayes ----------
gnb = GaussianNB().fit(X_train_scaled, y_train)

gnb_train_score, gnb_test_score, gnb_train_confusion_matrix, gnb_test_confusion_matrix, gnb_results_dict = model_results(gnb, 'Gaussian Naive Bayes')

print 'Mean Training Set Accuracy (based on 5 fold) - Gaussian Naive Bayes: {}'.format(gnb_train_score)
print 'Test Set Accuracy - Gaussian Naive Bayes: {}'.format(gnb_test_score)

### Logistic Regression ----------
lr_mod = LogisticRegression(random_state = 0, max_iter = 10000)
lr_grid = {'penalty': ['l1', 'l2']
           , 'C': np.power(10.0, np.arange(-10, 10))}
lr = GridSearchCV(lr_mod, lr_grid, scoring = 'accuracy')
lr.fit(X_train_scaled, y_train)

lr_train_score, lr_test_score, lr_train_confusion_matrix, lr_test_confusion_matrix, lr_results_dict = model_results(lr, 'Logistic Regression CV')

print 'Mean Training Set Accuracy (based on 5 fold) - Logistic Regression CV: {}'.format(lr_train_score)
print 'Test Set Accuracy - Logistic Regression CV: {}'.format(lr_test_score)

### K Nearest Neighbors ----------
knn_mod = KNeighborsClassifier()
knn_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
            , 'weights': ['uniform', 'distance']
            , 'p': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn = GridSearchCV(knn_mod, knn_grid, scoring = 'accuracy')
knn.fit(X_train_scaled, y_train)

knn_train_score, knn_test_score, knn_train_confusion_matrix, knn_test_confusion_matrix, knn_results_dict = model_results(knn, 'K Nearest Neighbors')

print 'Mean Training Set Accuracy (based on 5 fold) - K Nearest Neighbors: {}'.format(knn_train_score)
print 'Test Set Accuracy - K Nearest Neighbors: {}'.format(knn_test_score)

### Support Vector Machine ----------


### Random Forest ----------


### Gradient Boosting Machine ----------


### Neural Network ----------


### Results Data Frame ----------
results_final = (pd.DataFrame([dum_results_dict
                               , gnb_results_dict
                               , lr_results_dict
                               , knn_results_dict])
                 .reindex_axis(['Algorithm'
                                , 'Accuracy (Train, Test)'
                                , 'Sensitivity (Train, Test)'
                                , 'Specificity (Train, Test)'
                                , 'Positive Predictive Value (Train, Test)'
                                , 'Negative Predictive Value (Train, Test)'], axis = 1))
print results_final