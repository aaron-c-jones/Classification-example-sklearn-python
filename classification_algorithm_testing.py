import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import pandas
import seaborn
from sklearn import *


# Defining Functions
def accuracy_score(matrix):
    # Accuracy = (TP + TN) / (TP + TN + FN + FP)
    accuracy = (
        (matrix[0, 0] + matrix[1, 1])
        / float(matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])
    )
    return accuracy


def sensitivity_score(matrix):
    # Sensitivity = TP / (FN + TP)
    return matrix[1, 1] / float(matrix[1, 0] + matrix[1, 1])


def specificity_score(matrix):
    # Specificity = TN / (TN + FP)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[0, 1])


def positive_predictive_value_score(matrix):
    # Positive Predictive Value = TP / (FP + TP)
    return matrix[1, 1] / float(matrix[0, 1] + matrix[1, 1])


def negative_predictive_value_score(matrix):
    # Negative Predictive Value = TN / (TN + FN)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[1, 0])


def f1_score(actual, predicted):
    return metrics.f1_score(actual, predicted)


def roc_auc_score(actual, predicted):
    return metrics.roc_auc_score(actual, predicted)


def data_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = (
        model_selection.train_test_split(
            x, y, test_size=0.20, shuffle=True, random_state=0
        )
    )
    return x, y, x_train, x_test, y_train, y_test


def data_transform(train, test):
    std = preprocessing.StandardScaler()
    x_train_transform = std.fit_transform(train)
    x_test_transform = std.transform(test)
    return x_train_transform, x_test_transform


def model_scoring(which_data, actual, predicted):
    confusion = metrics.confusion_matrix(actual, predicted)

    acc = round(accuracy_score(confusion), 2)
    sen = round(sensitivity_score(confusion), 2)
    spe = round(specificity_score(confusion), 2)
    ppv = round(positive_predictive_value_score(confusion), 2)
    npv = round(negative_predictive_value_score(confusion), 2)
    f1s = round(f1_score(actual, predicted), 2)
    auc = round(roc_auc_score(actual, predicted), 2)

    print('Subset: {0}'.format(which_data))
    print('Accuracy: {0}'.format(acc))
    print('Sensitivity: {0}'.format(sen))
    print('Specificity: {0}'.format(spe))
    print('Positive Predictive Value: {0}'.format(ppv))
    print('Negative Predictive Value: {0}'.format(npv))
    print('F1 (Weighted Average of Precision and Recall): {0}'.format(f1s))
    print('ROC AUC: {0}'.format(auc))

    scores = (acc, sen, spe, ppv, npv, f1s, auc)
    return scores


def model_fit(algorithm, parameters, data, target):
    x, y, x_train, x_test, y_train, y_test = data_split(data, target)
    x_train_transform, x_test_transform = data_transform(x_train, x_test)

    model = model_selection.GridSearchCV(
        algorithm,
        param_grid=parameters,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    model.fit(x_train_transform, y_train)

    y_train_predicted = model.predict(x_train_transform)
    y_test_predicted = model.predict(x_test_transform)

    data_dictionary = {
        'x_train': x_train,
        'x_test': x_test,
        'x_train_transform': x_train_transform,
        'x_test_transform': x_test_transform,
        'y_train_actual': y_train,
        'y_test_actual': y_test,
        'y_train_predicted': y_train_predicted,
        'y_test_predicted': y_test_predicted
    }

    scoring_list = [
        ('Training', y_train, y_train_predicted),
        ('Holdout', y_test, y_test_predicted)
    ]

    for which_data, actual, predicted in scoring_list:
        scores = model_scoring(which_data, actual, predicted)

    outputs = (model, data_dictionary, scores)
    return outputs


# Iris Data
cancer = (
    pandas.read_csv(
        '/Users/aaronjones/Classification-example-sklearn-python/BreastCancer.csv',
        header=0,
        names=[
            'id', 'diag', 'radius', 'texture', 'perimeter', 'area',
            'smoothness', 'compactness', 'concavity', 'concave_points',
            'symmetry', 'fractal_dimension'
        ]
    )
    .sample(frac=1)
    .dropna(axis=0)
)

cancer.diag.replace(['M', 'B'], [1, 0], inplace=True)
cancer.drop(['id', 'perimeter', 'area'], axis=1, inplace=True)


# Plots
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features')
seaborn.heatmap(
  data=cancer.drop('diag', axis=1).astype(float).corr(),
  linewidths=0.1,
  vmax=1.0,
  square=True,
  cmap=plt.cm.RdBu,
  linecolor='white',
  annot=True
)

pairs = seaborn.pairplot(
    data=cancer,
    hue='diag',
    palette='seismic',
    size=0.8,
    diag_kind='hist'
).set(
    xticklabels=[]
)

scat = seaborn.FacetGrid(
    data=cancer[['radius', 'fractal_dimension', 'diag']],
    hue='diag',
    aspect=1.5
).map(
    plt.scatter, 'radius', 'fractal_dimension'
).add_legend()


# PCA Visualization
x_full, y_full, x_train, x_test, y_train, y_test = (
    data_split(cancer, 'diag')
)

pca = decomposition.PCA(n_components=2)
x_pca = pca.fit(x_full).transform(x_full)

print('Individual % Variance Explained by First Two Components: {0}'
      .format(str(pca.explained_variance_ratio_)))

print('Total % Variance Explained by First Two Components: {0}'
      .format(sum(pca.explained_variance_ratio_)))

colors = ['blue', 'red']
plot_info = zip(colors, [0, 1], ['Benign', 'Malignant'])
plt.figure()
for col, i, name in plot_info:
    plt.scatter(
        x_pca[y_full == i, 0], x_pca[y_full == i, 1],
        color=col, alpha=0.8, lw=2, label=name
    )
plt.legend(loc='best', shadow=True, scatterpoints=1)
plt.title('PCA Cancer Data')


# MDS Visualization
mds2 = manifold.MDS(
    n_components=2, metric=False, max_iter=10000, eps=1e-8,
    n_jobs=1, random_state=0, dissimilarity='euclidean'
)
em2d = mds2.fit(x_full).embedding_

mds3 = manifold.MDS(
    n_components=3, metric=False, max_iter=10000, eps=1e-8,
    n_jobs=1, random_state=0, dissimilarity='euclidean'
)
em3d = mds3.fit(x_full).embedding_

fig = plt.figure(figsize=(5 * 2, 5))
# plot 2d
plt2 = fig.add_subplot(121)
plt2.scatter(
    em2d[y_full == 0, 0], em2d[y_full == 0, 1],
    s=20, color='blue', label='Benign'
)
plt2.scatter(
    em2d[y_full == 1, 0], em2d[y_full == 1, 1],
    s=20, color='red', label='Malignant'
)
plt.axis('tight')
# plot 3d
plt3 = fig.add_subplot(122, projection='3d')
plt3.scatter(
    em3d[y_full == 0, 0], em3d[y_full == 0, 1], em3d[y_full == 0, 2],
    s=20, color='blue', label='Benign'
)
plt3.scatter(
    em3d[y_full == 1, 0], em3d[y_full == 1, 1], em3d[y_full == 1, 2],
    s=20, color='red', label='Malignant'
)
plt3.view_init(42, 101)
plt3.view_init(-130, -33)
plt.suptitle('2D and 3D Multidimensional Scaling Plots')
plt.axis('tight')
plt.legend(loc='best', shadow=True, scatterpoints=1)
plt.show()


# Dummy
dummy_grid = {
    'strategy': ['most_frequent', 'stratified', 'uniform'],
    'random_state': [0]
}

dummy_outputs = model_fit(
    algorithm=dummy.DummyClassifier(),
    parameters=dummy_grid, data=cancer, target='diag'
)
dummy_model, dummy_data_dict, dummy_scores = dummy_outputs


# Gaussian Naive Bayes
gnb_grid = {
    'priors': [None]
}

gnb_outputs = model_fit(
    algorithm=naive_bayes.GaussianNB(),
    parameters=gnb_grid, data=cancer, target='diag'
)
gnb_model, gnb_data_dict, gnb_scores = gnb_outputs


# Logistic Regression
lr_grid = {
    'penalty': ['l1', 'l2'],
    'C': numpy.linspace(0.01, 10000, 20),
    'random_state': [0]
}

logistic_outputs = model_fit(
    algorithm=linear_model.LogisticRegression(),
    parameters=lr_grid, data=cancer, target='diag'
)
logistic_model, logistic_data_dict, logistic_scores = logistic_outputs


# Linear Discriminant Analysis
lda_grid = {
    'solver': ['svd'],
    'store_covariance': [True]
}

lda_outputs = model_fit(
    algorithm=discriminant_analysis.LinearDiscriminantAnalysis(),
    parameters=lda_grid, data=cancer, target='diag'
)
lda_model, lda_data_dict, lda_scores = lda_outputs


# Quadratic Discriminant Analysis
qda_grid = {
    'store_covariance': [True]
}

qda_outputs = model_fit(
    algorithm=discriminant_analysis.QuadraticDiscriminantAnalysis(),
    parameters=qda_grid, data=cancer, target='diag'
)
qda_model, qda_data_dict, qda_scores = qda_outputs


# K Nearest Neighbors
knn_grid = {
    'n_neighbors': numpy.linspace(1, 10, 10).astype(int),
    'weights': ['uniform', 'distance'],
    'p': numpy.linspace(1, 6, 6).astype(int),
    'algorithm': ['auto']
}

knn_outputs = model_fit(
    algorithm=neighbors.KNeighborsClassifier(),
    parameters=knn_grid, data=cancer, target='diag'
)
knn_model, knn_data_dict, knn_scores = knn_outputs


# Support Vector Machine
svc_grid = {
    'C': numpy.linspace(0.01, 10000, 20),
    'kernel': ['linear', 'rbf'],
    'class_weight': ['balanced', None],
    'random_state': [0]
}

svc_outputs = model_fit(
    algorithm=svm.SVC(),
    parameters=svc_grid, data=cancer, target='diag'
)
svc_model, svc_data_dict, svc_scores = svc_outputs


# Random Forest
rf_grid = {
    'n_estimators': [1000],
    'max_features': numpy.linspace(2, cancer.shape[1]-1, 8).astype(int),
    'max_depth': numpy.linspace(2, 8, 7).astype(int),
    'class_weight': ['balanced', None],
    'bootstrap': [True],
    'oob_score': [True],
    'random_state': [0],
    'n_jobs': [-1]
}

rf_outputs = model_fit(
    algorithm=ensemble.RandomForestClassifier(),
    parameters=rf_grid, data=cancer, target='diag'
)
rf_model, rf_data_dict, rf_scores = rf_outputs


# Gradient Boosting
gb_grid = {
    'learning_rate': [0.01],
    'n_estimators': [500],
    'max_depth': numpy.linspace(2, 8, 7).astype(int),
    'max_features': numpy.linspace(2, cancer.shape[1]-1, 8).astype(int),
    'random_state': [0]
}

gb_outputs = model_fit(
    algorithm=ensemble.GradientBoostingClassifier(),
    parameters=gb_grid, data=cancer, target='diag'
)
gb_model, gb_data_dict, gb_scores = gb_outputs


# Results Data Frame
scores_list = [
    dummy_scores, gnb_scores, logistic_scores, knn_scores,
    svc_scores, rf_scores, gb_scores
]
holdout_stats = pandas.DataFrame(
    scores_list,
    columns=[
        'Accuracy', 'Sensitivity', 'Specificity',
        'Positive Predictive Value', 'Negative Predictive Value',
        'F1 Score', 'ROC AUC Score'
    ]
)
holdout_stats['Algorithms'] = [
    'Dummy', 'Gaussian Naive Bayes', 'Logistic Regression',
    'K Nearest Neighbors', 'Support Vector Classifier',
    'Random Forest', 'Gradient Boosting'
]
print(holdout_stats)
