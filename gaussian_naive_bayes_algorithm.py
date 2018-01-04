"""
Title: Binary Classification Naive Bayes
Author: Aaron Jones
Date: 2017-12-09
"""

### Loading Libraries ----------
from math import pi, exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

### Making Data ----------
X, y = make_classification(n_samples = 200, n_features = 4, n_informative = 4, n_redundant = 0, n_classes = 2, random_state = 0)

### Plots ----------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = False, sharey = False)
ax1.hist(X[:, 0])
ax2.hist(X[:, 1])
ax3.hist(X[:, 2])
ax4.hist(X[:, 3])

### Splitting The Data ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

### Gaussian Naive Bayes from SKLEARN ----------
model_gnb = GaussianNB().fit(X_train, y_train)

y_train_gnb = model_gnb.predict(X_train)
y_test_gnb = model_gnb.predict(X_test)

print 'Training Confusion Matrix (sklearn): {}'.format(confusion_matrix(y_train, y_train_gnb))
print 'Testing Confusion Matrix (sklearn): {}'.format(confusion_matrix(y_test, y_test_gnb))

print 'Training Accuracy Score (sklearn): {}'.format(accuracy_score(y_train, y_train_gnb))
print 'Testing Accuracy Score (sklearn: {}'.format(accuracy_score(y_test, y_test_gnb))

### Gaussian Naive Bayes from OWN CODE ----------
def normal_probability(x, mean, sd):
    variance = float(sd)**2
    numerator = exp(-(float(x)-float(mean))**2 / (2*variance))
    denominator = (2*pi*variance)**0.5
    return numerator / denominator


class gnbBinaryClassifier():
    def number_of_parameters(self, X_train):
        self.n_variable = X_train.shape[1]

    def index_of_class1(self, y_train):
        self.index1 = np.argwhere(y_train)

    def prior_probability_class1(self, y_train):
        length_y_train = len(y_train)
        length_class1 = len(self.index1)
        self.probability_class1 = length_class1 / float(length_y_train)

    def distribution_parameters_class1(self, X_train):
        mean1 = []
        sd1 = []
        for i in range(self.n_variable):
            compute_parameter_mean1 = np.mean(X_train[self.index1, i])
            mean.append(compute_parameter_mean1)
            compute_parameter_sd1 = np.std(X_train[self.index1, i])
            sd.append(compute_parameter_sd1)
        self.mean1 = mean1
        self.sd1 = sd1

    def naive_bayes_classifier(self, X_test):
        variable_probability1 = []
        for j in range(self.n_variable):
            compute_variable_probability1 = normal_probability(X_test, self.mean1[j], self.sd1[j])
            variable_probability1.append(compute_variable_probability1)
        product_variable_probability1 = np.prod(variable_probability1)







