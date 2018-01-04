from sklearn.model_selection import train_test_split, GridSearchCV
#from spark_sklearn.grid_search import GridSearchCV
#from spark_sklearn.util import createLocalSparkSession
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

#sc = createLocalSparkSession().sparkContext

### Functions ----------
def dataPrep(data, target, seed):
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    standard = StandardScaler()
    X_train_std = standard.fit_transform(X_train)
    X_test_std = standard.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std

def modelFit(algorithm, parameters, whichXTrain, whichXTest, whichYTrain):
    if parameters is not None:
        model = GridSearchCV(algorithm, param_grid=parameters, scoring='accuracy', cv=NFOLDS, n_jobs=-1, verbose=1)
        model.fit(whichXTrain, whichYTrain)
    else:
        model = algorithm
        model.fit(whichXTrain, whichYTrain)

    y_train_pred = model.predict(whichXTrain)
    y_test_pred = model.predict(whichXTest)

    return model, y_train_pred, y_test_pred

def confusionMatrix(actual, predicted):
    return confusion_matrix(actual, predicted)

def accuracyScore(matrix):
    # Accuary = (TP + TN) / (TP + TN + FN + FP)
    return (matrix[0, 0] + matrix[1, 1]) / float(matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])

def sensitivityScore(matrix):
    # Sensitivity = TP / (FN + TP)
    return matrix[1, 1] / float(matrix[1, 0] + matrix[1, 1])

def specificityScore(matrix):
    # Specificity = TN / (TN + FP)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[0, 1])

def positivePredictiveValueScore(matrix):
    # Positive Predictive Value = TP / (FP + TP)
    return matrix[1, 1] / float(matrix[0, 1] + matrix[1, 1])

def negativePredictiveValueScore(matrix):
    # Negative Predictive Value = TN / (TN + FN)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[1, 0])

def F1Score(actual, predicted):
    return f1_score(actual, predicted)

def rocAucScore(actual, predicted):
    return roc_auc_score(actual, predicted)

def modelScoring(actual, predicted, whichData, whichModel, table, printing = False):
    matrix = confusionMatrix(actual, predicted)
    accuracy = round(accuracyScore(matrix), 4)
    sensitivity = round(sensitivityScore(matrix), 4)
    specificity = round(specificityScore(matrix), 4)
    positive_predictive_value = round(positivePredictiveValueScore(matrix), 4)
    negative_predictive_value = round(negativePredictiveValueScore(matrix), 4)
    f1 = round(F1Score(actual, predicted), 4)
    roc_auc = round(rocAucScore(actual, predicted), 4)

    if printing == True:
        print '{} - Accuracy on {} Data: {}'.format(whichModel, whichData, accuracy)
        print '{} - Sensitivity on {} Data: {}'.format(whichModel, whichData, sensitivity)
        print '{} - Specificity on {} Data: {}'.format(whichModel, whichData, specificity)
        print '{} - Positive Predictive Value on {} Data: {}'.format(whichModel, whichData, positive_predictive_value)
        print '{} - Negative Predictive Value on {} Data: {}'.format(whichModel, whichData, negative_predictive_value)
        print '{} - F1 (Weighted Average of Precision and Recall) on {} Data: {}'.format(whichModel, whichData, f1)
        print '{} - ROC AUC on {} Data: {}'.format(whichModel, whichData, roc_auc)

    results = {'Algorithm': whichModel,
               'Accuracy': accuracy,
               'Sensitivity': sensitivity,
               'Specificity': specificity,
               'Positive Predictive Value': positive_predictive_value,
               'Negative Predictive Value': negative_predictive_value,
               'F1 Score': f1,
               'ROC AUC Score': roc_auc}

    table.append(results)
    return results

