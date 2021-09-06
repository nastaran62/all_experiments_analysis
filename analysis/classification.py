import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    mean_squared_error, classification_report, accuracy_score

def feature_selection_random_forest(x_train, x_test, y_train, y_test, num_features=10):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    rfe = RFE(clf, num_features)
    fit = rfe.fit(x_train, y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    x_train = fit.transform(x_train)
    x_test = fit.transform(x_test)
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    return acc, f_score, pred_values

def random_forest(x_train, x_test, y_train, y_test):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    return acc, f_score, pred_values

def svm_classification(x_train, x_test, y_train, y_test):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std
    clf = svm.SVC(probability=True,
                  class_weight='balanced',
                  C=500,
                  random_state=100,
                  kernel='rbf')
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    print(acc)
    return acc, f_score, pred_values