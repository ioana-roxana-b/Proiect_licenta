import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def svm(config, pca=False, scal=False):

    if config == 1:
        data_df = pd.read_csv('config1.csv')
    elif config == 2:
        data_df = pd.read_csv('config2.csv')
    elif config == 3:
        data_df = pd.read_csv('config3.csv')
    elif config == 4:
        data_df = pd.read_csv('config4.csv')

    # Split the data into X and y
    X = data_df.drop('label', axis=1).values
    y = data_df['label'].values

    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    if scal == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if pca == True:
        pca = PCA(n_components=10)
        new_X = pca.fit_transform(X)

        tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
        clf.fit(new_X, y)

        new_X_test = pca.fit_transform(X_test)
        y_pred = clf.predict(new_X_test)
    else:
        tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

    print(y_test)
    print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1, "\n")
