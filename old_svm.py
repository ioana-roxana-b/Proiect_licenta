import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import feature_vect


def svm(config, pca=False, scal=False):
    if config == 1:
        train_data_df = pd.read_csv('train_config1.csv')
        test_data_df = pd.read_csv('test_config1.csv')
    elif config == 2:
        train_data_df = pd.read_csv('train_config2.csv')
        test_data_df = pd.read_csv('test_config2.csv')
    elif config == 3:
        train_data_df = pd.read_csv('train_config3.csv')
        test_data_df = pd.read_csv('test_config3.csv')
    elif config == 4:
        train_data_df = pd.read_csv('train_config4.csv')
        test_data_df = pd.read_csv('test_config4.csv')

    X_train = train_data_df.drop('label', axis=1).values
    y_train = train_data_df['label'].values

    X_test = test_data_df.drop('label', axis=1).values
    y_test = test_data_df['label'].values

    if scal == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if pca == True:
        X = np.concatenate((X_test, X_train))
        y = np.concatenate((y_train, y_test))

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