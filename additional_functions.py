import glob
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVR


def read_data(config):
    train_data_path = f'configs/train_config{config}.csv'
    test_data_path = f'configs/test_config{config}.csv'

    train_data_df = pd.read_csv(train_data_path)
    test_data_df = pd.read_csv(test_data_path)

    return train_data_df, test_data_df


def read_data_once(config):
    data_path = f'configs/config{config}.csv'
    data_df = pd.read_csv(data_path)
    return data_df


def delete_files(dir):
    csv_files = glob.glob(os.path.join(dir, "*.csv"))
    for file_path in csv_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

def pca(X_train, X_test):
    pca = PCA(n_components=10)
    pca.fit(X_train)
    new_X = pca.transform(X_train)
    new_X_test = pca.transform(X_test)
    return new_X, new_X_test


def minmax_sc(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def stand_sc(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def lasso(X_train, X_test, y_train):
    lass = Lasso(alpha=0.01, max_iter=10000)
    lass.fit(X_train, y_train)
    coef = lass.coef_
    idx_nonzero = np.nonzero(coef)[0]
    X_train = X_train[:, idx_nonzero]
    X_test = X_test[:, idx_nonzero]
    return X_train, X_test

def lasso_threshold(X_train, X_test, y_train):
    threshold = 0.01
    lasso = Lasso(alpha=0.01, max_iter=100000, tol=1e-4)
    lasso.fit(X_train, y_train)
    coef = lasso.coef_

    idx_above_threshold = np.where(np.abs(coef) > threshold)[0]
    X_train = X_train[:, idx_above_threshold]
    X_test = X_test[:, idx_above_threshold]

    return X_train, X_test


def recursive_feature_elimination(X_train, y_train):
    #model = LogisticRegression(solver='liblinear')
    #model = LinearSVC()
    #model = SVR(kernel="linear")
    model = DecisionTreeClassifier(random_state=50)
    rfe = RFECV(estimator=model, n_jobs=-1, cv=5, scoring='accuracy')
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    return X_train_rfe, rfe




