import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

import pandas as pd


def read_data(config):
    train_data_path = f'new_configs/train_config{config}.csv'
    test_data_path = f'new_configs/test_config{config}.csv'

    train_data_df = pd.read_csv(train_data_path)
    test_data_df = pd.read_csv(test_data_path)

    return train_data_df, test_data_df


def read_data_once(config):
    data_path = f'new_configs/config{config}.csv'

    data_df = pd.read_csv(data_path)

    return data_df


def delete_files():
    if os.path.exists('Results/results_svm.csv'):
        os.remove('Results/results_svm.csv')
    if os.path.exists('Results/results_random_forest.csv'):
        os.remove('Results/results_random_forest.csv')
    if os.path.exists('Results/results_gradient_boosting.csv'):
        os.remove('Results/results_gradient_boosting.csv')
    if os.path.exists('Results/results_knn.csv'):
        os.remove('Results/results_knn.csv')
    if os.path.exists('Results/results_naive_bayes.csv'):
        os.remove('Results/results_naive_bayes.csv')
    if os.path.exists('Results/results_lightgbm.csv'):
        os.remove('Results/results_lightgbm.csv')


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
    # print(len(X_train))
    # print(len(X_test))
    return X_train, X_test
