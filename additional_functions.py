import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import svm
import random_forest
import knn
import naive_bayes
import voting_classifier as vote
import lgbm
import grad_boosting
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
    if os.path.exists('Results/results_voting_classifier.csv'):
        os.remove('Results/results_voting_classifier.csv')


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


def recursive_feature_elimination(X_train, y_train, n_features_to_select):
    # model = LogisticRegression(solver='liblinear')
    #model = LinearSVC()
    model = DecisionTreeClassifier(random_state=500)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    return X_train_rfe, rfe


def test(lasso=False, rfe=False):
    if lasso:
        for i in range(12):
            print(i + 1)
            train, test = read_data(i + 1)
            data = read_data_once(i + 1)
            print(data.shape)
            for pca, scal, lasso, minmax, shuffle in itertools.product([True, False], repeat=5):
                if not (scal and minmax) and ((scal or minmax) or not lasso):
                    clf1 = random_forest.random_forest(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                                shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                    clf2 = svm.svm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                            scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)

                    clf3 = naive_bayes.naive_bayes(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                            shuffle=shuffle,
                                            pc=pca, scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                    clf4 = lgbm.lightgbm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle,
                                  pc=pca,
                                  scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                    clf5 = grad_boosting.gradient_boosting(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                                    shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso,
                                                    rfe=rfe)
                    clf6 = knn.knn(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                            scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)

                    vote.voting(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                            scal=scal, minmax=minmax, lasso=lasso, rfe=rfe,
                            clf1=clf1, clf2=clf2, clf3=clf3, clf4=clf4, clf5=clf5, clf6=clf6)
    elif rfe:
        for i in range(4):
            print(i + 1)
            train, test = read_data(i + 1)
            data = read_data_once(i + 1)
            print(data.shape)
            for pca, scal, minmax, shuffle in itertools.product([True, False], repeat=4):
                if not (scal and minmax) and ((scal or minmax) or not rfe):
                    clf5 = grad_boosting.gradient_boosting(config=i + 1, train_data_df=train, test_data_df=test,
                                                           data_df=data,
                                                           shuffle=shuffle, pc=pca, scal=scal, minmax=minmax,
                                                           lasso=False,
                                                           rfe=True)
                    """""
                    clf1 = random_forest.random_forest(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                            shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                clf2 = svm.svm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                        scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)

                clf3 = naive_bayes.naive_bayes(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                        shuffle=shuffle,
                                        pc=pca, scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                clf4 = lgbm.lightgbm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle,
                              pc=pca,
                              scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                clf5 = grad_boosting.gradient_boosting(config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                                shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso,
                                                rfe=rfe)
                clf6 = knn.knn(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                        scal=scal, minmax=minmax, lasso=lasso, rfe=rfe)
                vote.voting(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca,
                            scal=scal, minmax=minmax, lasso=lasso, rfe=rfe,
                            clf1=clf1, clf2=clf2, clf3=clf3, clf4=clf4, clf5=clf5, clf6=clf6)
                            """
    elif rfe and lasso:
        return


