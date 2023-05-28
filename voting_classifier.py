import itertools

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import svm
import random_forest
import knn
import naive_bayes
import lgbm
import grad_boosting
import additional_functions as adf


def voting(config, train_data_df, test_data_df, data_df, shuffle=False, pc=False,
           scal=False, minmax=False, lasso=False, rfe=False,
           clf1=None, clf2=None, clf3=None, clf4=None, clf5=None, clf6=None):
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', clf1),
            #('svm', clf2),
            ('nb', clf3),
            ('lgbm', clf4),
            ('gb', clf5),
            ('knn', clf6)
        ],
        voting='soft'
    )

    if shuffle:
        X = data_df.drop('label', axis=1).values
        y = data_df['label'].values

        skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    else:
        X_train = train_data_df.drop('label', axis=1).values
        y_train = train_data_df['label'].values

        X_test = test_data_df.drop('label', axis=1).values
        y_test = test_data_df['label'].values

    # Train the VotingClassifier on the combined training data
    voting_clf.fit(X_train, y_train)

    # Make predictions with the VotingClassifier
    y_pred = voting_clf.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    results_df = pd.DataFrame({
        'Configuration': [
            f'config={config},  shuffle={shuffle}, pca={pc}, scal={scal}, minmax={minmax}, lasso={lasso},rfe={rfe}'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    results_df.to_csv('Results/results_voting_classifier.csv', mode='a', index=False)
