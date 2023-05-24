import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import additional_functions as adf

def gradient_boosting(config, train_data_df, test_data_df, data_df, shuffle=False, pc=False,
                      scal=False, minmax=False, lasso=False, rfe=False):
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

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    if minmax == True:
        X_train, X_test = adf.minmax_sc(X_train, X_test)

    if scal == True:
        X_train, X_test = adf.stand_sc(X_train, X_test)

    if lasso == True:
        X_train, X_test = adf.lasso(X_train, X_test, y_train)

    if rfe:  # Add a flag for RFE in the function parameters
        X_train, rfe_selector = adf.recursive_feature_elimination(X_train, y_train, 45)

        X_test = rfe_selector.transform(X_test)

    if pc == True and config != 9 and config != 18:
        if shuffle:
            new_X, new_X_test = adf.pca(X_train, X_test)
            clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
            clf.fit(new_X, y_train)
            y_pred = clf.predict(new_X_test)
        else:
            if config != 4:
                X = np.concatenate((X_test, X_train))
                y = np.concatenate((y_train, y_test))
                new_X, new_X_test = adf.pca(X, X_test)
                clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
                clf.fit(new_X, y)
                y_pred = clf.predict(new_X_test)
            else:
                new_X, new_X_test = adf.pca(X_train, X_test)
                clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
                clf.fit(new_X, y_train)
                y_pred = clf.predict(new_X_test)

    elif (pc == True and (config == 9 or config == 18)) or (pc == False):
        clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    results_df = pd.DataFrame({
        'Configuration': [
            f'config={config},  shuffle={shuffle}, pca={pc}, scal={scal}, minmax={minmax}, lasso={lasso}'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    results_df.to_csv('Results/results_gradient_boosting.csv', mode='a', index=False)
