import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split


def random_forest(config, pca_red=False, scal=False):
    # Read the data from the CSV file
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

    # Perform cross-validation
    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    if scal == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if pca_red == True:
        pca = PCA(n_components=10)
        new_X_train = pca.fit_transform(X)

        # Training the model
        clf = RandomForestClassifier(n_estimators=500, random_state=50)
        clf.fit(new_X_train, y)

        new_X_test = pca.fit_transform(X_test)
        # Evaluating the model
        y_pred = clf.predict(new_X_test)

    else:
        # Training the model
        clf = RandomForestClassifier(n_estimators=500, random_state=50)
        clf.fit(X_train, y_train)

        # Evaluating the model
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
    print("F1 Score: ", f1)
