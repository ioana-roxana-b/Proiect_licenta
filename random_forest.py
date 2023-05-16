import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def random_forest(config, data_df, pc=False, scal=False, lasso=False, minmax=False):

    # Split the data into X and y
    X = data_df.drop('label', axis=1).values
    y = data_df['label'].values

    # Perform cross-validation
    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    if minmax == True:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if scal == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if lasso == True:
        las = Lasso(alpha=0.01, max_iter=10000)
        las.fit(X_train, y_train)
        coef = las.coef_
        idx_nonzero = np.nonzero(coef)[0]
        X_train = X_train[:, idx_nonzero]
        X_test = X_test[:, idx_nonzero]
        #print(len(X_train))
        #print(len(X_test))

    if pc == True:
        pca = PCA(n_components=10)
        new_X_train = pca.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=500, random_state=50)
        clf.fit(new_X_train, y)

        new_X_test = pca.fit_transform(X_test)
        y_pred = clf.predict(new_X_test)
        y_pred = le.transform(y_pred)

    else:
        clf = RandomForestClassifier(n_estimators=500, random_state=50)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)


    #print(y_test)
    #print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    """""
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    """

    results_df = pd.DataFrame({
        'Configuration': [f'config={config}, pca={pc}, scal={scal}, lasso={lasso}, minmax={minmax}'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
        })
    results_df.to_csv('results_random_forest.csv', mode='a', index=False)

