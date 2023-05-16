import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
def gradient_boosting(config, train_data_df, test_data_df, pc=False, scal=False, lasso=False, minmax=False):
    X_train = train_data_df.drop('label', axis=1).values
    y_train = train_data_df['label'].values

    X_test = test_data_df.drop('label', axis=1).values
    y_test = test_data_df['label'].values

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
        lass = Lasso(alpha=0.01, max_iter=10000)
        lass.fit(X_train, y_train)
        coef = lass.coef_
        idx_nonzero = np.nonzero(coef)[0]
        X_train = X_train[:, idx_nonzero]
        X_test = X_test[:, idx_nonzero]

    if pc == True:
        X = np.concatenate((X_test, X_train))
        y = np.concatenate((y_train, y_test))

        pca = PCA(n_components=10)
        pca.fit(X)
        new_X = pca.transform(X)
        new_X_test = pca.transform(X_test)

        clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
        clf.fit(new_X, y)
        y_pred = clf.predict(new_X_test)

    elif pc == False:
        clf = GradientBoostingClassifier(n_estimators=500, random_state=50)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    results_df = pd.DataFrame({
        'Configuration': [f'config={config}, pca={pc}, scal={scal}, lasso={lasso}, minmax={minmax}'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    results_df.to_csv('results_gradient_boosting.csv', mode='a', index=False)