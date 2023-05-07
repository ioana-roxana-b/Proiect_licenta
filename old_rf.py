import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import feature_vect

def random_forest(config, pca_red=False):
    # Read the data from the CSV file
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

    # Split the data into X and y
    X_train = train_data_df.drop('label', axis=1).values
    y_train = train_data_df['label'].values

    X_test = test_data_df.drop('label', axis=1).values
    y_test = test_data_df['label'].values

    """""
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X = np.concatenate((X_test, X_train))
    y = np.concatenate((y_train, y_test))

    pca = PCA(n_components=5)
    new_X = pca.fit_transform(X)
    """
    # Training the model
    clf = RandomForestClassifier(n_estimators=500, random_state=50)
    clf.fit(X_train, y_train)

    #new_X_test = pca.fit_transform(X_test)
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