import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import feature_vect

def random_forest():
    train_data = feature_vect.all_sentence_features_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.items():
        labels.append(i[0].split()[0])
        values.append(i[1])

    X_train = np.array(values)
    y_train = np.array(labels)

    test_data = feature_vect.all_sentence_features_vect('Test_dataset')

    labels_test = []
    values_test = []
    for i in test_data.items():
        labels_test.append(i[0].split()[0])
        values_test.append(i[1])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    # Feature scaling
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    X = np.concatenate((X_test, X_train))
    y = np.concatenate((y_train, y_test))

    pca = PCA(n_components=5)
    new_X = pca.fit_transform(X)

    # Training the model
    clf = RandomForestClassifier(n_estimators=500, random_state=50)
    clf.fit(new_X, y)

    new_X_test = pca.fit_transform(X_test)
    # Evaluating the model
    y_pred = clf.predict(new_X_test)

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