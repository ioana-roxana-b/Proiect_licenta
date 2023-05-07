import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import feature_vect


def svm():
    train_data = feature_vect.small_scene_feat_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.items():
        labels.append(i[0].split()[0])
        values.append(i[1])

    X_train = np.array(values)
    y_train = np.array(labels)

    test_data = feature_vect.small_scene_feat_vect('Test_dataset')

    labels_test = []
    values_test = []
    for i in test_data.items():
        labels_test.append(i[0].split()[0])
        values_test.append(i[1])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    X = np.concatenate((X_test, X_train))
    y = np.concatenate((y_train, y_test))

    pca = PCA(n_components=10)
    new_X = pca.fit_transform(X)

    tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
    clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
    clf.fit(new_X, y)

    new_X_test = pca.fit_transform(X_test)
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
    print("F1 Score: ", f1, "\n")