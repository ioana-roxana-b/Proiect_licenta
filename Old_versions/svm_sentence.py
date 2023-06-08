import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import feature_vect


def svm():
    train_data = feature_vect.all_sentence_features_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.keys():
        for j in train_data[i]:
            labels.append(i.split()[0])
            values.append(train_data[i][j])

    X_train = np.array(values)
    y_train = np.array(labels)

    test_data = feature_vect.all_sentence_features_vect('Test_dataset')

    labels_test = []
    values_test = []
    for i in test_data.keys():
        for j in test_data[i]:
            labels_test.append(i.split()[0])
            values_test.append(test_data[i][j])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)
    #pca = PCA(n_components=10)
    #new_X_train = pca.fit_transform(X_train)


    tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
    clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
    clf.fit(X_train, y_train)

    #new_X_test = pca.fit_transform(X_test)
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
    print("F1 Score: ", f1, "\n")