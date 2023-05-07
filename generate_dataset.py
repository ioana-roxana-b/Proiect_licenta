import numpy as np
import pandas as pd
import feature_vect


def gen_config1():
    # Generate and save the training data
    train_data = feature_vect.all_scene_feat_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.items():
        labels.append(i[0].split()[0])
        values.append(i[1])

    X_train = np.array(values)
    y_train = np.array(labels)

    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    # Generate and save the test data
    test_data = feature_vect.all_scene_feat_vect('Test_dataset')
    labels_test = []
    values_test = []
    for i in test_data.items():
        labels_test.append(i[0].split()[0])
        values_test.append(i[1])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    # Save all data in a single file
    data_df = pd.concat([train_df, test_df])
    data_df.to_csv('config1.csv', index=False)


def gen_config2():
    # Generate and save the training data
    train_data = feature_vect.tf_idf_scene_feat_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.items():
        labels.append(i[0].split()[0])
        values.append(i[1])

    X_train = np.array(values)
    y_train = np.array(labels)

    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    # Generate and save the test data
    test_data = feature_vect.tf_idf_scene_feat_vect('Test_dataset')
    labels_test = []
    values_test = []
    for i in test_data.items():
        labels_test.append(i[0].split()[0])
        values_test.append(i[1])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    # Save all data in a single file
    data_df = pd.concat([train_df, test_df])
    data_df.to_csv('config2.csv', index=False)
def gen_config3():

    # Generate and save the training data
    train_data = feature_vect.small_scene_feat_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.items():
        labels.append(i[0].split()[0])
        values.append(i[1])

    X_train = np.array(values)
    y_train = np.array(labels)

    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    # Generate and save the test data
    test_data = feature_vect.small_scene_feat_vect('Test_dataset')
    labels_test = []
    values_test = []
    for i in test_data.items():
        labels_test.append(i[0].split()[0])
        values_test.append(i[1])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    # Save all data in a single file
    data_df = pd.concat([train_df, test_df])
    data_df.to_csv('config3.csv', index=False)


def gen_config4():
    # Generate and save the training data
    train_data = feature_vect.all_sentence_features_vect('Train_dataset')
    labels = []
    values = []
    for i in train_data.keys():
        for j in train_data[i]:
            labels.append(i.split()[0])
            values.append(train_data[i][j])

    X_train = np.array(values)
    y_train = np.array(labels)

    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    # Generate and save the test data
    test_data = feature_vect.all_sentence_features_vect('Test_dataset')
    labels_test = []
    values_test = []
    for i in test_data.keys():
        for j in test_data[i]:
            labels_test.append(i.split()[0])
            values_test.append(test_data[i][j])

    X_test = np.array(values_test)
    y_test = np.array(labels_test)

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    # Save all data in a single file
    data_df = pd.concat([train_df, test_df])
    data_df.to_csv('config4.csv', index=False)