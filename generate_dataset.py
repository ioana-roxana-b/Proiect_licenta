import numpy as np
import pandas as pd
import configs as feature_vect

def gen_config(config):
    if config != 4 and config != 14:
        feature_extraction_func_name = f'config{config}'

        # Access the function from the module
        feature_extraction_func = getattr(feature_vect, feature_extraction_func_name)

        # Now use feature_extraction_func as the function
        train_data = feature_extraction_func('Train_dataset')
        labels = []
        values = []
        for i in train_data.items():
            labels.append(i[0].split()[0])
            values.append(i[1])

        X_train = np.array(values)
        y_train = np.array(labels)

        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        train_df.to_csv(f'new_configs/train_config{config}.csv', index=False)

        # Generate and save the test data
        test_data = feature_extraction_func('Test_dataset')
        labels_test = []
        values_test = []
        for i in test_data.items():
            labels_test.append(i[0].split()[0])
            values_test.append(i[1])

        X_test = np.array(values_test)
        y_test = np.array(labels_test)

        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test
        test_df.to_csv(f'new_configs/test_config{config}.csv', index=False)

        # Save all data in a single file
        data_df = pd.concat([train_df, test_df])
        data_df.to_csv(f'new_configs/config{config}.csv', index=False)
    elif config == 4:
        # Generate and save the training data
        train_data = feature_vect.config4('Train_dataset')
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
        train_df.to_csv(f'new_configs/train_config{config}.csv', index=False)

        # Generate and save the test data
        test_data = feature_vect.config4('Test_dataset')
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
        test_df.to_csv(f'new_configs/test_config{config}.csv', index=False)

        # Save all data in a single file
        data_df = pd.concat([train_df, test_df])
        data_df.to_csv(f'new_configs/config{config}.csv', index=False)
    elif config == 14:
        new_c = 14
        for k in [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            train_data = feature_vect.config14('Train_dataset', k)
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
            train_df.to_csv(f'new_configs/train_config{new_c}.csv', index=False)


            # Generate and save the test data
            test_data = feature_vect.config14('Test_dataset', k)
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
            test_df.to_csv(f'new_configs/test_config{new_c}.csv', index=False)

            # Save all data in a single file
            data_df = pd.concat([train_df, test_df])
            data_df.to_csv(f'new_configs/config{new_c}.csv', index=False)
            new_c += 1



