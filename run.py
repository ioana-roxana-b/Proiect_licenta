import itertools
import classification
import additional_functions as adf

def test():
    for c in range(7):
        #print(c+1)
        for i in range(1):
            print(i + 1)
            train, test = adf.read_data(i + 1)
            data = adf.read_data_once(i + 1)
            print(data.shape)
            for pca, scal, lasso, minmax, shuffle in itertools.product([True, False], repeat=5):
                if not (scal and minmax) and ((scal or minmax) or not lasso):
                    classification.classification(c=c+1, config=i + 1, train_data_df=train, test_data_df=test, data_df=data,
                                                       shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso,
                                                       rfe=False)
