import itertools
import classification
import additional_functions as adf


def test():
    for c in range(7):
        print(c + 1)
        for i in range(9):
            print(i + 1)
            train, test = adf.read_data(i + 1)
            data = adf.read_data_once(i + 1)
            print(data.shape)
            for pca, scal, lasso, lasso_t, minmax, shuffle in itertools.product([True, False], repeat=6):
                if not (scal and minmax) and ((scal or minmax) or not lasso) and ((scal or minmax) or not lasso_t):
                    if not (lasso and lasso_t) and not (pca and lasso_t):
                        classification.classification(c=c, config=i + 1, train_data_df=train, test_data_df=test,
                                                      data_df=data,
                                                      shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso,
                                                      lasso_t=lasso_t,
                                                      rfe=False)


def test_rfe():
    train, test = adf.read_data(4)
    data = adf.read_data_once(4)
    print(data.shape)
    classification.classification(c=2, config=4, train_data_df=train, test_data_df=test, data_df=data,
                              shuffle=True, pc=False, scal=True, minmax=False, lasso=False, lasso_t=False,
                              rfe=True)
