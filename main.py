import svm
import random_forest
import knn
import naive_bayes
import lgbm
import grad_boosting
import itertools
import additional_functions as adf
import generate_dataset

if __name__ == '__main__':
   adf.delete_files()
   for i in range(4):
      print(i+1)
      train, test = adf.read_data(i+1)
      data = adf.read_data_once(i+1)
      #print(data.shape)
      for pca, scal, lasso, minmax, shuffle in itertools.product([True, False], repeat=5):
         if not (scal and minmax) and ((scal or minmax) or not lasso):
            svm.svm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)
            random_forest.random_forest(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)
            naive_bayes.naive_bayes(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)
            lgbm.lightgbm(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)
            grad_boosting.gradient_boosting(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)
            knn.knn(config=i + 1, train_data_df=train, test_data_df=test, data_df=data, shuffle=shuffle, pc=pca, scal=scal, minmax=minmax, lasso=lasso)