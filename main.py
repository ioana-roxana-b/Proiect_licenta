import os
import pandas as pd
import svm
import random_forest
import old_rf
import old_svm
import grad_boosting
import grad_boosting_v2
import itertools
import knn_v1
import knn_v2
import nb_v1
import nb_v2
import lightgbm1
import lightgbm2
def read_data(config):
   if config == 1:
      train_data_df = pd.read_csv('configs/train_config1.csv')
      test_data_df = pd.read_csv('configs/test_config1.csv')
   elif config == 2:
      train_data_df = pd.read_csv('configs/train_config2.csv')
      test_data_df = pd.read_csv('configs/test_config2.csv')
   elif config == 3:
      train_data_df = pd.read_csv('configs/train_config3.csv')
      test_data_df = pd.read_csv('configs/test_config3.csv')
   elif config == 4:
      train_data_df = pd.read_csv('configs/train_config4.csv')
      test_data_df = pd.read_csv('configs/test_config4.csv')

   return train_data_df, test_data_df


def read_data_once(config):
   if config == 1:
      data_df = pd.read_csv('configs/config1.csv')
   elif config == 2:
      data_df = pd.read_csv('configs/config2.csv')
   elif config == 3:
      data_df = pd.read_csv('configs/config3.csv')
   elif config == 4:
      data_df = pd.read_csv('configs/config4.csv')
   return data_df

def delete_files():
   if os.path.exists('Results/results_svm.csv'):
      os.remove('Results/results_svm.csv')
   if os.path.exists('Results/results_old_svm.csv'):
      os.remove('Results/results_old_svm.csv')
   if os.path.exists('Results/results_old_random_forest.csv'):
      os.remove('Results/results_old_random_forest.csv')
   if os.path.exists('Results/results_random_forest.csv'):
      os.remove('Results/results_random_forest.csv')
   if os.path.exists('Results/results_gradient_boosting.csv'):
      os.remove('Results/results_gradient_boosting.csv')
   if os.path.exists('Results/results_gradient_boosting_v2.csv'):
      os.remove('Results/results_gradient_boosting_v2.csv')
   if os.path.exists('Results/results_knn1.csv'):
      os.remove('Results/results_knn1.csv')
   if os.path.exists('Results/results_knn2.csv'):
      os.remove('Results/results_knn2.csv')
   if os.path.exists('Results/results_naive_bayes1.csv'):
      os.remove('Results/results_naive_bayes1.csv')
   if os.path.exists('Results/results_naive_bayes2.csv'):
      os.remove('Results/results_naive_bayes2.csv')

if __name__ == '__main__':

   delete_files()
   for i in range(4):
      print(i+1)
      train, test = read_data(i+1)
      data = read_data_once(i+1)
      for pca, scal, lasso, minmax in itertools.product([True, False], repeat=4):
         if not (scal and minmax) and ((scal or minmax) or not lasso):
            """""
            random_forest.random_forest(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_rf.random_forest(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_svm.svm(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            svm.svm(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            grad_boosting.gradient_boosting(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            grad_boosting_v2.gradient_boosting(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            knn_v1.knn(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            knn_v2.knn(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            nb_v1.naive_bayes(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            nb_v2.naive_bayes(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            """
            lightgbm1.lightgbm(config=i+1, data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            lightgbm2.lightgbm(config=i+1, train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)