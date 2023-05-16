import pandas as pd

import svm
import random_forest
import old_rf
import old_svm

import itertools

def read_data(config):
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

   return train_data_df, test_data_df


def read_data_once(config):
   if config == 1:
      data_df = pd.read_csv('config1.csv')
   elif config == 2:
      data_df = pd.read_csv('config2.csv')
   elif config == 3:
      data_df = pd.read_csv('config3.csv')
   elif config == 4:
      data_df = pd.read_csv('config4.csv')
   return data_df


if __name__ == '__main__':

   for i in range(4):
      print(i+1)
      train, test = read_data(i+1)
      data = read_data_once(i+1)
      for pca, scal, lasso, minmax in itertools.product([True, False], repeat=4):
         if not (scal and minmax) and ((scal or minmax) or not lasso):
            random_forest.random_forest(config=i+1,data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_rf.random_forest(config=i+1,train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_svm.svm(config=i+1,train_data_df=train, test_data_df=test, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            svm.svm(config=i+1,data_df=data, pc=pca, scal=scal, lasso=lasso, minmax=minmax)