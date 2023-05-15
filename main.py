import dataset
import scenes_features
import sentence_features
import svm
import random_forest
import old_rf
import old_svm
import extra_features
import feature_vect
import generate_dataset
import itertools
if __name__ == '__main__':
   for i in range(4):
      print(i + 1)
      for pca, scal, lasso, minmax in itertools.product([True, False], repeat=4):
         if not (scal and minmax) and ((scal or minmax) or not lasso):
            random_forest.random_forest(config=i + 1, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_rf.random_forest(config=i + 1, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            old_svm.svm(config=i + 1, pc=pca, scal=scal, lasso=lasso, minmax=minmax)
            svm.svm(config=i + 1, pc=pca, scal=scal, lasso=lasso, minmax=minmax)



