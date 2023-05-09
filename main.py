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

if __name__ == '__main__':
   #random_forest.random_forest(config=1, pca=True, scal=True, lasso=True, minmax=False)
   old_rf.random_forest(config=3, pca=False, scal=True, lasso=True, minmax=False)
   #old_svm.svm(config=1, pca=True, scal=True, lasso=True, minmax=False)
   #svm.svm(config=1, pca=True, scal=True, lasso=True, minmax=False)



