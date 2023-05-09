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
   #random_forest.random_forest(config=4, pca=False, scal=False, lasso=True, minmax=True)
   old_rf.random_forest(config=4, pca=False, scal=False, lasso=True, minmax=True)
   #old_svm.svm(config=1, pca=False, scal=False, lasso=True, minmax=True)
   #svm.svm(config=4, pca=False, scal=False, lasso=True, minmax=True)



