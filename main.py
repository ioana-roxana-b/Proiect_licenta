import dataset
import scenes_features
import sentence_features
import svm
import random_forest
import extra_features
import feature_vect
import generate_dataset

if __name__ == '__main__':
   #dataset.split_scenes_into_phrases('Test_dataset')

   #generate_dataset.gen_config41()
   random_forest.random_forest(config=1)
   #svm.svm(config=1)



