import dataset
import scenes_features
import sentence_features
import svm
import random_forest
import extra_features
import feature_vect
import rf_sentence
import svm_sentence

if __name__ == '__main__':
   random_forest.random_forest()
   #svm.svm()
   #rf_sentence.random_forest()
   #feature_vect.tf_idf_scene_feat_vect('Train_dataset')
   #feature_vect.small_scene_feat_vect('Train_dataset')
   #svm_sentence.svm()


