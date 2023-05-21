import string

import nltk
import pandas as pd

import create_vocabs
import dataset
import math
import scenes_features
import numpy as np

from nltk import ngrams, pos_tag, word_tokenize
from nltk.corpus import stopwords

# Returns the number of times a word appears in a scene
def n_grams_tf(ngram, scene):
    words = list(ngrams(scene.split(), len(ngram.split())))
    words_str = [' '.join(word) for word in words]
    return words_str.count(ngram)


# Returns the logarithmically scaled inverse fraction of the scenes that contain the ngram
def n_grams_idf(ngram, scenes):
    num_scenes_with_ngram = sum(1 for scene in scenes.values() if ngram in scene)
    if num_scenes_with_ngram == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_ngram)


def n_grams_tf_idf(dir, n):
    scenes = dataset.lower_case_no_punct(dir)
    tokens = dataset.text_tokenized_stopwords(dir)
    ngrams_set = create_vocabs.create_vocab_n_grams(n)
    word_index = {}
    for i, word in enumerate(ngrams_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(tokens.keys()), len(ngrams_set)))
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        vec = np.zeros((len(ngrams_set),))
        for ng in ngrams(tokens[i], n):
            word = ' '.join(ng)
            tf = n_grams_tf(word, scenes[i])
            idf = n_grams_idf(word, scenes)

            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec
    # print(tf_idf_matrix)
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]
    # print(scenes)
    return scenes


def tf_idf_without_stopwords(dir):
    scenes = dataset.lower_case_no_punct(dir)
    tokens = dataset.text_tokenized_no_stopwords(dir)
    word_set = create_vocabs.create_vocab_without_stopwords()
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(tokens.keys()), len(word_set)))
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        vec = np.zeros((len(word_set),))
        for word in tokens[i]:
            tf = scenes_features.term_frequency(word, scenes[i])
            idf = scenes_features.inverse_document_frequency(word, scenes)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec
    # print(tf_idf_matrix)
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]
    # print(scenes)
    return scenes

def punct_tf(char, scene):
    return scene.count(char)

def punct_idf(char, scenes):
    num_scenes_with_char = sum(1 for scene in scenes.values() if char in scene)

    if num_scenes_with_char == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_char)
def punc_tf_idf(dir):
    scenes = dataset.lower_case_with_punct(dir)
    word_set = string.punctuation
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(scenes.keys()), len(word_set)))
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        vec = np.zeros((len(word_set),))
        for word in word_set:
            tf = punct_tf(word, scenes[i])
            idf = punct_idf(word, scenes)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec
        # print(tf_idf_matrix)
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]
        # print(scenes)
    return scenes

def tf_idf_for_stopwords(dir):
    scenes = dataset.lower_case_with_punct(dir)
    word_set = stopwords.words('english')
    #print(word_set)
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(scenes.keys()), len(word_set)))
    for (i,j) in zip(scenes.keys(), range(len(scenes))):
        vec = np.zeros((len(word_set),))
        for word in word_set:
            tf = scenes_features.term_frequency(word, scenes[i])
            idf = scenes_features.inverse_document_frequency(word, scenes)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec
    # print(tf_idf_matrix)
    for (i, j) in zip(scenes.keys(), range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]
    # print(scenes)
    return scenes

# Returns the number of times a word appears in a scene
def pos_tf(pos, scene):
    count = 0
    for j in scene:
        word, tag =j
        if tag==pos:
            count +=1
    return count

# Returns the logarithmically scaled inverse fraction of the scenes that contain the word
def pos_idf(pos, scenes):
    num_scenes_with_pos = 0
    for i in scenes.keys():
        for j in scenes[i]:
            word, tag = j
            if tag == pos:
                num_scenes_with_pos +=1

    if num_scenes_with_pos == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_pos)

def pos_tf_idf(dir):
    scenes = dataset.lower_case_no_punct(dir)
    tokens = dataset.text_pos_tokenized_stopwords(dir)
    pos_set = create_vocabs.pos_vocab()
    pos_index = {}
    for i, pos in enumerate(pos_set):
        pos_index[pos] = i

    tf_idf_matrix = np.zeros((len(tokens.keys()), len(pos_set)))
    for (i,j) in zip(tokens.keys(), range(len(tokens))):
        vec = np.zeros((len(pos_set),))
        for z in tokens[i]:
            word,pos = z
            tf = pos_tf(pos, tokens[i])
            idf = pos_idf(pos, tokens)
            vec[pos_index[pos]] = tf * idf
        tf_idf_matrix[j] = vec
    #print(tf_idf_matrix)
    for (i,j) in zip(scenes.keys(),range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]

    return scenes

#Compute the average word lenghth for each scene
def avg_word_length(dir):
    scenes = dataset.delete_punctuation(dir)
    no_of_ch = {}
    for i in scenes.keys():
        len(scenes[i])
        no_of_ch[i] = len(scenes[i])

    words = dataset.text_tokenized_stopwords(dir)
    for i in words.keys():
        avg_word_len = len(words[i])/(no_of_ch[i])
        words[i]=avg_word_len
    #print(words)
    return words