import string

import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

import dataset
import create_vocabs
import re
import math


#Count the number of words from each scene
def no_of_words(dir):
    scenes = dataset.text_tokenized_stopwords(dir)
    no_of_w = {}
    for i in scenes.keys():
        no_of_w[i] = len(scenes[i])
    #print(no_of_w)

    return no_of_w

#Count the number of stopwords from each scene
def no_of_stop_words(dir):
    scenes = dataset.text_tokenized_stopwords(dir)
    scenes_no_sw = dataset.text_tokenized_no_stopwords(dir)
    no_of_sw = {}
    for (i, j) in zip(scenes.keys(), scenes_no_sw.keys()):
        no_of_sw[i] = len(scenes[i]) - len(scenes_no_sw[j])

    #print(no_of_sw)
    return no_of_sw

#Count the number of contracted wordforms from each scene
def no_of_contracted_wordforms(dir):
    scenes = dataset.split_text_into_scenes(dir)
    pattern = r"\b\w+'\w+\b"
    for i in scenes.keys():
        # Use re.findall() to find all occurrences of the pattern in the text
        contracted_word_forms = re.findall(pattern, scenes[i])
        #print(contracted_word_forms)
        num_contracted_word_forms = len(contracted_word_forms)
        scenes[i]=num_contracted_word_forms
    #print(scenes)
    return scenes

#Count the number of characters from each scene
def no_of_characters(dir):
    scenes = dataset.split_text_into_scenes(dir)
    no_of_ch = {}
    for i in scenes.keys():
        no_of_ch[i] = len(scenes[i])
    #print(no_of_ch)
    return no_of_ch

#Count the number of sentences from each scene
def no_of_sentences(dir):
    scenes = dataset.split_scenes_into_phrases(dir)
    no_of_s = {}
    for i in scenes.keys():
        no_of_s[i] = len(scenes[i])
    #print(no_of_s)
    return no_of_s

#Compute the average sentence lenghth for each scene
def avg_sentence_length(dir):
    phrases = dataset.split_scenes_into_phrases(dir)
    words = dataset.text_tokenized_stopwords(dir)
    for i in phrases.keys():
        avg_sentence_len = len(words[i]) / len(phrases[i])
        phrases[i] = avg_sentence_len
    #print(phrases)
    return phrases

#Count the number of punctuation characters from each scene
def no_of_punctuation(dir):
    scenes = dataset.split_text_into_scenes(dir)
    no_of_ch = {}

    for i in scenes.keys():
        nr_p = 0
        for j in scenes[i]:
            if j in string.punctuation:
                nr_p +=1
        no_of_ch[i] = nr_p
    #print(no_of_ch)
    return no_of_ch

#Returns the number of times a word appears in a scene
def term_frequency(word, scene):
    words = scene.split()
    return words.count(word)

#Returns the logarithmically scaled inverse fraction of the scenes that contain the word
def inverse_document_frequency(word, scenes):
    num_scenes_with_word = sum(1 for scene in scenes.values() if word in scene)

    if num_scenes_with_word == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_word)

def tf_idf_with_stopwords(dir):
    scenes = dataset.lower_case_no_punct(dir)
    tokens = dataset.text_tokenized_stopwords(dir)
    word_set = create_vocabs.create_vocab_with_stopwords()
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(tokens.keys()), len(word_set)))

    for (i,j) in zip(scenes.keys(), range(len(scenes))):
        vec = np.zeros((len(word_set),))
        for word in tokens[i]:
            tf = term_frequency(word, scenes[i])
            idf = inverse_document_frequency(word, scenes)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec
    #print(tf_idf_matrix)
    for (i,j) in zip(scenes.keys(),range(len(scenes))):
        scenes[i] = tf_idf_matrix[j]
    #print(scenes)
    return scenes

