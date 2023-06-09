import string
import numpy as np
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

#Compute the average word lenghth for each scene
def scene_avg_word_length(dir):
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


