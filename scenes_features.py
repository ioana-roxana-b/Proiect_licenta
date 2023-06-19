import string
import dataset
import re

def no_of_words(dir):
    scenes = dataset.text_tokenized_stopwords(dir)
    no_of_w = {}
    for i in scenes.keys():
        no_of_w[i] = len(scenes[i])
    return no_of_w

def no_of_stop_words(dir):
    scenes = dataset.text_tokenized_stopwords(dir)
    scenes_no_sw = dataset.text_tokenized_no_stopwords(dir)
    no_of_sw = {}
    for (i, j) in zip(scenes.keys(), scenes_no_sw.keys()):
        no_of_sw[i] = len(scenes[i]) - len(scenes_no_sw[j])
    return no_of_sw

def no_of_contracted_wordforms(dir):
    scenes = dataset.split_text_into_scenes(dir)
    pattern = r"\b\w+'\w+\b"
    for i in scenes.keys():
        contracted_word_forms = re.findall(pattern, scenes[i])
        num_contracted_word_forms = len(contracted_word_forms)
        scenes[i]=num_contracted_word_forms
    return scenes

def no_of_characters(dir):
    scenes = dataset.split_text_into_scenes(dir)
    no_of_ch = {}
    for i in scenes.keys():
        no_of_ch[i] = len(scenes[i])
    return no_of_ch

def no_of_sentences(dir):
    scenes = dataset.split_scenes_into_phrases(dir)
    no_of_s = {}
    for i in scenes.keys():
        no_of_s[i] = len(scenes[i])
    return no_of_s

def avg_sentence_length(dir):
    phrases = dataset.split_scenes_into_phrases(dir)
    words = dataset.text_tokenized_stopwords(dir)
    for i in phrases.keys():
        avg_sentence_len = len(words[i]) / len(phrases[i])
        phrases[i] = avg_sentence_len
    return phrases

def no_of_punctuation(dir):
    scenes = dataset.split_text_into_scenes(dir)
    no_of_ch = {}
    for i in scenes.keys():
        nr_p = 0
        for j in scenes[i]:
            if j in string.punctuation:
                nr_p +=1
        no_of_ch[i] = nr_p
    return no_of_ch

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
    return words


