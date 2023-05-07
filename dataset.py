import os
import re
import nltk
import string
from nltk.corpus import stopwords


def read_text(dir_path):
    files = []
    aut_dict = {}
    for file in os.listdir(dir_path):
        f = os.path.join(dir_path, file)
        with open(f, 'r') as file:
            text = file.read()

    # Delete the names of the characters
    text = re.sub(r'(^[A-Z]+\d*:)|(^[A-Z_]+:)', '', text, flags=re.MULTILINE)
    # print(text)
    return text


# Split text in scenes
def split_text_into_scenes(dir):
    text = read_text(dir)
    sc = {}
    limits = r"(Shakespeare|Fletcher)+\s*(Prologue|Epilogue|ACT|Act\s+\w+\s+SCENE|Scene\s\d)"
    lines = text.splitlines()
    list_of_acts = [l for l in lines if re.match(limits, l)]

    # print(list_of_acts)
    lim = '|'.join(map(re.escape, list_of_acts))
    scenes = re.split(lim, text)[1:]

    for (i, (j, scene)) in zip(range(len(list_of_acts)), enumerate(scenes)):
        sc[list_of_acts[i]] = scene.strip()

    # print(sc["Shakespeare ACT V. Scene 4."])
    return sc


def split_scenes_into_phrases(dir):
    scenes = split_text_into_scenes(dir)
    for i in scenes.keys():
        # Split text into phrases
        phrases = re.split(r'[.!?]+', scenes[i])
        phrases = [phrase.strip() for phrase in phrases]
        scenes[i] = phrases
    #print(len(scenes[i]))
    # print(scenes["Shakespeare ACT V. Scene 4."])
    return scenes


def delete_punctuation(dir):
    scenes = split_text_into_scenes(dir)
    for i in scenes.keys():
        scenes[i] = scenes[i].translate(str.maketrans('', '', string.punctuation))
        scenes[i] = scenes[i].replace('\n\n', ' ')
    for i in scenes.keys():
        scenes[i] = re.sub(r'\d+', '', scenes[i])
    # print(scenes["Shakespeare ACT V. Scene 4."])
    return scenes


def lower_case_no_punct(dir):
    scenes = delete_punctuation(dir)
    for i in scenes.keys():
        scenes[i] = str.lower(scenes[i])
    return scenes
def lower_case_with_punct(dir):
    scenes = split_text_into_scenes(dir)
    for i in scenes.keys():
        scenes[i] = str.lower(scenes[i])
    return scenes


# Tokenize text including stopwords
def text_tokenized_stopwords(dir):
    scenes = lower_case_no_punct(dir)
    for i in scenes.keys():
        scenes[i] = nltk.word_tokenize(scenes[i])
    # print(scenes["Shakespeare ACT V. Scene 4."])
    return scenes


# Tokenize text without stopwords
def text_tokenized_no_stopwords(dir):
    scenes = text_tokenized_stopwords(dir)
    nltk_stopw = stopwords.words('english')
    ia_stopw = [line.strip() for line in open('stop_words')]
    for i in scenes.keys():
        for j in scenes[i]:
            if j in ia_stopw:
                scenes[i].remove(j)
    # print(scenes["Shakespeare ACT V. Scene 4."])
    return scenes


def text_tokenized_including_punctuation(dir):
    scenes = split_text_into_scenes(dir)
    for i in scenes.keys():
        scenes[i] = str.lower(scenes[i])
        scenes[i] = nltk.word_tokenize(scenes[i])
    # print(scenes["Shakespeare ACT V. Scene 4."])
    return scenes


def text_pos_tokenized_stopwords(dir):
    scenes = text_tokenized_stopwords(dir)
    for i in scenes.keys():
        scenes[i] = nltk.pos_tag(scenes[i])
    #print(scenes)
    return scenes
