import dataset
from nltk.corpus import stopwords

def sentence_length_by_characters(dir):
    sentences = dataset.split_scenes_into_phrases(dir)
    lengths = {}
    for i in sentences.keys():
        for j in sentences[i]:
            lengths[j]=len(j)
        sentences[i] = lengths
        lengths = {}

    return sentences

def sentence_length_by_word(dir):
    sentences = dataset.split_scenes_into_phrases(dir)
    lengths = {}

    for i in sentences.keys():
        for j in sentences[i]:
            words = j.split()
            lengths[j] = len(words)
        sentences[i] = lengths
        lengths = {}
    return sentences

def avg_word_length(dir):
    sentences = dataset.split_scenes_into_phrases(dir)
    lengths = {}

    for i in sentences.keys():
        for j in sentences[i]:
            words = j.split()
            if len(words):
                avg = sum(len(word) for word in words) / len(words)
                lengths[j] = avg
        sentences[i] = lengths
        lengths = {}
    return sentences

def stopwords_count(dir):
    stop_words = set(stopwords.words('english'))
    sentences = dataset.split_scenes_into_phrases(dir)

    stopword_count = {}
    for i in sentences.keys():
        for j in sentences[i]:
            sentence = str.lower(j)
            words = sentence.split()
            stopword_count[j] = sum([1 for word in words if word in stop_words])
        sentences[i] = stopword_count
        stopword_count = {}
    return sentences

