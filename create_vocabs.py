from nltk import ngrams
import dataset


# Create a list of unique words
def create_vocab_with_stopwords():
    train_scenes = dataset.text_tokenized_stopwords('Train_dataset')
    test_scenes = dataset.text_tokenized_stopwords('Test_dataset')
    unique_words = []
    for i in train_scenes.keys():
        for word in train_scenes[i]:
            if word not in unique_words:
                unique_words.append(word)
    for j in test_scenes.keys():
        for word in test_scenes[j]:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words

def create_vocab_n_grams(n):
    train_scenes = dataset.text_tokenized_stopwords('Train_dataset')
    test_scenes = dataset.text_tokenized_stopwords('Test_dataset')
    unique_ngrams = []
    for i in train_scenes.keys():
        for ng in ngrams(train_scenes[i], n):
            word = ' '.join(ng)
            if word not in unique_ngrams:
                unique_ngrams.append(word)

    for j in test_scenes.keys():
        for ng in ngrams(test_scenes[j], n):
            word = ' '.join(ng)
            if word not in unique_ngrams:
                unique_ngrams.append(word)
    return unique_ngrams

def create_vocab_without_stopwords():
    train_scenes = dataset.text_tokenized_no_stopwords('Train_dataset')
    test_scenes = dataset.text_tokenized_no_stopwords('Test_dataset')
    unique_words = []
    for i in train_scenes.keys():
        for word in train_scenes[i]:
            if word not in unique_words:
                unique_words.append(word)
    for j in test_scenes.keys():
        for word in test_scenes[j]:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words

def pos_vocab():
    train_scenes = dataset.text_pos_tokenized_stopwords('Train_dataset')
    test_scenes = dataset.text_pos_tokenized_stopwords('Test_dataset')
    unique_pos = []
    for i in train_scenes.keys():
        for j in train_scenes[i]:
            word, pos = j
            if pos not in unique_pos:
                unique_pos.append(pos)

    for k in test_scenes.keys():
        for a in test_scenes[k]:
            word, pos = a
            if pos not in unique_pos:
                unique_pos.append(pos)
    return unique_pos