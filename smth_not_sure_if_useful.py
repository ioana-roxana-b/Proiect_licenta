import dataset


#The number of times each word appears in a scene
def word_freq(dir):
    scenes = dataset.text_tokenized_stopwords(dir)
    word_freq={}
    #print(scenes)
    for i in scenes.keys():
        for j in scenes[i]:
            if j in word_freq:
                word_freq[j] += 1
            else:
                word_freq[j] = 1
        scenes[i] = word_freq
        word_freq = {}

    #print(scenes["Fletcher Prologue"])

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
