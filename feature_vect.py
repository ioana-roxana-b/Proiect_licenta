import scenes_features
import extra_features
import sentence_features

def config1(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = scenes_features.tf_idf_with_stopwords(dir)
    vect5 = extra_features.tf_idf_without_stopwords(dir)
    vect6 = extra_features.tf_idf_for_stopwords(dir)
    vect7 = extra_features.n_grams_tf_idf(dir,2)
    vect8 = extra_features.pos_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        for a in vect4[i]:
            concat.append(a)
        for b in vect5[i]:
            concat.append(b)
        for c in vect6[i]:
            concat.append(c)
        for d in vect7[i]:
            concat.append(d)
        for e in vect8[i]:
            concat.append(e)
        all_feat[i] = concat

        concat = []
    return all_feat

def config2(dir):
    vect4 = scenes_features.tf_idf_with_stopwords(dir)
    vect5 = extra_features.tf_idf_without_stopwords(dir)
    vect6 = extra_features.tf_idf_for_stopwords(dir)
    vect7 = extra_features.n_grams_tf_idf(dir,2)
    vect8 = extra_features.pos_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect4.keys():
        for a in vect4[i]:
            concat.append(a)
        for b in vect5[i]:
            concat.append(b)
        for c in vect6[i]:
            concat.append(c)
        for d in vect7[i]:
            concat.append(d)
        for e in vect8[i]:
            concat.append(e)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat
def config3(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        for c in vect4[i]:
            concat.append(c)
        all_feat[i] = concat
        #print(len(concat))
        concat = []

    return all_feat



def config4(dir):
    sentence_length_by_word = sentence_features.sentence_length_by_word(dir)
    sentence_length_by_characters = sentence_features.sentence_length_by_characters(dir)
    avg_word_length = sentence_features.avg_word_length(dir)
    stopwords_count = sentence_features.stopwords_count(dir)

    all_feat_vect = config1(dir)
    sentence_features_vect = {}
    aux1 = []
    for i in sentence_length_by_characters.keys():
        sentence_features_vect.setdefault(i, {})
        for j in sentence_length_by_characters[i].keys():
            if j == '':
                continue
            aux1.append(sentence_length_by_characters[i][j])
            aux1.append(sentence_length_by_word[i][j])
            aux1.append(avg_word_length[i][j])
            aux1.append(stopwords_count[i][j])
            aux1 += all_feat_vect[i]
            #print(len(aux1))
            sentence_features_vect[i][j] = aux1
            aux1 = []

    return sentence_features_vect

def config5(dir):
    vect4 = scenes_features.tf_idf_with_stopwords(dir)
    vect5 = extra_features.tf_idf_without_stopwords(dir)
    vect6 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect4.keys():
        for a in vect4[i]:
            concat.append(a)
        for b in vect5[i]:
            concat.append(b)
        for c in vect6[i]:
            concat.append(c)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config6(dir):
    vect7 = extra_features.n_grams_tf_idf(dir,2)
    vect8 = extra_features.pos_tf_idf(dir)
    all_feat = {}
    concat = []
    for i in vect7.keys():
        for d in vect7[i]:
            concat.append(d)
        for e in vect8[i]:
            concat.append(e)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config7(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = extra_features.tf_idf_for_stopwords(dir)
    vect5 = extra_features.pos_tf_idf(dir)
    vect6 = extra_features.avg_word_length(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        concat.append(vect6[i])
        for c in vect4[i]:
            concat.append(c)
        for d in vect5[i]:
            concat.append(d)
        all_feat[i] = concat
        #print(len(concat))
        concat = []

    return all_feat

def config8(dir):
    vect7 = extra_features.n_grams_tf_idf(dir,2)
    #vect8 = extra_features.pos_tf_idf(dir)
    all_feat = {}
    concat = []
    for i in vect7.keys():
        for d in vect7[i]:
            concat.append(d)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config9(dir):
    vect8 = extra_features.pos_tf_idf(dir)
    all_feat = {}
    concat = []
    for i in vect8.keys():
        for d in vect8[i]:
            concat.append(d)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config10(dir):
    vect4 = scenes_features.tf_idf_with_stopwords(dir)
    #vect5 = extra_features.tf_idf_without_stopwords(dir)
    #vect6 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect4.keys():
        for a in vect4[i]:
            concat.append(a)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config11(dir):
    #vect4 = scenes_features.tf_idf_with_stopwords(dir)
    vect5 = extra_features.tf_idf_without_stopwords(dir)
    #vect6 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect5.keys():
        for a in vect5[i]:
            concat.append(a)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config12(dir):
    #vect4 = scenes_features.tf_idf_with_stopwords(dir)
    #vect5 = extra_features.tf_idf_without_stopwords(dir)
    vect6 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect6.keys():
        for a in vect6[i]:
            concat.append(a)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat