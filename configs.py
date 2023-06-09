import sys
import scenes_features
import extra_features
import sentence_features

#All scene features
def config1(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = scenes_features.scene_avg_word_length(dir)
    vect5 = extra_features.tf_idf_for_stopwords(dir)
    vect6 = extra_features.tf_idf_with_stopwords(dir)
    vect7 = extra_features.tf_idf_without_stopwords(dir)
    vect8 = extra_features.n_grams_tf_idf(dir, 2)
    vect9 = extra_features.pos_tf_idf(dir)
    vect10 = extra_features.punc_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        concat.append(vect4[i])
        for a in vect5[i]:
            concat.append(a)
        for b in vect6[i]:
            concat.append(b)
        for c in vect7[i]:
            concat.append(c)
        for d in vect8[i]:
            concat.append(d)
        for e in vect9[i]:
            concat.append(e)
        for f in vect10[i]:
            concat.append(f)
        all_feat[i] = concat

        concat = []
    return all_feat

#All tf-idf for scenes
def config2(dir):
    vect5 = extra_features.tf_idf_for_stopwords(dir)
    vect6 = extra_features.tf_idf_with_stopwords(dir)
    vect7 = extra_features.tf_idf_without_stopwords(dir)
    vect8 = extra_features.n_grams_tf_idf(dir, 2)
    vect9 = extra_features.pos_tf_idf(dir)
    vect10 = extra_features.punc_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect5.keys():
        for a in vect5[i]:
            concat.append(a)
        for b in vect6[i]:
            concat.append(b)
        for c in vect7[i]:
            concat.append(c)
        for d in vect8[i]:
            concat.append(d)
        for e in vect9[i]:
            concat.append(e)
        for f in vect10[i]:
            concat.append(f)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat


#All non tf-idf features for scenes + tf-idf for stopwords
def config3(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = scenes_features.scene_avg_word_length(dir)
    vect5 = extra_features.tf_idf_for_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        concat.append(vect4[i])
        for c in vect5[i]:
            concat.append(c)
        all_feat[i] = concat
        #print(len(concat))
        concat = []

    return all_feat

#All sentence features + all scenes features
def config4(dir):
    vect11 = sentence_features.sentence_length_by_word(dir)
    vect12 = sentence_features.sentence_length_by_characters(dir)
    vect13 = sentence_features.avg_word_length(dir)
    vect14 = sentence_features.stopwords_count(dir)

    all_feat_vect = config1(dir)
    sentence_features_vect = {}
    aux1 = []
    for i in vect11.keys():
        sentence_features_vect.setdefault(i, {})
        for j in vect11[i].keys():
            if j == '':
                continue
            aux1.append(vect11[i][j])
            aux1.append(vect12[i][j])
            aux1.append(vect13[i][j])
            aux1.append(vect14[i][j])
            aux1 += all_feat_vect[i]
            #print(len(aux1))
            sentence_features_vect[i][j] = aux1
            aux1 = []

    return sentence_features_vect

#TF-IDF for words
def config5(dir):
    vect5 = extra_features.tf_idf_for_stopwords(dir)
    vect6 = extra_features.tf_idf_with_stopwords(dir)
    vect7 = extra_features.tf_idf_without_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect5.keys():
        for a in vect5[i]:
            concat.append(a)
        for b in vect6[i]:
            concat.append(b)
        for c in vect7[i]:
            concat.append(c)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config6(dir):
    vect8 = extra_features.n_grams_tf_idf(dir, 2)
    vect9 = extra_features.pos_tf_idf(dir)
    vect10 = extra_features.punc_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect8.keys():
        for d in vect8[i]:
            concat.append(d)
        for e in vect9[i]:
            concat.append(e)
        for f in vect10[i]:
            concat.append(f)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat


def config7(dir):
    vect1 = scenes_features.no_of_words(dir)
    vect2 = scenes_features.no_of_stop_words(dir)
    vect3 = scenes_features.no_of_contracted_wordforms(dir)
    vect4 = scenes_features.scene_avg_word_length(dir)
    vect5 = extra_features.tf_idf_for_stopwords(dir)
    vect9 = extra_features.pos_tf_idf(dir)

    all_feat = {}
    concat = []
    for i in vect1.keys():
        concat.append(vect1[i])
        concat.append(vect2[i])
        concat.append(vect3[i])
        concat.append(vect4[i])
        for c in vect5[i]:
            concat.append(c)
        for d in vect9[i]:
            concat.append(d)
        all_feat[i] = concat
        #print(len(concat))
        concat = []

    return all_feat

def config8(dir):
    vect8 = extra_features.n_grams_tf_idf(dir,2)
    all_feat = {}
    concat = []
    for i in vect8.keys():
        for d in vect8[i]:
            concat.append(d)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config9(dir):
    vect9 = extra_features.pos_tf_idf(dir)
    all_feat = {}
    concat = []
    for i in vect9.keys():
        for d in vect9[i]:
            concat.append(d)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config10(dir):
    vect6 = extra_features.tf_idf_with_stopwords(dir)
    all_feat = {}
    concat = []
    for i in vect6.keys():
        for a in vect6[i]:
            concat.append(a)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config11(dir):
    vect7 = extra_features.tf_idf_without_stopwords(dir)

    all_feat = {}
    concat = []
    for i in vect7.keys():
        for a in vect7[i]:
            concat.append(a)
        all_feat[i] = concat
        concat = []
    return all_feat


def config12(dir):
    vect5 = extra_features.tf_idf_for_stopwords(dir)
    all_feat = {}
    concat = []
    for i in vect5.keys():
        for a in vect5[i]:
            concat.append(a)
        all_feat[i] = concat
        #print(len(concat))
        concat = []
    return all_feat

def config13(dir):
    vect8 = extra_features.n_grams_tf_idf(dir,3)
    all_feat = {}
    concat = []
    for i in vect8.keys():
        for d in vect8[i]:
            concat.append(d)
        all_feat[i] = concat
        # print(len(concat))
        concat = []
    return all_feat

def config14(dir, config):
    vect11 = sentence_features.sentence_length_by_word(dir)
    vect12 = sentence_features.sentence_length_by_characters(dir)
    vect13 = sentence_features.avg_word_length(dir)
    vect14 = sentence_features.stopwords_count(dir)

    config_func = getattr(sys.modules[__name__], f'config{config}')
    all_feat_vect = config_func(dir)

    sentence_features_vect = {}
    aux1 = []
    for i in vect11.keys():
        sentence_features_vect.setdefault(i, {})
        for j in vect11[i].keys():
            if j == '':
                continue
            aux1.append(vect11[i][j])
            aux1.append(vect12[i][j])
            aux1.append(vect13[i][j])
            aux1.append(vect14[i][j])
            aux1 += all_feat_vect[i]
            # print(len(aux1))
            sentence_features_vect[i][j] = aux1
            aux1 = []

    return sentence_features_vect
