#!/usr/bin/python3
import pickle
from os.path import join, abspath, isdir
from os import listdir, makedirs
from operator import contains

import re

from nltk import word_tokenize
from gensim.models import Word2Vec

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Convolution1D, MaxPooling1D

# Files are in the following order (on Alex's computer):
# 0 - medline_ner_test.pkl
# 1 - medline_train.pkl
# 2 - drug_bank_ddi_test.pkl
# 3 - drug_bank_train.pkl
# 4 - drug_bank_ner_test.pkl
# 5 - medline_ddi_test.pkl

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

# function that returns full path of a file


def get_file_full_path(file_name, pickled_files):
    for p_f in pickled_files:
        if contains(p_f, file_name):
            return p_f
    return ""


class DDIClassifier:

    def __init__(self):
        self.path = ""

    def set_path(self, path):
        self.path = path

    """
        Preprocess documents loaded from the path. Each sentence in each document will be preprocessed in the following way.
        In each sentence substitute drugs that are interacting, i.e. ddi = True in pair instances with drug1 and drug2 respectfully.
        All other drugs that are not interacting, i.e. ddi = False, are labeled as drug0. After that we tokenize sentence and remove punctuations
        from each tokened word. Each sentence is mapped into list of tokenized words

        output: List of lists of tokenized words
    """
    def preprocess_data(self):
        if len(self.path) == 0:
            raise ValueError("Path can't be empty")

        with open(self.path, 'rb') as f:
            docs = pickle.load(f)

        all_sentences = []
        for doc in docs:
            for sent in doc.sentences:
                sentence = sent.text

                for pair in sent.pairs:
                    if pair.ddi:
                        # since there is an interaction, we find two entities that are interacting
                        e1 = self.find_entity(sent, pair.e1)
                        e2 = self.find_entity(sent, pair.e2)
                        
                        # replace interacted entities with drug1 and drug2 respectfully
                        sentence = sentence.replace(e1.text,'drug1')
                        sentence = sentence.replace(e2.text,'drug2')

                # replace all drugs that are not interacted with drug0. We are safe here since interacted drugs are alredy substituded,
                # which means we won't substitute interacting drug with drug0. I.e. drug1 or drug2 are not in original entities
                for entity in sent.entities:
                    sentence = sentence.replace(entity.text,'drug0')

                # Tokenize each word and remove punctuations in each word
                sent_words = []
                regex = re.compile('[%s]' % re.escape('.,'))
                for w in word_tokenize(sentence):
                    w_no_p = regex.sub('', w)  # word without , or .
                    if len(w_no_p) > 2:
                        sent_words.append(w_no_p.lower())

                all_sentences.append(sent_words)

        return all_sentences

    # Find entity given pair id
    def find_entity(self, sentence, pair_id):
        for entity in sentence.entities:
            if entity.id == pair_id:
                return entity

    # Return input embedded matrix that will be used in training
    def embed_sentences(self, sentences):
        from multiprocessing import cpu_count
        model = Word2Vec(sentences = sentences, size = 300, sg = 1, window = 3, min_count = 1, iter = 10, workers = cpu_count()-1)
        model.init_sims(replace = True)
        return model.wv.syn0


    def get_model(self, word2vec):
        model = Sequential()
        model.add(Convolution1D(200, 3, activation='tanh',input_shape=(None, word2vec.shape[1])))
        model.add(Dropout(0.5))

        model.add(Convolution1D(200, 4, activation = 'tanh'))
        model.add(Dropout(0.5))

        model.add(Convolution1D(200,5,activation = 'tanh'))
        model.add(Dropout(0.5))

        model.add(MaxPooling1D(pool_size = 5))
        print(model.output_shape)

def main():
    ddi_clf = DDIClassifier()
    ddi_clf.set_path(get_file_full_path('drug_bank_train', pickled_files))
    sentences = ddi_clf.preprocess_data()
    word2vec = ddi_clf.embed_sentences(sentences)
    ddi_clf.get_model(word2vec)

if __name__ == "__main__":
    main()
