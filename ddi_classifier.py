#!/usr/bin/python3
import pickle
from os.path import join, abspath, isdir
from os import listdir, makedirs
from operator import contains

import re
import string


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

    def preprocess_data(self):
        if len(self.path) == 0:
            raise ValueError("Path can't be empty")

        with open(self.path, 'rb') as f:
            docs = pickle.load(f)

        all_sentences = []
        for doc in docs:
            for sent in doc.sentences:
                sentence = sent.text

                # Substitute entities that have interaction with drug1 and drug2. Array will contain entities that are interacted
                interacted_entities = []
                for pair in sent.pairs:
                    if pair.ddi:
                        e1 = self.find_entity(sent, pair.e1)
                        e2 = self.find_entity(sent, pair.e2)
                        
                        # we don't want to double insert same entities
                        if e1.text not in interacted_entities:
                            interacted_entities.append(e1.text)
                        if e2.text not in interacted_entities:
                            interacted_entities.append(e2.text)

                        sentence = sentence.replace(e1.text,'drug1')
                        sentence = sentence.replace(e2.text,'drug2')

                for entity in sent.entities:
                    if entity.text not in interacted_entities:
                        sentence = sentence.replace(entity.text,'drug0')

                sent_words = []
                regex = re.compile('[%s]' % re.escape('.,'))
                for w in sentence.split():
                    w_no_p = regex.sub('', w)  # word without , or .
                    if len(w_no_p) > 2:
                        sent_words.append(w_no_p.lower())

                all_sentences.append(sent_words)

        return all_sentences

    def find_entity(self, sentence, id):
        for entity in sentence.entities:
            if entity.id == id:
                return entity

def main():
    ddi_clf = DDIClassifier()
    ddi_clf.set_path(get_file_full_path('medline_train', pickled_files))
    ddi_clf.preprocess_data()

if __name__ == "__main__":
    main()
