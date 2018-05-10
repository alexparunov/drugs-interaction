#!/usr/bin/python3
from os.path import join, abspath, exists
from os import listdir
import pickle
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# Files are in the following order:
# 0 - medline_ner_test.pkl
# 1 - medline_train.pkl
# 2 - drug_bank_ddi_test.pkl
# 3 - drug_bank_train.pkl
# 4 - drug_bank_ner_test.pkl
# 5 - medline_ddi_test.pkl

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

class NERClassifier:
    def __init__(self):
        self.path = ""

    def set_path(self, path):
        self.path = path

    # split dataset into classes and sub-dictionaries
    # return classes and dictionaries (i.e. feature vectors)
    def split_dataset(self):
        if len(self.path) == 0:
            raise ValueError("Path can't be empty")

        f = open(self.path, 'rb')
        docs = pickle.load(f)
        f.close()

        feature_vectors_dict = [] # feature vectors expressed as dicts. train data
        classes = [] # B,I,O classes
        for doc in docs:
            for m_dict in doc.featured_words_dict:
                classes.append(m_dict['0'])
                # we want sub-dictionary of all elements besides the class
                sub_dict = {k:v for k,v in  m_dict.items() if k > '0'}

                feature_vectors_dict.append(sub_dict)

        return (classes, feature_vectors_dict)

    # train dataset, where X is a list of feature vectors expressed as dictionary
    # and Y is class variable, which is BIO tag in our case
    def train_dataset(self, X, Y):
        vec = DictVectorizer(sparse=False)
        svm_clf = svm.SVC()
        vec_clf = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        vec_clf.fit(X, Y)
        pickle.dump(vec_clf, "vectorizer_and_SVM.pkl")

    def train_drugbank(self):
        self.set_path(pickled_files[3])

        Y_train, X_train = self.split_dataset()
        self.train_dataset(X_train, Y_train)

def main():
    nerCl = NERClassifier()

    nerCl.train_drugbank()

if __name__ == "__main__":
    #test()
    main()
