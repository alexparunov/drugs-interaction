#!/usr/bin/python3
from os.path import join, abspath, isdir
from os import listdir, makedirs
import pickle
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import warnings

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

        with open(self.path, 'rb') as f:
            docs = pickle.load(f)

        feature_vectors_dict = [] # feature vectors expressed as dicts. train data
        classes = [] # B,I,O classes
        dict_metadatas = []

        for doc in docs:
            for m_dict in doc.featured_words_dict:
                classes.append(m_dict['0'])
                # we want sub-dictionary of all elements besides the class
                sub_dict = {k:v for k,v in  m_dict.items() if k > '0'}

                dict_metadatas.append(m_dict['-1'])
                feature_vectors_dict.append(sub_dict)

        return (feature_vectors_dict, classes,  dict_metadatas)

    # train dataset, where X is a list of feature vectors expressed as dictionary
    # and Y is class variable, which is BIO tag in our case
    def train_dataset(self, X, Y, kernel = 'rbf'):
        vec = DictVectorizer(sparse=False)
        svm_clf = svm.SVC(kernel = kernel, cache_size = 1200)
        vec_clf = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        vec_clf.fit(X, Y)

        return vec_clf

    def train_drugbank(self, kernel):
        self.set_path(pickled_files[3])

        X_train, Y_train, metadatas = self.split_dataset()
        vec_clf = self.train_dataset(X_train, Y_train, kernel)

        if not isdir('models'):
            makedirs('models')

        model_names = [join(abspath("models"), f) for f in listdir(abspath("models"))]

        from operator import contains
        drugbank_models = list(filter(lambda x: contains(x, 'drugbank_model_'), model_names))
        model_index = len(drugbank_models) + 1 # save next model

        model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'

        with open(model_name,'wb') as f:
            pickle.dump(vec_clf, f)
            print("Model trained and saved into", model_name)

    def test_drugbank(self, model_index):
        self.set_path(pickled_files[4])
        print("Testing model", model_index)

        model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'

        with open(model_name,'rb') as f:
            vec_clf = pickle.load(f)

        # metadatas are of type: sentenceId | offsets... | text | type

        X_test, Y_test, metadatas = self.split_dataset()
        predictions = vec_clf.predict(X_test)
        assert len(predictions) == len(Y_test) == len(metadatas)

        if not isdir('predictions'):
            makedirs('predictions')

        predictions_name = 'predictions/drugbank_model_'+str(model_index)+'.txt'
        pr_f = open(predictions_name,'w')

        for i, pred in enumerate(predictions):
            metadata = metadatas[i]

            if len(metadata) > 0:
                line = metadata[0] + '|' + metadata[1] + '|' + metadata[2] + '|' + pred
                pr_f.write(line + '\n')

        print("Predictions are saved in file", predictions_name)
        pr_f.close()

def main():
    nerCl = NERClassifier()
    # stupid scikit warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nerCl.test_drugbank(0)

if __name__ == "__main__":
    main()
