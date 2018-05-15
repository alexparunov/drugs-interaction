#!/usr/bin/python3
from os.path import join, abspath, isdir
from os import listdir, makedirs
import pickle
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import warnings
import argparse

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
                dict_metadatas.append(m_dict['-1'])

                # we want sub-dictionary of all elements besides the class
                sub_dict = {k:v for k,v in  m_dict.items() if k > '0' and not isinstance(v, list)}
                feature_vectors_dict.append(sub_dict)

        return (feature_vectors_dict, classes,  dict_metadatas)

    # train dataset, where X is a list of feature vectors expressed as dictionary
    # and Y is class variable, which is BIO tag in our case
    def train_dataset(self, X, Y, kernel):
        vec = DictVectorizer(sparse=False)
        svm_clf = svm.SVC(kernel = kernel, cache_size = 1800, C = 20, verbose = True, tol = 0.01)
        vec_clf = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        vec_clf.fit(X, Y)

        return vec_clf

    def train_NER_model(self, train_folder, kernel = 'linear'):
        if not isdir('models'):
            makedirs('models')

        model_name = ""
        model_index = 0
        model_names = [join(abspath("models"), f) for f in listdir(abspath("models"))]

        from operator import contains
        if train_folder == 1:
            self.set_path(pickled_files[3])
            drugbank_models = list(filter(lambda x: contains(x, 'drugbank_model_'), model_names))
            model_index = len(drugbank_models) # save next model
            model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'
        elif train_folder == 2:
            self.set_path(pickled_files[0])
            medline_models = list(filter(lambda x: contains(x, 'medline_model_'), model_names))
            model_index = len(medline_models) # save next model
            model_name = 'models/medline_model_'+str(model_index)+'.pkl'
        else:
            raise ValueError('train_folder value should be 1 - drugbank, or 2 - medline')

        X_train, Y_train, metadatas = self.split_dataset()
        vec_clf = self.train_dataset(X_train, Y_train, kernel)

        joblib.dump(vec_clf, model_name)
        print("\nModel trained and saved into", model_name)

    def test_NER_model(self, model_index, test_folder):
        model_name = ""
        predictions_name = ""
        if test_folder == 1:
            model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/drugbank_model_'+str(model_index)+'.txt'
            self.set_path(pickled_files[4])
        elif test_folder == 2:
            model_name = 'models/medline_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/medline_model_'+str(model_index)+'.txt'
            self.set_path(pickled_files[0])
        else:
            raise ValueError('test_folder value should be 1 - drugbank, or 2 - medline')

        print("Testing model", model_index,"...")

        vec_clf = joblib.load(model_name)

        # metadatas are of type: sentenceId | offsets... | text | type
        X_test, Y_test, metadatas = self.split_dataset()
        predictions = vec_clf.predict(X_test)
        assert len(predictions) == len(Y_test) == len(metadatas)

        if not isdir('predictions'):
            makedirs('predictions')

        pr_f = open(predictions_name,'w')
        # clear file, i.e. remove all
        pr_f.close()

        # reopen clean file
        pr_f = open(predictions_name, 'w')

        for i, pred in enumerate(predictions):
            metadata = metadatas[i]
            # if prediction is B_type or I_type then we predicted the drug and it's type is after B_
            if pred[:2] == 'B_' or pred[:2] == 'I_':
                line = metadata[0] + '|' + metadata[1] + '|' + metadata[2] + '|' + pred[2:]
                pr_f.write(line + '\n')

        print("Predictions are saved in file", predictions_name)
        pr_f.close()

parser = argparse.ArgumentParser(description = "Train or Test model")
parser.add_argument('--train', help = "Train model", action="store_true")
parser.add_argument('--test', help = "Test model at index i", action="store_true")
parser.add_argument('-f','--folder_index', type=int, help = "Folder number. 1 is drugbank, 2 is medline", action = "store", default = -1)
parser.add_argument('-i','--model_index', type=int, help = "Index of a model to test", action = "store", default = -1)

def main():
    nerCl = NERClassifier()
    # stupid scikit warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        args = parser.parse_args()
        if args.train:
            folder_index = args.folder_index
            if folder_index == 1 or folder_index == 2:
                nerCl.train_NER_model(train_folder = folder_index)
            else:
                parser.print_help()
        elif args.test:
            model_index = args.model_index
            folder_index = args.folder_index
            if model_index >= 0 and folder_index >= 1:
                nerCl.test_NER_model(model_index = model_index, test_folder = folder_index)
            else:
                parser.print_help()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
