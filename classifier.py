#!/usr/bin/python3
import pickle
import warnings
import argparse
from os.path import join, abspath, isdir
from os import listdir, makedirs
from operator import contains

from numpy.random import randint
import scipy

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.grid_search import RandomizedSearchCV

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

class Classifier:
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
        ner_classes = [] # B,I,O classes
        ddi_classes = [] # B,I,O classes
        dict_metadatas = []

        for doc in docs:
            for m_dict in doc.featured_words_dict:
                ner_classes.append(m_dict['0'])
                ddi_classes.append(m_dict['-1'])
                dict_metadatas.append(m_dict['-2'])

                # we want sub-dictionary of all elements besides the class
                sub_dict = {k:v for k,v in  m_dict.items() if k > '0' and not isinstance(v, list)}
                feature_vectors_dict.append(sub_dict)

        return (feature_vectors_dict, ner_classes, ddi_classes, dict_metadatas)

    def split_dataset_crf(self):
        if len(self.path) == 0:
            raise ValueError("Path can't be empty")

        with open(self.path, 'rb') as f:
            docs = pickle.load(f)

        all_sentences_features = []
        for doc in docs:
            # sent is a list of dictionaries
            for sent in doc.featured_sent_dict:
                ner_classes = []
                ddi_classes = []
                dict_metadatas = []
                sub_dicts = []
                for m_dict in sent:
                    ner_classes.append(m_dict['0'])
                    ddi_classes.append(m_dict['-1'])
                    dict_metadatas.append(m_dict['-2'])

                    # we want sub-dictionary of all elements besides the class
                    sub_dict = {k:v for k,v in  m_dict.items() if k > '0' and not isinstance(v, list)}
                    sub_dicts.append(sub_dict)

                all_sentences_features.append((sub_dicts, ner_classes, ddi_classes, dict_metadatas))

        return all_sentences_features

    # train dataset, where X is a list of feature vectors expressed as dictionary
    # and Y is class variable, which is BIO tag_type in our case. ratio is proportion of data to use to train
    def train_dataset_svm(self, X, Y, kernel, ratio):
        vec = DictVectorizer(sparse=False)
        svm_clf = svm.SVC(kernel = kernel, cache_size = 1800, C = 20, verbose = True, tol = 0.01)
        vec_clf = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        assert len(X) == len(Y)

        # subset of indexes to used in training
        r_indexes = randint(low = 0, high = len(X)-1, size = round(ratio*(len(X)-1)))

        X_subset = [X[i] for i in r_indexes]
        Y_subset = [Y[i] for i in r_indexes]

        vec_clf.fit(X_subset, Y_subset)

        return vec_clf

    def train_dataset_crf(self, X, Y, ratio):
        crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs', max_iterations = 100, all_possible_transitions = True)

        params_space = { 'c1': scipy.stats.expon(scale = 0.5), 'c2': scipy.stats.expon(scale = 0.05)}

        import multiprocessing
        cpus = multiprocessing.cpu_count()
        rs = RandomizedSearchCV(crf, params_space, cv = 5, verbose = 1, n_jobs = cpus-1, n_iter = 500)

        assert len(X) == len(Y)

        # subset of indexes to used in training
        r_indexes = randint(low = 0, high = len(X)-1, size = round(ratio*(len(X)-1)))

        X_subset = [X[i] for i in r_indexes]
        Y_subset = [Y[i] for i in r_indexes]

        rs.fit(X_subset, Y_subset)

        return rs

    def train_NER_model(self, train_folder, kernel = 'linear', ratio = 1, classifier = 1):
        if not isdir('models'):
            makedirs('models')

        model_name = ""
        model_index = 0
        model_names = [join(abspath("models"), f) for f in listdir(abspath("models"))]

        if train_folder == 1:
            path = get_file_full_path("drug_bank_train.pkl", pickled_files)
            self.set_path(path)
            drugbank_models = list(filter(lambda x: contains(x, 'drugbank_model_'), model_names))
            model_index = len(drugbank_models) # save next model
            model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'
            print("Started training NER Drugbank model...")
        elif train_folder == 2:
            path = get_file_full_path("medline_train.pkl", pickled_files)
            self.set_path(path)
            medline_models = list(filter(lambda x: contains(x, 'medline_model_'), model_names))
            model_index = len(medline_models) # save next model
            model_name = 'models/medline_model_'+str(model_index)+'.pkl'
            print("Started training NER Medline model...")
        else:
            raise ValueError('train_folder value should be 1 - drugbank, or 2 - medline')

        if classifier == 2:
            featured_sent_dict = self.split_dataset_crf()
            X_train = [f[0] for f in featured_sent_dict]
            Y_train = [f[1] for f in featured_sent_dict]
            clf = self.train_dataset_crf(X_train, Y_train, ratio)
        else:
            # we ignore Y_ddi classes since they are not used for NER model training
            X_train, Y_train, Y_ddi, metadatas = self.split_dataset()
            clf = self.train_dataset_svm(X_train, Y_train, kernel, ratio)

        joblib.dump(clf, model_name)
        print("\nNER Model trained and saved into", model_name)

    def test_NER_model(self, model_index, test_folder, classifier = 1):
        model_name = ""
        predictions_name = ""
        if test_folder == 1:
            model_name = 'models/drugbank_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/drugbank_model_'+str(model_index)+'.txt'
            path = get_file_full_path("drug_bank_ner_test.pkl", pickled_files)
            self.set_path(path)
            print("Testing NER Drugbank model", model_index,"...")
        elif test_folder == 2:
            model_name = 'models/medline_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/medline_model_'+str(model_index)+'.txt'
            path = get_file_full_path("medline_ner_test.pkl", pickled_files)
            self.set_path(path)
            print("Testing NER Medline model", model_index,"...")
        else:
            raise ValueError('test_folder value should be 1 - drugbank, or 2 - medline')

        vec_clf = joblib.load(model_name)

        if classifier == 2:
            featured_sent_dict = self.split_dataset_crf()
            X_test = [f[0] for f in featured_sent_dict]
            Y = [f[1] for f in featured_sent_dict] # ner classes
            Y_test = []
            for y in Y:
                for y_test in y:
                    Y_test.append(y_test)

            met = [f[3] for f in featured_sent_dict]
            metadatas = []
            for metadata in met:
                for met in metadata:
                    metadatas.append(met)
        else:
            # metadatas are of type: sentenceId | offsets... | text | type
            # we ignore Y_ddi classes since they are not used for NER model training
            X_test, Y_test, Y_ddi, metadatas = self.split_dataset()

        if classifier == 2:
            preds = vec_clf.predict(X_test)
            predictions = []
            for pred in preds:
                for prediction in pred:
                    predictions.append(prediction)
        else:
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
            # if prediction is B_type or I_type then we predicted the drug and it's type is after B_, thus we can write into check file
            if pred[:2] == 'B_' or pred[:2] == 'I_':
                line = metadata[0] + '|' + metadata[1] + '|' + metadata[2] + '|' + pred[2:]
                pr_f.write(line + '\n')

        print("\nNER Predictions are saved in file", predictions_name)
        pr_f.close()

    def train_DDI_model(self, train_folder, kernel = 'linear', ratio = 1, classifier = 1):
        if not isdir('models'):
            makedirs('models')

        model_name = ""
        model_index = 0
        model_names = [join(abspath("models"), f) for f in listdir(abspath("models"))]
        if train_folder == 1:
            path = get_file_full_path("drug_bank_train.pkl", pickled_files)
            self.set_path(path)
            drugbank_ddi_models = list(filter(lambda x: contains(x, 'drugbank_ddi_model_'), model_names))
            model_index = len(drugbank_ddi_models) # save next model
            model_name = 'models/drugbank_ddi_model_'+str(model_index)+'.pkl'
            print("Started training DDI Drugbank model...")
        elif train_folder == 2:
            path = get_file_full_path("medline_train.pkl", pickled_files)
            self.set_path(path)
            medline_ddi_models = list(filter(lambda x: contains(x, 'medline_ddi_model_'), model_names))
            model_index = len(medline_ddi_models) # save next model
            model_name = 'models/medline_ddi_model_'+str(model_index)+'.pkl'
            print("Started training DDI Medline model...")
        else:
            raise ValueError('train_folder value should be 1 - drugbank, or 2 - medline')

        X_train, Y_ner, Y_train, metadatas = self.split_dataset()

        if classifier == 2:
            featured_sent_dict = self.split_dataset_crf()
            X_train = [f[0] for f in featured_sent_dict]
            Y_train = [f[2] for f in featured_sent_dict] # ddi classes
            clf = self.train_dataset_crf(X_train, Y_train, ratio)
        else:
            # we ignore Y_ddi classes since they are not used for NER model training
            X_train, Y_train, Y_ddi, metadatas = self.split_dataset()
            clf = self.train_dataset_svm(X_train, Y_train, kernel, ratio)

        joblib.dump(clf, model_name)
        print("\nDDI Model trained and saved into", model_name)

    def test_DDI_model(self, model_index, test_folder, classifier):
        model_name = ""
        predictions_name = ""
        if test_folder == 1:
            model_name = 'models/drugbank_ddi_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/drugbank_ddi_model_'+str(model_index)+'.txt'
            path = get_file_full_path("drug_bank_ddi_test.pkl", pickled_files)
            self.set_path(path)
            print("Testing DDI Drugbank model", model_index,"...")
        elif test_folder == 2:
            model_name = 'models/medline_ddi_model_'+str(model_index)+'.pkl'
            predictions_name = 'predictions/medline_ddi_model_'+str(model_index)+'.txt'
            path = get_file_full_path("medline_ddi_test.pkl", pickled_files)
            self.set_path(path)
            print("Testing DDI Medline model", model_index,"...")
        else:
            raise ValueError('test_folder value should be 1 - drugbank, or 2 - medline')

        vec_clf = joblib.load(model_name)

        if classifier == 2:
            featured_sent_dict = self.split_dataset_crf()
            X_test = [f[0] for f in featured_sent_dict]
            Y = [f[1] for f in featured_sent_dict]
            Y_test = []
            for y in Y:
                for y_test in y:
                    Y_test.append(y_test)

            met = [f[3] for f in featured_sent_dict]
            metadatas = []
            for metadata in met:
                for met in metadata:
                    metadatas.append(met)
        else:
            # metadatas are of type: sentenceId | offsets... | text | type
            # we ignore Y_ddi classes since they are not used for NER model training
            X_test, Y_test, Y_ddi, metadatas = self.split_dataset()

        if classifier == 2:
            preds = vec_clf.predict(X_test)
            predictions = []
            for pred in preds:
                for prediction in pred:
                    predictions.append(prediction)
        else:
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
            # if prediction is 1_type then we predicted ddi correctly, thus we can write into check file
            if len(metadata[5]) > 0 and len(metadata[6]) > 0:
                # some document sentence ids are completely wrong, we want only DDI-Medline
                if metadata[0][:12] == 'DDI-MedLine.':
                    line = metadata[0]+'|'+metadata[5]+'|'+metadata[6]+'|'+pred[0]+'|'+pred[2:]
                    pr_f.write(line + '\n')

        print("\nDDIPredictions are saved in file", predictions_name)
        pr_f.close()

parser = argparse.ArgumentParser(description = "Train or Test model")
parser.add_argument('-t','--task', type = int, help = "Task of problem. 1 - NER task, 2 - DDI task.", action = "store", default = -1)
parser.add_argument('--train', help = "Train model", action="store_true")
parser.add_argument('--test', help = "Test model at index i", action="store_true")
parser.add_argument('-f','--folder_index', type = int, help = "Folder number. 1 - drugbank, 2 - medline", action = "store", default = -1)
parser.add_argument('-i','--model_index', type = int, help = "Index of a model to test", action = "store", default = -1)
parser.add_argument('-r','--ratio', type = float, help = "Ratio of data to use for training", action = "store", default = 1)
parser.add_argument('-c','--classifier', type = int, help = "Classifier to use. 1 - SVM, 2 - CRF", action = "store", default = 1)

def main():
    clasf = Classifier()

    # stupid scikit warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        args = parser.parse_args()

        if args.train:
            folder_index = args.folder_index
            ratio = args.ratio
            if ratio <= 0 and ratio > 1:
                ratio = 1
            if folder_index == 1 or folder_index == 2:
                if args.task == 1:
                    clasf.train_NER_model(train_folder = folder_index, ratio = ratio,
                            classifier = args.classifier)
                elif args.task == 2:
                    clasf.train_DDI_model(train_folder = folder_index, ratio = ratio,
                            classifier = args.classifier)
                else:
                    parser.print_help()
            else:
                parser.print_help()
        elif args.test:
            model_index = args.model_index
            folder_index = args.folder_index
            if model_index >= 0 and folder_index >= 1:
                if args.task == 1:
                    clasf.test_NER_model(model_index = model_index, test_folder = folder_index,
                            classifier = args.classifier)
                elif args.task == 2:
                    clasf.test_DDI_model(model_index = model_index, test_folder = folder_index,
                            classifier = args.classifier)
                else:
                    parser.print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
