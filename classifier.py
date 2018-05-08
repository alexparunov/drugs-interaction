#!/usr/bin/python3
from os.path import join, abspath, exists
from os import listdir
import pickle
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

def train_drugbank():
    # train/drugbank
    f = open(pickled_files[3], 'rb')
    docs = pickle.load(f)
    f.close()

    classes, feature_vectors_dict = split_dataset(docs)

    train = transform_dictionaries(feature_vectors_dict)



# split dataset into classes and sub-dictionaries
# return classes and dictionaries (i.e. feature vectors)
def split_dataset(docs):
    feature_vectors_dict = [] # feature vectors expressed as dicts. train data
    classes = [] # B,I,O classes
    for doc in docs:
        for m_dict in doc.featured_words_dict:
            classes.append(m_dict['0'])
            # we want sub-dictionary of all elements besides the class
            sub_dict = {k:v for k,v in m_dict.items() if k > '0'}
            feature_vectors_dict.append(sub_dict)

    return (classes, feature_vectors_dict)

# Transform dictionaries into binary array
def transform_dictionaries(feature_vectors_dict):
    vec = DictVectorizer()
    tr = vec.fit_transform(feature_vectors_dict).toarray()

    return tr

def main():
    train_drugbank()

if __name__ == "__main__":
    #test()
    main()
