#!/usr/bin/python3
from os.path import join, abspath, exists
from os import listdir
import pickle

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]


def train_drugbank():
    # train/drugbank
    f = open(pickled_files[3], 'rb')
    docs = pickle.load(f)
    f.close()

    feature_vectors = []
    for doc in docs:
        feature_vectors.append(doc.featured_words)

    print(len(feature_vectors))

def main():
    pass

if __name__ == "__main__":
    test()
    #main()
