#!/usr/bin/python3
from os.path import join, abspath
from os import listdir
import pickle
from nltk import word_tokenize, pos_tag

#nltk.download() call this method only once

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

# train/drugbank
f = open(pickled_files[3], 'rb')
docs = pickle.load(f)
f.close()

def main():
    sent = docs[0].sentences[-1]
    print(sent.__str__())
    sent.set_features()

if __name__ == "__main__":
    main()
