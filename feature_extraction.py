#!/usr/bin/python3
from os.path import join, abspath
from os import listdir
import pickle

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

def extract_features():
    for file_name in pickled_files:
        f = open(file_name, 'rb')
        docs = pickle.load(f)
        f.close()
        all_featured_docs = []
        for doc in docs:
            doc.set_features()
            all_featured_docs.append(doc)

        with open(file_name,'wb') as f:
            pickle.dump(all_featured_docs, f)
            print("All documents with features are set in "+file_name)

def main():
    extract_features()

def test():
    # train/drugbank
    f = open(pickled_files[3], 'rb')
    docs = pickle.load(f)
    f.close()

    sent = docs[0].sentences[-1]
    print(docs[0].featured_words_dict)

if __name__ == "__main__":
    #main()
    test()
