#!/usr/bin/python3
from os.path import join, abspath, exists
from os import listdir
import pickle
from nltk import word_tokenize, pos_tag
import nltk

# Call this method once
# nltk.download()

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

# train/drugbank
f = open(pickled_files[3], 'rb')
docs = pickle.load(f)
f.close()

def extract_features():
    for file_name in pickled_files:
        f = open(file_name, 'rb')
        docs = pickle.load(f)
        f.close()
        all_featured_docs = []
        for doc in docs:
            print("Setting feature for Document: "+doc.id)
            doc.set_features()
            all_featured_docs.append(doc)

        with open(file_name,'wb') as f:
            pickle.dump(all_featured_docs, f)
            print("All documents with features are set in "+file_name)

def main():
    extract_features()

    #sent = docs[2].sentences[0]
    #print(docs[2].__str__())
    #print(sent.set_features())
if __name__ == "__main__":
    main()
