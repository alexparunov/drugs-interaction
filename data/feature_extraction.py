#!/usr/bin/python3
from os.path import join, abspath, exists
from os import listdir
import pickle
from nltk import word_tokenize, pos_tag

#nltk.download() call this method only once

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path)) if f.find("_f") < 0]

# train/drugbank
#f = open(pickled_files[3], 'rb')

def extractFeatures(debug = False):
    for file_name in pickled_files:
        featured_doc_file_name = file_name[:file_name.find(".pkl")]+"_f.pkl"
        if exists(featured_doc_file_name) and not debug:
            return

        if file_name.find("_train") > 0:
            f = open(file_name, 'rb')
            docs = pickle.load(f)
            f.close()
            all_featured_docs = []
            for doc in docs:
                print("Setting feature for Document: "+doc.id)
                doc.set_features()
                all_featured_docs.append(doc)
                print("Features set.")

            with open(featured_doc_file_name,'wb') as f:
                pickle.dump(all_featured_docs, f)
                print("All documents with features are set in "+file_name)

def main():
    extractFeatures()

if __name__ == "__main__":
    main()
