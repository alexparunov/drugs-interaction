#!/usr/bin/python3
from os.path import join, abspath
from os import listdir
import pickle
import nltk

pickle_path = "data/pickle"
pickled_files = [join(abspath(pickle_path), f) for f in listdir(abspath(pickle_path))]

# train/drugbank
f = open(pickled_files[3], 'rb')
doc = pickle.load(f)[0]
f.close()

sentence = """CNS-Active Drugs Ethanol An additive effect on psychomotor performance was seen
with coadministration of eszopiclone and ethanol 0.70 g/kg for up to 4 hours after ethanol administration."""

def main():
    tokens = nltk.word_tokenize(sentence)
    print(tokens)

if __name__ == "__main__":
    main()
