from nltk import word_tokenize, pos_tag
from numpy import isfinite
import re
import string

class Document:
    def __init__(self, id):
        self.id = id
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def __str__(self):
        st = "DOCUMENT. Id: "+self.id + '\n'
        for sentence in self.sentences:
            st = st + sentence.__str__() + '\n'
        return st

    # Sets features for each sentence
    def set_features(self):
        featured_words = []
        for sentence in self.sentences:
            featured_words.extend(sentence.set_features())
        self.featured_words = featured_words

class Sentence:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.entities = []
        self.pairs = []

    def add_entity(self, entity):
        self.entities.append(entity)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def __str__(self):
        st = "\t---SENTENCE. Id: "+self.id+", Text: "+self.text + '\n'
        for entity in self.entities:
            st = st + entity.__str__() +'\n'
        # for pair in self.pairs:
        #     st = st + pair.__str__() +'\n'
        return st

    def set_features(self):
        B_tags = [] #list with words that are of type B tag
        I_tags = [] #list of words that are of type I tag
        for entity in self.entities:
            words = entity.text.split(" ") #split words in text to tag
            for index, word in enumerate(words):
                if index == 0:
                    B_tags.append(word)
                else:
                    I_tags.append(word)

        tagged_words = pos_tag(word_tokenize(self.text))
        all_features = []

        for index, tagged_word in enumerate(tagged_words):
            if tagged_word[0] in B_tags:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'B'))
            elif tagged_word[0] in I_tags:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'I'))
            else:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'O'))

        return all_features

    def get_featured_tuple(self, index, tagged_words, bio_tag):
        word = tagged_words[index][0]
        pos_tag = tagged_words[index][1]
        Ngram = 3

        # ORTHOGRAPHIC Features:
        # Starting with a uppercase letter, containing only alphanumeric characters, 
        # containing a hyphen, digits and capitalized characters counting, etc.

        # WORD SHAPE:
        # Uppercase letters, lowercase letters, digits, and other characters in a word
        # are converted to “A”, “a”, “0” and “O”, respectively. For example, 
        # “Phenytoin” is mapped to “Aaaaaaaaa”.

        # Calculate how many numerics and several other symbols like ./().
        # Lambda function which uses ascii code of a character
        f = lambda w: 1 if ord(w) >= 40 and ord(w) <= 57 else 0
        numerics = list(map(f, word))

        # Calculate number of uppercase letters
        f = lambda w: 1 if ord(w) >= 65 and ord(w) <= 90 else 0
        upper_case = list(map(f, word))

        # Character feature, N-grams of characters in a word

        # Word feature, N-grams of words in a context window

        # Lemma, N-grams of lemmas of words.

        # Stem, N-grams of stems of words.

        # POS, N-grams of POS tags.

        # Text Chunking, N-grams of text chunking tags.

        # Dependency Parsing, Dependency parsing results of words in a sentence.

        # Affix, Suffixes and prefixes of a word.

        # Dictionary Feature, Whether an n-gram matches with part of a drug name in drug dictionaries.

        # Outputs of NER tools, Features derived from the output of existing chemical NER tools.

        # Word Representation, Word representation features based on Brown clustering, word2vec, etc.

        # Conjunction Feature, Conjunctions of different types of features, e.g., conjunction of lemma and POS features.

        feature = [word, bio_tag, pos_tag, len(word), sum(numerics), sum(upper_case)]

        return tuple(feature)

class Entity:
    def __init__(self, id, charOffset, type, text):
        self.id = id
        self.charOffset = charOffset
        self.type = type
        self.text = text

    def __str__(self):
        st = "\t\t---ENTITY. Id: "+self.id+", CharOffSet: "+self.charOffset+", Type: "+self.type+", Text: "+self.text
        return st

class Pair:
    def __init__(self, id, e1, e2, ddi):
        self.id = id
        self.e1 = e1
        self.e2 = e2
        self.ddi = ddi
        self.type = ""

    def set_type(self, type):
        self.type = type

    def __str__(self):
        st = "\t\t---PAIR. Id: "+self.id+", E1: "+self.e1+", E2: "+self.e2+", DDI: "+str(self.ddi)
        if self.ddi:
            st += ", Type: "+self.type
        return st
