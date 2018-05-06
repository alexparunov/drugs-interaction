from nltk import word_tokenize, pos_tag
from nltk.stem import SnowballStemmer
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
            # We don't want to save punctuations
            if len(tagged_word[0]) < 2:
                continue
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

        orthographical_feature = "alphanumeric"
        f_uppercase = lambda w: 1 if ord(w) >= 65 and ord(w) <= 90 else 0
        upper_case = list(map(f_uppercase, word))

        if sum(upper_case) == len(word):
            orthographical_feature = "all-capitalized"
        elif f_uppercase(word[0]) == 1:
            orthographical_feature = "is-capitalized"

        # Lambda function which uses ascii code of a character
        f_numerics = lambda w: 1 if w.isnumeric() else 0
        numerics = list(map(f_numerics, word))

        if sum(numerics) == len(word):
            orthographical_feature = "all-digits"

        if "-" in word:
            orthographical_feature += "Y"
        else:
            orthographical_feature += "N"


        snowball_stemmer = SnowballStemmer("english")
        stemmed_word = snowball_stemmer.stem(word)
        ind = word.find(stemmed_word)

        prefix_len = len(word[:ind])
        suffix_len = len(word) - prefix_len - len(stemmed_word)

        pl3 = int(prefix_len == 3); sufl3 = int(suffix_len == 3)
        pl4 = int(prefix_len == 4); sufl4 = int(suffix_len == 4)
        pl5 = int(prefix_len == 5); sufl5 = int(suffix_len == 5)

        # Generalized Word Shape Feature. Map upper case, lower case, digit and
        # other characters to X,x,0 and O respectively
        # Aspirin1+ will be mapped to Xxxxxxx0O, for example

        word_shape = ""
        for w in word:
            if w.isupper():
                word_shape += "X"
            elif w.islower():
                word_shape += "x"
            elif w.isnumeric():
                word_shape += "0"
            else:
                word_shape += "O"


        # Brief word shape. maps consecutive uppercase letters, lowercase letters,
        # digits, and other characters to “X,” “x,” “0,” and “O,” respectively.
        # Aspirin1+ will be mapped to Xx0O

        # Lambda function to determine if character belongs to category other based on its ascii value
        f_other = lambda w: True if (ord(w) < 48 or (ord(w) >= 58 and ord(w) <= 64) or
        (ord(w) >= 91 and ord(w) <= 96) or ord(w) > 122) else False

        word_shape_brief = ""
        i = 0
        while i < len(word):
            if word[i].isupper():
                word_shape_brief += "X"
                while i < len(word) and word[i].isupper():
                    i += 1
                if i == len(word):
                    break
            if word[i].islower():
                word_shape_brief += "x"
                while i < len(word) and word[i].islower():
                    i += 1
                if i == len(word):
                    break
            if word[i].isnumeric():
                word_shape_brief += "0"
                while i < len(word) and word[i].isnumeric():
                    i += 1
                if i == len(word):
                    break
            if f_other(word[i]):
                word_shape_brief += "O"
                while i < len(word) and f_other(word[i]):
                    i += 1
                    if i == len(word):
                        break
            i += 1
            if i == len(word):
                break


        # May be add Y,N if drug is in drugbank or FDA approved list of drugs?

        # Following this table https://www.hindawi.com/journals/cmmm/2015/913489/tab1/
        # we get feature vector of following type
        #[bio_tag, f1,f2, len(word), f4, f9, f10, f11, f12, f13, f14, f15, f16]

        features = [bio_tag, word, pos_tag, len(word), orthographical_feature,
                    pl3, pl4, pl5, sufl3, sufl4, sufl5, word_shape, word_shape_brief]

        return tuple(features)

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
