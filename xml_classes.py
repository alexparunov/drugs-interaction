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
        charOffsets = []
        offsetWords = []
        for entity in self.entities:
            charOffsets.append(entity.charOffset)
            offsetWords.append(entity.text)

        # This will contain the list of ranges that are composed of offsets
        # This will be used lated to determine if ranges in tagged_words are matching
        # ranges of existing offsets
        all_offsets = []
        for charOffset in charOffsets:
            # Some offsets are combination of several offsets. For example
            # 105-115;130-140. While they are considered as one. We should take care of this case
            split_charOffsets = charOffset.split(";")
            # if length was changed, then split was done
            if len(split_charOffsets) > 1:
                # Since those ranges are together we can't include them inside all ranges
                # So we append subranges, which is another array
                subranges = []
                for spl_offset in split_charOffsets:
                    ofst = spl_offset.split("-")
                    beg = int(ofst[0]); end = int(ofst[1])
                    subranges.append(list(range(beg, end+1)))
                all_offsets.append(subranges)
            else:
                ofst = charOffset.split("-")
                beg = int(ofst[0]); end = int(ofst[1])
                _range = list(range(beg,end+1))
                all_offsets.append(_range)

        # Tuple: (word, pos_tag). For example ('Warfarin', 'NN')
        tagged_words = pos_tag(word_tokenize(self.text))

        all_features = []
        pos = 0
        for index, tagged_word in enumerate(tagged_words):
            # (TODO alex) check if this word is in our charOffsets
            # Must take care of nested ranges in all_offsets array.

            (inside_tag, pos) = self.is_inside_offsets(tagged_word[0], pos, all_offsets)

            if inside_tag == 1:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'B'))
            elif inside_tag == 2:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'I'))
            else:
                all_features.append(self.get_featured_tuple(index, tagged_words, 'O'))

        return all_features

    def is_inside_offsets(self, word, pos, all_offsets):
        # Find position in text where tagged_word starts
        p = self.text.find(word, pos)

        endpos = p+len(word)
        # offset where this tagged word exists
        offset = list(range(p, endpos))
        if offset in all_offsets:
            return (1, endpos)

        # This will deal with very special cases, when drugs are split in several offsets
        for m_offsets in all_offsets:
            if isinstance(m_offsets, list):
                # Our current offset is not in the array of offsets, so we return 'O' tag, i.e. 3
                if offset not in m_offsets:
                    return (3, endpos)

                for index, m_off in enumerate(m_offsets):
                    # length of a word inside various
                    if len(m_off) > 5:
                        if index == 0:
                            return (1, endpos)
                        else:
                            return (2, endpos)
                    else:
                        return (1, endpos)

        return (3, endpos)

    def get_featured_tuple(self, index, tagged_words, bio_tag):
        word = tagged_words[index][0]
        pos_tag = tagged_words[index][1]

        # Calculate how many numerics and several other symbols like ./().
        # Lambda function which uses ascii code of a character
        f = lambda w: 1 if ord(w) >= 40 and ord(w) <= 57 else 0
        numerics = list(map(f, word))

        # Calculate number of uppercase letters
        f = lambda w: 1 if ord(w) >= 65 and ord(w) <= 90 else 0
        upper_case = list(map(f, word))

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
