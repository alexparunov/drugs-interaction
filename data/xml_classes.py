from nltk import word_tokenize, pos_tag
from pyjarowinkler import distance

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
        featured_sentences = []
        for sentence in self.sentences:
            featured_sentences.append(sentence.set_features())
        self.featured_sentences = featured_sentences

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
        for pair in self.pairs:
            st = st + pair.__str__() +'\n'
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
        all_features = []
        for i in range(len(charOffsets)):
            # Some offsets are combination of several offsets. For example
            # 105-115;130-140. While they are considered as one. We should take care of this case
            split_charOffsets = charOffsets[i].split(";")
            # if length was changed, then split was done
            if len(split_charOffsets) > 1:
                # If lengths is 0 it means no word was added, so it is definitely a beginning tag
                if len(all_features) == 0:
                    all_features.append(self.get_featured_tuple(offsetWords[i],'NN','B'))
                else:
                    all_features.append(self.get_featured_tuple(offsetWords[i],'NN','I'))
            else:
                ofst = charOffsets[i].split("-")
                beg = int(ofst[0]); end = int(ofst[1])
                _range = list(range(beg,end+1))
                all_offsets.append(_range)

        # Tuple: (word, pos_tag). For example ('Warfarin', 'NN')
        tagged_words = pos_tag(word_tokenize(self.text))

        pos = 0
        for tagged_word in tagged_words:
            # Find position in text where tagged_word starts
            pos = self.text.find(tagged_word[0], pos)
            _range = list(range(pos, pos+len(tagged_word[0])))

            #If the range is in all_offsets it mean this word is inside the offset chars
            if _range in all_offsets:
                if len(all_features) == 0:
                    all_features.append(self.get_featured_tuple(tagged_word[0],tagged_word[1],'B'))
                else:
                    all_features.append(self.get_featured_tuple(tagged_word[0],tagged_word[1],'I'))
            else:
                all_features.append(self.get_featured_tuple(tagged_word[0],tagged_word[1],'O'))

        return all_features

    def get_featured_tuple(self, word, pos_tag, bio_tag):
        feature = (word, bio_tag, pos_tag, len(word))
        return feature
    
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
