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

        # Some offsets are combination of several offsets. For example
        # 105-115;130-140. While they are considered as one. We should take care of this case

        # This index will determine is word is beginning of chunk of offsetted words
        # If i == 0 it means that word is beginning of chunk.
        # This will be used later in BIO tagging. So in this case we will assign tag B
        i = 0
        for charOffset in charOffsets:
            split_charOffsets = charOffset.split(";")
            # if length was changed, then split was done
            if len(split_charOffsets) > 1:
                for offset in split_charOffsets:
                    ofst = offset.split('-')
                    beg = int(ofst[0]); end = int(ofst[1])
                    word = self.text[beg:end+1]
                    offsetWords.append((word.strip(), i))
            else:
                ofst = charOffset.split('-')
                beg = int(ofst[0]); end = int(ofst[1])
                offsetWords.append((self.text[beg:end+1].strip(), i))
            i += 1

        # Tuple: (word, pos_tag). For example ('Warfarin', 'NN')
        tagged_words = pos_tag(word_tokenize(self.text))
        print(tagged_words)
        all_features = []

        for tagged_word in tagged_words:
            # Initial features
            features = [tagged_word[0],'O']
            distance_words = list(map(lambda x: (tagged_word[0], distance.get_jaro_distance(tagged_word[0], x[0])), offsetWords))
            distance_words = list(filter(lambda x: x[1] > 0.85, distance_words))

            for offsetWord in offsetWords:
                # Need to calculate the distance between words. The most similar ones to tagged are offsetwords
                word_distance = 1
                if word_distance > 0.85:
                    # In this case words are more or less similar, so we take the offsetword instead of tagged_word.
                    # Tuple will be (offsetword,'I') where I means this word is inside the offsets
                    # We are doing BIO tagging, which will be used later for classification

                    # In this case offsetWord[1] == 0, which means it is a beginning of chunks of offsetWords
                    # In other words it is inside
                    if offsetWord[1] == 0:
                        features = [tagged_word[0], 'B']
                    else:
                        features = [tagged_word[0], 'I']

            features.append(tagged_word[1])
            features.append(len(features[0]))

            all_features.append(tuple(features))

        return all_features

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
