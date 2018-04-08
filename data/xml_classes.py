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
