from nltk import word_tokenize, pos_tag
from nltk.stem import SnowballStemmer

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
        featured_words_dict = [] #we need dictionary for DictVectorizer
        featured_sent_dict = []
        for sentence in self.sentences:
            sent_features = sentence.set_features()
            sent_dict = []
            for s_feature in sent_features:
                # first indext contains BIO tag
                # last index contains DDI bio tag
                # previous to last index contains metadata
                ddi_tag = s_feature.pop()
                metadata = s_feature.pop()

                m_dict = {'-2': metadata, '-1': ddi_tag}
                for i in range(len(s_feature)):
                    m_dict[str(i)] = s_feature[i]

                featured_words_dict.append(m_dict)
                sent_dict.append(m_dict)

            featured_sent_dict.append(sent_dict)

        self.featured_words_dict = featured_words_dict
        self.featured_sent_dict = featured_sent_dict

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

        all_features = self.get_vector_metadatas(all_features)

        return all_features

    # We need this loop in order to assign metadata to a drug-type word.
    # It's necessary since our output should be of type:
    # for NER task we need sentenceId|offsets...|text|type
    # for DDI prediction we need sentenceId|idDrug1|idDrug2|prediction (ddi = 1 or ddi = 0)|type (advice, effect, etc.)
    def get_vector_metadatas(self, all_features):
        pos = 0 #initial search positions
        new_all_features = [] #vector of new features with appended metadata
        for i in range(len(all_features)):
            charOffset = ""
            type = "" #type of drug which is empty by default
            f_vector = all_features[i] #feature vector
            f_word = str(f_vector[5]) #word which is contained in postion 3
            w_text = "" # word text
            # if BIO tag of feature vector is B then we proceed with special case assignment
            if f_vector[0] == 'B':
                pos = self.text.find(f_word, pos) #find position where word starts in the sentence
                # this should not be since there are always words in a sentence, but we don't want to deal with negative positions just in case
                if pos < 0:
                    continue

                # beginning and end positions of word, so offset will be set accordingly
                beg = pos; end = pos + len(f_word) - 1
                charOffset = str(beg)+"-"+str(end)
                pos = end #set a new search position to end of previous word, so that we search different words in sentence
                w_text = f_word

                metadata = [self.id, charOffset, w_text, type]
                # appending metadata to last extracted feature vector (might be from inner while loop)
                f_vector.append(metadata)
                new_all_features.append(f_vector)

                while i < len(all_features) - 1:
                    f_vector = all_features[i+1] #next word in a feature vectors

                    # As soon as next words BIO tag is not I, we break the inner loop
                    # otherwise we continue appending to charOffsetString. So eventually it looks like
                    # 100-150;155-170;190-200...
                    if f_vector[0] != 'I':
                        break

                    f_word = str(f_vector[5])
                    pos = self.text.find(f_word, pos)

                    if pos < 0:
                        continue

                    w_text += " "+ f_word
                    beg = pos; end = pos + len(f_word) - 1
                    charOffset += ";" + str(beg)+"-"+str(end)
                    pos = end
                    i += 1

                    metadata = [self.id, charOffset, w_text, type]
                    # appending metadata to last extracted feature vector (might be from inner while loop)
                    f_vector.append(metadata)
                    new_all_features.append(f_vector)
            else:
                # Otherwise BIO tag is O so we simply have charOffset and empty type
                f_word = str(f_vector[5])
                w_text = f_word
                pos = self.text.find(f_word, pos)
                if pos < 0:
                    continue

                beg = pos; end = pos + len(f_word) - 1
                charOffset += str(beg)+"-"+str(end)
                pos = end

                metadata = [self.id, charOffset, w_text, type]
                # appending metadata to last extracted feature vector (might be from inner while loop)
                f_vector.append(metadata)
                new_all_features.append(f_vector)

        updated_features = []
        for f_vector in new_all_features:
            # Update tags. It means each tag will be of type B_drug/B_group/I_drug/I_group/etc.
            try:
                metadata = f_vector.pop()
                if not isinstance(metadata, list):
                    continue

                word_ddi = self.get_word_ddi(str(f_vector[5]))
                metadata.extend(word_ddi)

                assert len(metadata) == 8
                # if ddi = True then it's 1, otherwise it's 0
                ddi_tag = int(metadata[4])

                # append type of interaction in both cases
                if ddi_tag > 0:
                    ddi_tag = str(ddi_tag)+"_"+metadata[len(metadata)-1]
                else:
                    ddi_tag = str(ddi_tag)+"_null"

                # update metadata
                f_vector.append(metadata)

                # set class of ddi to the last element
                f_vector.append(ddi_tag)

                tag = f_vector[0]
                if tag == 'B' or tag == 'I':
                    type = self.get_word_entity(str(f_vector[5]))
                    tag = tag + "_"+type
                    f_vector[0] = tag
                # remove words at windows. Words are located at positions 1,3,7,9 in window of n = 2
                # We need to remove them otherwise training takes forever
                ff_vector = [f_vector[j] for j in range(len(f_vector)) if j != 1 and j != 3 and j != 7 and j != 9]
                #print(ff_vector)
                updated_features.append(ff_vector)
            except TypeError:
                pass

        return updated_features

    # since words is of type BI tag, then it must have type.
    # So we search through all entities and if word is contained then we set type
    # NOTE that all types of word in offsets like this 100-150;155-170;190-200 will be the same
    def get_word_entity(self, f_word):
        for entity in self.entities:
            text_ar = entity.text.split()
            if f_word in text_ar:
                return entity.type

    def get_word_ddi(self, f_word):
        ddi = False
        idDrug1 = ""
        idDrug2 = ""
        type = ""
        for entity in self.entities:
            text_ar = entity.text.split()
            if f_word in text_ar:
                for pair in self.pairs:
                    if pair.e1 == entity.id or pair.e2 == entity.id:
                        ddi = pair.ddi
                        idDrug1 = pair.e1
                        idDrug2 = pair.e2
                        type = pair.type

        return [ddi, idDrug1, idDrug2, type]



    # Following some guidelines from this table https://www.hindawi.com/journals/cmmm/2015/913489/tab1/
    def get_featured_tuple(self, index, tagged_words, bio_tag):
        features = [bio_tag]
        word = tagged_words[index][0]

        # get array of [word,pos_tag] for +-2 word window
        if len(tagged_words) > 2:
            windows = get_words_window(index, tagged_words, 2)
            features.extend(windows)

        # add boolean as length is more >= 7
        features.append(int(len(word) >= 7))

        orthographical_feature = get_orthographical_feature(word)
        features.append(orthographical_feature)

        # Prefix and suffix is of lengths 3,4,5 respectively
        prefix_suffix_features = get_prefix_suffix_feature(word)
        features.extend(prefix_suffix_features)

        # General word shape and brief word shape
        word_shapes = get_word_shapes(word)
        features.extend(word_shapes)

        # May be add Y,N if drug is in drugbank or FDA approved list of drugs?
        return features

# Getting words and pos tags of window +/- n
# return will be [word-n,pos_tag-n,.....word+n,pos_tag+n]
def get_words_window(index, tagged_words, n):
    windows = []
    if n >= len(tagged_words):
        raise ValueError("n must be less than length of tagged_words")

    for i in range(-n,n+1):
        # we can reach the first and last element, so we are safe to get them
        if index + i >= 0 and index + i < len(tagged_words):
            word = tagged_words[index + i][0]
            pos_tag = tagged_words[index + i][1]
        else:
            word = ''
            pos_tag = ''

        windows.append(word)
        windows.append(pos_tag)
    return windows

def get_orthographical_feature(word):
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

    return orthographical_feature

def get_prefix_suffix_feature(word):
    snowball_stemmer = SnowballStemmer("english")
    stemmed_word = snowball_stemmer.stem(word)
    ind = word.find(stemmed_word)

    prefix_len = len(word[:ind])
    suffix_len = len(word) - prefix_len - len(stemmed_word)

    pl3 = int(prefix_len == 3); sufl3 = int(suffix_len == 3)
    pl4 = int(prefix_len == 4); sufl4 = int(suffix_len == 4)
    pl5 = int(prefix_len == 5); sufl5 = int(suffix_len == 5)

    return (pl3, pl4, pl5, sufl3, sufl4, sufl5)

def get_word_shapes(word):
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
    # We assume ascii unicode, which is true since our XML has UTF-8 encoding (English text)
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

    return (word_shape, word_shape_brief)

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
