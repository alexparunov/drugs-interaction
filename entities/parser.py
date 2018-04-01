#!/usr/bin/python3
from xml_classes import *
import xml.etree.ElementTree as ET
from os.path import abspath, join, isdir
from os import listdir, makedirs
import sys
import pickle

# XML files
train_path = abspath("../data/train/DrugBank")
drug_bank_train = [join(train_path, f) for f in listdir(train_path)]

train_path = abspath("../data/train/MedLine")
medline_train =   [join(train_path, f) for f in listdir(train_path)]

# Test for DDI extraction task

test_path = abspath("../data/test/Test_DDI_Extraction_task/DrugBank")
drug_bank_ddi_test = [join(test_path, f) for f in listdir(test_path)]
test_path = abspath("../data/test/Test_DDI_Extraction_task/MedLine")
medline_ddi_test =   [join(test_path, f) for f in listdir(test_path)]

# Test for DrugNER task
test_path = abspath("../data/test/Test_DrugNER_task/DrugBank")
drug_bank_ner_test = [join(test_path, f) for f in listdir(test_path)]
test_path = abspath("../data/test/Test_DrugNER_task/MedLine")
medline_ner_test =   [join(test_path, f) for f in listdir(test_path)]

class Parser:
    def set_path(self, xml_path):
        self.path = xml_path

    def parse_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        document = Document(root.attrib['id'])
        for child in root:
            if child.tag == "sentence":
                sentence = Sentence(child.attrib['id'], child.attrib['text'])
                if len(sentence.text) < 2:
                    continue
                for second_child in child:
                    attr = second_child.attrib
                    if second_child.tag == "entity":
                        entity = Entity(attr['id'], attr['charOffset'], attr['type'], attr['text'])
                        sentence.add_entity(entity)
                    elif second_child.tag == "pair":
                        ddi = False
                        if attr['ddi'] == "true":
                            ddi = True

                        pair = Pair(attr['id'],attr['e1'],attr['e2'], ddi)
                        if pair.ddi and 'type' in attr:
                            pair.set_type(attr['type'])

                        sentence.add_pair(pair)

                document.add_sentence(sentence)
        return document

    def parse_save_drug_bank_train(self):
        parsed_docs = []
        for doc in drug_bank_train:
            print("Parsing: "+doc)
            self.set_path(doc)
            d = self.parse_xml()
            parsed_docs.append(d)

        dir_path = abspath("../data/pickle")
        if not isdir(dir_path):
            makedirs(dir_path)

        with open(join(dir_path, "drug_bank_train.pkl"),"wb") as f:
            pickle.dump(parsed_docs, f)
            print("Saved DrugBank parsed documents into pickle.")

def main():
    parser = Parser()
    parser.parse_save_drug_bank_train()


if __name__ == "__main__":
    main()
