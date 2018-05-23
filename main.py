#!/usr/bin/python3
import warnings
import argparse

from parser import main as parse_all_files
from classifier import Classifier

parser = argparse.ArgumentParser(description = "Train or Test model")
parser.add_argument('-p','--parse', help = "Parse files", action="store_true")
parser.add_argument('-t','--task', type = int, help = "Task of problem. 1 - NER task, 2 - DDI task.", action = "store", default = -1)
parser.add_argument('--train', help = "Train model", action="store_true")
parser.add_argument('--test', help = "Test model at index i", action="store_true")
parser.add_argument('-f','--folder_index', type = int, help = "Folder number. 1 - drugbank, 2 - medline", action = "store", default = -1)
parser.add_argument('-i','--model_index', type = int, help = "Index of a model to test", action = "store", default = -1)
parser.add_argument('-r','--ratio', type = float, help = "Ratio of data to use for training", action = "store", default = 1)
parser.add_argument('-c','--classifier', type = int, help = "Classifier to use. 1 - SVM, 2 - CRF", action = "store", default = 1)

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        args = parser.parse_args()

        clasf = Classifier()

        if args.parse:
            parse_all_files()
            from feature_extraction import main as extract_features
            extract_features()

        if args.train:
            folder_index = args.folder_index
            ratio = args.ratio
            if ratio <= 0 and ratio > 1:
                ratio = 1
            if folder_index == 1 or folder_index == 2:
                if args.task == 1:
                    clasf.train_NER_model(train_folder = folder_index, ratio = ratio,
                            classifier = args.classifier)
                elif args.task == 2:
                    clasf.train_DDI_model(train_folder = folder_index, ratio = ratio,
                            classifier = args.classifier)
                else:
                    parser.print_help()
            else:
                parser.print_help()
        elif args.test:
            model_index = args.model_index
            folder_index = args.folder_index
            if model_index >= 0 and folder_index >= 1:
                if args.task == 1:
                    clasf.test_NER_model(model_index = model_index, test_folder = folder_index,
                            classifier = args.classifier)
                elif args.task == 2:
                    clasf.test_DDI_model(model_index = model_index, test_folder = folder_index,
                            classifier = args.classifier)
                else:
                    parser.print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
