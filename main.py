#!/usr/bin/python3
from data.parser import main as parseAllFiles
from data.feature_extraction import main as extractFeatures

def main():
    parseAllFiles()
    extractFeatures()

if __name__ == "__main__":
    main()
