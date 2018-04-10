#!/usr/bin/python3
from data.parser import main as parseAllFiles

def main():
    parseAllFiles()
    from data.feature_extraction import main as extractFeatures
    extractFeatures()

if __name__ == "__main__":
    main()
