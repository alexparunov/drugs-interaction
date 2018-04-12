#!/usr/bin/python3
from parser import main as parse_all_files

def main():
    parse_all_files()
    from feature_extraction import main as extract_features
    extract_features()

if __name__ == "__main__":
    main()
