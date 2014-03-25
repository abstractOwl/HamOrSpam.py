#!/usr/bin/env python

from naivebayes import NaiveBayes
import sys

def main(args):
    """ HamOrSpam entrypoint """

    # Get arguments
    try:
        script, train, test = args
    except ValueError:
        print 'usage: python hamorspam.py train.txt test.txt'
        sys.exit(-1)

    classifier = NaiveBayes(['ham', 'spam'])

    # Train classifier
    train_file = open(train, 'r')
    for line in train_file:
        # Discard empty lines
        if not line.strip():
            continue
        typ, message = line.split('\t', 1)
        classifier.teach(typ, message)
    train_file.close()

    # Query classifier
    test_file = open(test, 'r')
    for line in test_file:
        # Discard empty lines
        if not line.strip():
            continue
        print classifier.classify(line)
    test_file.close()

if __name__ == '__main__':
    main(sys.argv)
