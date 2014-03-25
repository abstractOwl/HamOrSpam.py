import re
from collections import defaultdict
from classifier import Classifier

# Omega value for Laplace smoothing
OMEGA = 1

class NaiveBayes(Classifier):
    """ A Classifier that classifies messages using the Naive Bayes technique """

    """ Categories for classification """
    classes = False
    """ Number of times a term has occurred, per type """
    terms   = defaultdict(lambda: defaultdict(int))
    """ Number of messages per type """
    stats   = defaultdict(int)

    def __init__(self, classes):
        self.classes = classes

    def normalize(self, message):
        """ Normalize terms """
        return re.sub('[^0-9a-zA-Z]+', '', message)

    def teach(self, typ, message):
        """ Adds a message to the learning set """
        self.stats[typ] += 1
        for term in message.split(' '):
            self.teach_term(typ, term)

    def teach_term(self, typ, term):
        """ Adds a term to the learning set """
        self.terms[typ][self.normalize(term)] += 1

    def classify(self, message):
        """ Classifies a term based on the learning set """
        max_prob = 0
        max_typ  = None

        # Find maximum probability classification
        for typ in self.classes:
            prob = float(self.stats[typ] + OMEGA) \
                / (self.total_messages() + OMEGA * len(self.stats))

            for term in message.split(' '):
                prob *= self.classify_term(typ, term, len(message.split(' ')))

            if prob > max_prob:
                max_typ  = typ
                max_prob = prob

        return max_typ

    def classify_term(self, typ, term, term_count):
        """
        Returns the classification of a term with respect to the current learning set
        """
        return float(self.terms[typ][self.normalize(term)] + OMEGA) \
                / (self.total_terms(typ) + OMEGA * term_count)

    def total_messages(self):
        """ Returns the total number of messages in the learning set """
        return sum(self.stats.values())

    def total_terms(self, typ):
        """ Returns the total number of terms of a specified classification type """
        return sum(self.terms[typ].values())
