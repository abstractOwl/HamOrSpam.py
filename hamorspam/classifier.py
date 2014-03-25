class Classifier(object):
    """ Interface implemented by classes that classify messages """

    def teach(typ, message):
        """ Adds a message to the training set """
        raise NotImplementedError("Not implemented yet.")

    def classify(message):
        """ Classifies a message based on the training set """
        raise NotImplementedError("Not implemented yet.")

