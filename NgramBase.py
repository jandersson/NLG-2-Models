import abc

class NgramBase():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, train, estimator):
        """ Must implement internal method for training the model """
        return

    @abc.abstractmethod
    def prob(self, word, context):
        """ Must implement method for returning probability of word given context """
        return


    def perplexity(self):
        return "it always works"