import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist
import collections

class nGramModel():

    def __init__(self, samples, estimator, n = 2):
        cfd = ConditionalFreqDist()
        for sent in samples:
            sent.insert(0,'<s>')
            sent.append('</s>')
            grams = nltk.ngrams(sent,n)
            if n == 2:
                for (w1,w2) in list(grams):
                    cfd[w1][w2] +=1
            if n == 3:
                for (w1,w2,w3) in list(grams):
                    cfd[w1+ " " +w2][w3] +=1
            if n == 4:
                for (w1,w2,w3,w4) in list(grams):
                    cfd[w1+ " " +w2+ " " +w3][w4] +=1
            if n == 5:
                for (w1,w2,w3,w4,w5) in list(grams):
                    cfd[w1+ " " +w2+ " " +w3+ " " +w4][w5] +=1
        self.model = ConditionalProbDist(cfd, estimator)

    def estimateWord(self, word, numberOfEstimates):
        wordEstimates = {}
        for key in self.model[word].freqdist():
            print(key + ": " + repr(self.model[word].prob(key)))
            wordEstimates[key] = self.model[word].prob(key)
        orderedEstimates = ((k, wordEstimates[k]) for k in sorted(wordEstimates, key=wordEstimates.get, reverse=True))
        return [(k,v) for k, v in orderedEstimates][:numberOfEstimates]

    def prob(self, word, previous):
        return self.model[previous][word]

    def probSentence(self, sentence):
        prob = 1
        for i in range(len(sentence)-1):
            #print(self.model[sentence[i]].prob(sentence[i+1]))
            prob *= self.model[sentence[i]].prob(sentence[i+1])
        return prob


sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:10000]]
adventureModel = nGramModel(sentences,MLEProbDist,4)

testSentence = ["<s>","i", "can", "like", "apples","</s>"]
#print(adventureModel.probSentence(testSentence))

print(adventureModel.estimateWord('in the car',5))
