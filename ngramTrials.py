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
        for sent in samples:
            for (w1,w2) in list(zip(sent[:-1],sent[1:])):
                cfd[w1][w2] +=1
        self.model = ConditionalProbDist(cfd, estimator)

    def estimateWord(self, word, numberOfEstimates):
        
        print("Grouping all the bigram by first word")
        wordEstimates = {}
        for key in self.model[word].freqdist():
            #print(self.model[word].prob(key))
            wordEstimates[key] = self.model[word].prob(key)
        orderedEstimates = ((k, wordEstimates[k]) for k in sorted(wordEstimates, key=wordEstimates.get, reverse=True))
        return [(k,v) for k, v in orderedEstimates][:numberOfEstimates]

    def prob(self, word, previous):
        return model[previous][word]

    def probSentence(self, sentence):
        prob = 1
        for i in range(len(sentence)-1):
            #print(self.model[sentence[i]].prob(sentence[i+1]))
            prob *= self.model[sentence[i]].prob(sentence[i+1])
        return prob

    def __repr__(self):
        return this.model

sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:10000]]
adventureModel = nGramModel(sentences,MLEProbDist)

testSentence = ["<s>","i", "can", "like", "apples","</s>"]



print(adventureModel.probSentence(testSentence))
print(adventureModel.estimateWord('like',5))
