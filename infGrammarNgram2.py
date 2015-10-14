import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist
import collections

class nGramModel():
    
    def __init__(self, samples, estimator, n = 2, isTagged=False):
        self.tagged = isTagged
        self.order = n
        print("my order: " + str(self.order))

        self.grams = list()
        for sent in samples:
            self.grams += list(nltk.ngrams(sent,n)) # get iterator over ngrams
        self.countNgrams(self.grams, n, estimator) # sets self.model


    def countNgrams(self, grams, n, smoothingF):
        if (n > 2):
            self.model = rec_dd()
        else:
            self.model = ConditionalFreqDist()

        for item in list(grams):
            item = list(item)
            # For each new gram
            thisDict = self.model  # point to top-level dictionary
            thisN = n # reset n-counter

            while(thisN >= 2):
                if( thisN > 3):
                    wn = item.pop(0) # pop first word in gram
                    if self.tagged:
                        wn = wn[0] #just get the word
                    thisDict = thisDict[wn] ## returns dict.
                    thisN -= 1

                elif (thisN == 3):
                    w3 = item.pop(0)
                    if self.tagged: # Do another level of dicts
                        w3 = w3[0] #just get the word
                    else: # Make freq-dists
                        if w3 not in thisDict: 
                            thisDict[w3] = ConditionalFreqDist()
                    thisDict = thisDict[w3]
                    thisN -= 1

                elif (thisN == 2):
                    w2 = item.pop(0)
                    if self.tagged:
                        w2 = w2[0] #just get the word
                        if w2 not in thisDict: # Make a freq-dist for tags of w1 under w2
                            thisDict[w2] = ConditionalFreqDist()
                        thisDict = thisDict[w2]
                        w1 = item.pop(0)
                        w1_tag = w1[1]
                        w1_word = w1[0]
                        thisDict[w1_tag][w1_word] += 1
                    else: 
                        w1 = item.pop(0)

                        thisDict[w2][w1] += 1
                        thisN -= 1

        self.probNGrams(n,self.model,smoothingF)
        return

    def probNGrams(self, n, dic, smoothingF):
        if (n == 2): # Special for bigrams
            self.model = ConditionalProbDist(dic, smoothingF)
            return

        else:
            for (k,v) in dic.items():
                if (n == 3):
                    dic[k] = ConditionalProbDist(v, smoothingF)
                else:
                    self.probNGrams(n-1, v, smoothingF)
        return

    def estimateWord(self, word, numberOfEstimates):
        wordEstimates = {}
        for key in self.model[word].freqdist():
            print(key + ": " + repr(self.model[word].prob(key)))
            wordEstimates[key] = self.model[word].prob(key)
        orderedEstimates = ((k, wordEstimates[k]) for k in sorted(wordEstimates, key=wordEstimates.get, reverse=True))
        return [(k,v) for k, v in orderedEstimates][:numberOfEstimates]

    def prob(self, word, previous):
        return self.model[previous].prob(word)


    def generate(self, words):
        order = self.order
        dictionary = self.model
        while (order > 2):
            if (words[0] not in dictionary):
                print("uh oh, back off!")
                return
            dictionary = dictionary[words.pop(0)]
            order -= 1

        probdist = dictionary[words.pop(0)]
        return probdist.generate()

    def probSentence(self, sentence):
        prob = 1
        for i in range(len(sentence)-1):
            #print(self.model[sentence[i]].prob(sentence[i+1]))
            prob *= self.model[sentence[i]].prob(sentence[i+1])
        return prob

    def getOrder(self):
        return self.order

    def prob_tri(self, w1, w2, word):
        return self.model[w1][w2].prob(word)

def rec_dd():
    return collections.defaultdict(rec_dd)
'''
sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:10000]]
adventureModel = nGramModel(sentences,MLEProbDist,4)

testSentence = ["<s>","i", "can", "like", "apples","</s>"]
#print(adventureModel.probSentence(testSentence))

print(adventureModel.estimateWord('in the car',5))
'''
