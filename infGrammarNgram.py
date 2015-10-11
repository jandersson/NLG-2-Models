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
        
        #print(self.grams[:100])

        self.countNgrams(self.grams, n, estimator) # sets self.model

        print("NO ERRORS!!!!")

        '''
        if n == 2:
            for (w1,w2) in list(grams):
                cfd[w1][w2] +=1
        

        if n == 3:
            for (w1,w2,w3) in list(grams):
                cfd[(w1,w2)][w3] +=1
                cfd[w1][w2] [w3] += 1

                #cfd[w1+ " " +w2][w3] +=1
        if n == 4:
            for (w1,w2,w3,w4) in list(grams):

                cfd[(w1,w2,w3)][w4]

                cfd[w1+ " " +w2+ " " +w3][w4] +=1
        '''

        #self.model = ConditionalProbDist(cfd, estimator)


    def countNgrams(self, grams, n, smoothingF, isTagged=False):

        if isTagged:
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
                        thisDict = thisDict[wn] ## returns dict.
                        thisN -= 1

                    elif (thisN == 3):
                        w3 = item.pop(0)
                        if w3 not in thisDict:
                            thisDict[w3] = ConditionalFreqDist()

                        thisDict = thisDict[w3]
                        thisN -= 1

                    elif (thisN == 2):
                        w2 = item.pop(0)
                        thisDict[w2][item.pop(0)] += 1
                        thisN -= 1

            self.probNGrams(n,self.model,smoothingF)
            return
        else:
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
                        thisDict = thisDict[wn] ## returns dict.
                        thisN -= 1

                    elif (thisN == 3):
                        w3 = item.pop(0)
                        if w3 not in thisDict:
                            thisDict[w3] = ConditionalFreqDist()

                        thisDict = thisDict[w3]
                        thisN -= 1

                    elif (thisN == 2):
                        w2 = item.pop(0)
                        thisDict[w2][item.pop(0)] += 1
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


def rec_dd():
    return collections.defaultdict(rec_dd)
'''
sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:10000]]
adventureModel = nGramModel(sentences,MLEProbDist,4)

testSentence = ["<s>","i", "can", "like", "apples","</s>"]
#print(adventureModel.probSentence(testSentence))

print(adventureModel.estimateWord('in the car',5))
'''
