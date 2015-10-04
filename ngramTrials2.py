import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist

import warnings
warnings.filterwarnings("ignore")

class nGramModel():


    
    def __init__(self, samples, estimator, n = 2):
        self.n = n
        self.models = []
        TaggedSentences = [w for w in brown.tagged_sents(tagset='universal')[:1000]]

        if n > 1: ##bigrams

            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples]
            biGrams = nltk.ngrams(TaggedTuples,2)
            cfdBiGrams = ConditionalFreqDist()
            for (gram1,gram2) in list(biGrams):
                cfdBiGrams[gram1][gram2] +=1
            self.models.append(ConditionalProbDist(cfdBiGrams, estimator))
            self.model = self.models[0]

        if n > 2: #triGrams
            self.addPseudo(TaggedSentences,1)
            trigram_tuples = (item for sublist in TaggedSentences for item in sublist)
            tagged_trigram_tuples = [(t1.lower(), t2) for t1, t2 in trigram_tuples]
            triGrams = nltk.ngrams(tagged_trigram_tuples,3)
            cfdTriGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3) in list(triGrams):
                cfdTriGrams[(gram1,gram2)][gram3] +=1
            self.models.append(ConditionalProbDist(cfdTriGrams, estimator))
            self.model = self.models[1]

        if n > 3: #quadGrams
            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples]
            quadGrams = nltk.ngrams(TaggedTuples,4)
            cfdQuadGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3,gram4) in list(quadGrams):
                cfdQuadGrams[(gram1,gram2,gram3)][gram4] +=1
            self.models.append(ConditionalProbDist(cfdQuadGrams, estimator))
            self.model = self.models[2]


    def estimateWord(self, context, numberOfEstimates):
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset='universal')

        if self.n == 3:
            biGram = list(nltk.ngrams(tagged, 2))[0]
            wordEstimates = {}
            print(self.model.generate(biGram))
            for value in self.model[biGram].freqdist():
                wordEstimates[value] = self.model[biGram].prob(value)
            orderedEstimates = ((k, wordEstimates[k]) for k in sorted(wordEstimates, key=wordEstimates.get, reverse=True))
            print([(k,v) for k, v in orderedEstimates][:numberOfEstimates])
            return

    def addPseudo(self, TaggedSentences,count):
        for sent in TaggedSentences:
            for i in range(count):
                sent.insert(0,("<s>","<s>"))
                sent.append(("</s>","</s>"))


    def prob(self, word, previous):
        return self.model[previous].prob(word)

    def generate(self, context):
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset='universal')

        if len(tagged) < self.n:
            for i in range(self.n -len(tagged)-1):
                tagged.insert(0, ("<s>","<s>"))
        tagged = tagged[-(self.n - 1):]
        grams =  list(nltk.ngrams(tagged, self.n-1))[0][0] if self.n == 2  else list(nltk.ngrams(tagged, self.n-1))[0]
        ret = ""
        try:
            ret = self.model[grams].generate()
        except:
            ret = ""
        return ret

    def probSentence(self, sentence):
        prob = 1
        for i in range(len(sentence)-1):
            prob *= self.model[sentence[i]].prob(sentence[i+1])
        return prob

model = nGramModel([()], MLEProbDist, 3)
# print(model.generate("problem"))
print(model.generate("the man"))
print(model.generate("the"))
print(model.generate("horse"))

# ('in', 'IN'), ('the', 'AT')
# ('which', 'WDT'), ('was', 'BEDZ')

      #
      # if source == 'wsj':
      #       source = 'en-ptb'
      #   if source == 'brown':
      #       source = 'en-brown'


'''
sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:10000]]
adventureModel = nGramModel(sentences,MLEProbDist,4)

testSentence = ["<s>","i", "can", "like", "apples","</s>"]
#print(adventureModel.probSentence(testSentence))

print(adventureModel.estimateWord('in the car',5))
'''
