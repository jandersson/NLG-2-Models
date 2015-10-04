import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist

import warnings
warnings.filterwarnings("ignore")

class nGramModel():


    
    def __init__(self, TaggedSentences, estimator, n = 2):
        self.n = n
        self.models = []

        if n > 1: ##bigrams

            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples if t1.isalnum()]
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

    def addPseudo(self, TaggedSentences,count):
        for sent in TaggedSentences:
            for i in range(count):
                sent.insert(0,("<s>","<s>"))
                sent.append(("</s>","</s>"))

    def generateOld(self, context):
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

    def generateRec(self, context, curNGram = -1):
        if curNGram == -1:
            curNGram = self.n
        if curNGram == 1:
            return ""
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset='universal')

        if len(tagged) < curNGram:
            for i in range(curNGram -len(tagged)-1):
                tagged.insert(0, ("<s>","<s>"))
        tagged = tagged[-(curNGram - 1):]
        grams =  list(nltk.ngrams(tagged, curNGram-1))[0][0] if curNGram == 2  else list(nltk.ngrams(tagged, curNGram-1))[0]
        ret = ""
        try:
            ret = self.models[curNGram-2][grams].generate()
        except:
            ret = self.generateRec(context, curNGram-1)
        return ret


TaggedCorpus = [w for w in brown.tagged_sents(tagset='universal')[:10000]]
model = nGramModel(TaggedCorpus, MLEProbDist, 4)
# print(model.generate("problem"))
print(model.generateRec("the man who won the man"))
print(model.generateRec("the"))
print(model.generateRec("horse"))

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
