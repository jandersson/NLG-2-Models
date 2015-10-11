import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from  nltk import probability
from math import log

import warnings
warnings.filterwarnings("ignore")

class utils:
    @staticmethod
    def generateSentenceFromModel(model, smoothingName,count):
        print("-----------------Starting sentence generations-----------------")
        print("GramCount:"+ str(model.n));
        print("Smoothing:"+ smoothingName);
        model.createSentences(count)
        print("-----------------Ending sentence generations-----------------")

    @staticmethod
    def generateTrainAndTestSets(tCorpus):
        split = 9*len(tCorpus)//10
        return tCorpus[:split], tCorpus[split:] #train, test


class nGramModel():

    def __init__(self, TaggedSentences, estimator, n = 2):
        self.n = n
        self.models = []
        print("Training the model")
        if n > 1: ##bigrams

            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples if (t1.isalpha() or t1 in("<s>","</s>"))]
            biGrams = nltk.ngrams(TaggedTuples,2)
            cfdBiGrams = ConditionalFreqDist()
            for (gram1,gram2) in list(biGrams):
                cfdBiGrams[gram1][gram2] +=1
            self.models.append(ConditionalProbDist(cfdBiGrams, estimator))
            self.model = self.models[0]

        if n > 2: #triGrams
            self.addPseudo(TaggedSentences,1)
            tagged_trigram_tuples = self.listToTuples(TaggedSentences)
            triGrams = nltk.ngrams(tagged_trigram_tuples,3)
            cfdTriGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3) in list(triGrams):
                cfdTriGrams[(gram1,gram2)][gram3] +=1
            self.models.append(ConditionalProbDist(cfdTriGrams, estimator))
            self.model = self.models[1]

        if n > 3: #quadGrams
            self.addPseudo(TaggedSentences,1)
            TaggedTuples = self.listToTuples(TaggedSentences)
            quadGrams = nltk.ngrams(TaggedTuples,4)
            cfdQuadGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3,gram4) in list(quadGrams):
                cfdQuadGrams[(gram1,gram2,gram3)][gram4] +=1
            self.models.append(ConditionalProbDist(cfdQuadGrams, estimator))
            self.model = self.models[2]
        print("Done training the model, moving onto more important things!")

    def addPseudo(self, TaggedSentences,count):
        for sent in TaggedSentences:
            for i in range(count):
                sent.insert(0,("<s>","<s>"))
                sent.append(("</s>","</s>"))

    def listToTuples(self,sents):
        t = (item for sublist in sents for item in sublist)
        t = [(t1.lower(), t2) for t1, t2 in t if (t1.isalpha() or t1 in("<s>","</s>"))]
        return t

    def prob(self, word, context):
        return self.model[context].prob(word)


    def generate(self,context):
        return self.generateRec(context,self.n)[0]

    def generateRec(self, context,curNGram):
        if curNGram == 1:
            return ("</s>","</s>")
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset='universal')
        if len(tagged) < curNGram:
            for i in range(curNGram -len(tagged)):
                tagged.insert(0, ("<s>","<s>"))
        tagged = tagged[-(curNGram - 1):]
        grams =  list(nltk.ngrams(tagged, curNGram-1))[0][0] if curNGram == 2  else list(nltk.ngrams(tagged, curNGram-1))[0]
        try:
            return self.models[curNGram-2][grams].generate()
        except:
            try:
                return self.generateRec(context, curNGram-1)
            except:
                return "</s>"

    def createSentences(self,count = 1):
        for i in range(count):
            text = ""
            tk = ""
            while(tk != "</s>"):
                text += tk + " "
                tk = self.generate(text.strip())
            print(text.strip().capitalize()+".")

    def entropy(self, testSet):
        print("Calculating entropy for a given testSet")
        p = 0
        self.addPseudo(testSet,self.n)
        t = self.listToTuples(testSet)
        grams = nltk.ngrams(t,self.n)
        for gram in grams:
            context =  tuple(gram)[:self.n-1]
            word =  tuple(gram)[self.n-1]
            prob = self.prob(word,context)
            if prob > 0:
                test = -log(prob)
            p += test
        return p/len(t)

    def perplexity(self, testSet):
        e = self.entropy(testSet)
        return 2**e



TaggedCorpus = [w for w in brown.tagged_sents(tagset='universal')]
gramCount = 3
train, test = utils.generateTrainAndTestSets(TaggedCorpus);
#model1 = nGramModel(TaggedCorpus, probability.MLEProbDist, gramCount)
#model3 = nGramModel(TaggedCorpus, probability.LaplaceProbDist, gramCount)
#model4 = nGramModel(TaggedCorpus, probability.ELEProbDist, gramCount)
model7 = nGramModel(train, probability.SimpleGoodTuringProbDist, gramCount)
print(model7.perplexity(test))
#generateSentenceFromModel(model1,"MLEProbDist",20)
#generateSentenceFromModel(model3,"LaplaceProbDist",20)
#generateSentenceFromModel(model4,"ELEProbDist",20)
utils.generateSentenceFromModel(model7,"SimpleGoodTuringProbDist",20)

