import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from  nltk import probability

import warnings
warnings.filterwarnings("ignore")

class nGramModel():


    
    def __init__(self, TaggedSentences, estimator, n = 2):
        self.n = n
        self.models = []
        print("Starting the training of the model")
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
            trigram_tuples = (item for sublist in TaggedSentences for item in sublist)
            tagged_trigram_tuples = [(t1.lower(), t2) for t1, t2 in trigram_tuples if (t1.isalpha() or t1 in("<s>","</s>"))]
            triGrams = nltk.ngrams(tagged_trigram_tuples,3)
            cfdTriGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3) in list(triGrams):
                cfdTriGrams[(gram1,gram2)][gram3] +=1
            self.models.append(ConditionalProbDist(cfdTriGrams, estimator))
            self.model = self.models[1]

        if n > 3: #quadGrams
            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples if (t1.isalpha() or t1 in("<s>","</s>"))]
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

def printTags(sent):
    tokenized = nltk.word_tokenize(sent.lower())
    print(tokenized)
    tagged = nltk.pos_tag(tokenized, tagset='universal')
    print(tagged)

def generateSentenceFromModel(model, smoothingName,count):
    print("-----------------Starting sentence generations-----------------")
    print("GramCount:"+ str(model.n));
    print("Smoothing:"+ smoothingName);
    model.createSentences(count)
    print("-----------------Ending sentence generations-----------------")

TaggedCorpus = [w for w in brown.tagged_sents(tagset='universal')[:1000]]
model1 = nGramModel(TaggedCorpus, probability.MLEProbDist, 4)
#model3 = nGramModel(TaggedCorpus, probability.LaplaceProbDist, 4)
#model4 = nGramModel(TaggedCorpus, probability.ELEProbDist, 4)
#model7 = nGramModel(TaggedCorpus, probability.SimpleGoodTuringProbDist, 4)

generateSentenceFromModel(model1,"MLEProbDist",20)
#generateSentenceFromModel(model3,"LaplaceProbDist",20)
#generateSentenceFromModel(model4,"ELEProbDist",20)
#generateSentenceFromModel(model7,"SimpleGoodTuringProbDist",20)

