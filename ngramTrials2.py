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
        print("Starting the training of model")
        if n > 1: ##bigrams

            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples if (t1.isalnum() or t1 in("<s>","</s>"))]
            biGrams = nltk.ngrams(TaggedTuples,2)
            cfdBiGrams = ConditionalFreqDist()
            for (gram1,gram2) in list(biGrams):
                cfdBiGrams[gram1][gram2] +=1
            self.models.append(ConditionalProbDist(cfdBiGrams, estimator))
            self.model = self.models[0]

        if n > 2: #triGrams
            self.addPseudo(TaggedSentences,1)
            trigram_tuples = (item for sublist in TaggedSentences for item in sublist)
            tagged_trigram_tuples = [(t1.lower(), t2) for t1, t2 in trigram_tuples if (t1.isalnum() or t1 in("<s>","</s>"))]
            triGrams = nltk.ngrams(tagged_trigram_tuples,3)
            cfdTriGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3) in list(triGrams):
                cfdTriGrams[(gram1,gram2)][gram3] +=1
            self.models.append(ConditionalProbDist(cfdTriGrams, estimator))
            self.model = self.models[1]

        if n > 3: #quadGrams
            self.addPseudo(TaggedSentences,1)
            TaggedTuples = (item for sublist in TaggedSentences for item in sublist)
            TaggedTuples = [(t1.lower(), t2) for t1, t2 in TaggedTuples if (t1.isalnum() or t1 in("<s>","</s>"))]
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
        print("Finding the most probable word given a certain context")
        return self.generateRec(context,self.n)[0]

    def generateRec(self, context,curNGram):
        if curNGram == 1:
            return ("","")
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset='universal')

        if len(tagged) < curNGram:
            for i in range(curNGram -len(tagged)-1):
                tagged.insert(0, ("<s>","<s>"))
        tagged = tagged[-(curNGram - 1):]
        grams =  list(nltk.ngrams(tagged, curNGram-1))[0][0] if curNGram == 2  else list(nltk.ngrams(tagged, curNGram-1))[0]
        try:
            return self.models[curNGram-2][grams].generate()
        except:
            try:
                return self.generateRec(context, curNGram-1)
            except:
                return ""
        


print("Loading the corpus")
TaggedCorpus = [w for w in brown.tagged_sents(tagset='universal')]
model = nGramModel(TaggedCorpus, MLEProbDist, 4)
# print(model.generate("problem"))
model.generate("the man who won the man")
model.generate("blarg")
model.generate("horse")

print("Starting to generate the sentence")
text = ""
prevTk = "the"
tk = ""
while(tk != "</s>"):
    text += tk + " "
    tk = model.generate(prevTk)
    prevTk = tk
print(text.strip())



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
