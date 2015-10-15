import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk import probability
from math import log

import warnings
warnings.filterwarnings("ignore")

class utils:
    @staticmethod
    def generateSentenceFromModel(M, smoothingName,count):
        """
        Prints out and generates a certain count of sentences from a given model
        :param M: The nGram model
        :param smoothingName: The name of the smoothing technique that was used to train the model
        :param count: How many sentences should be printed out
        """
        print("-----------------Starting sentence generations-----------------")
        print("GramCount:"+ str(M.n));
        print("Smoothing:"+ smoothingName);
        for _ in range(count):
            print(M.createSentences(count))
        print("-----------------Ending sentence generations-----------------")

    @staticmethod
    def IsSentenceInBrownCorpus(iSent):
        iSent = iSent.lower()
        sent = set([w.lower() for w in iSent.split()])
        for i, s in enumerate(brown_sets):
            if sent.issubset(s):
                brown_sent = " ".join(brown.sents()[i]).lower()
                if iSent in brown_sent:
                    return True
        return False

    @staticmethod
    def createSentenceFile(sents, smoothingName,fileName):
        for sent in sents:
            f = open(fileName,'a')
            f.write("%s\t%s\t%s\t%.02f\n" % ("SmoothOperator", smoothingName, sent[0], sent[1]))
        f.close()

    @staticmethod
    def generateTrainAndTestSets(tCorpus, ratio):
        """
        Splits a given dataset into test and training sets
        :param tCorpus: The corpus of tagged sentences
        :return: traning and test sets
        """
        split = int((ratio*10)*len(tCorpus)//10)
        return tCorpus[:split], tCorpus[split:] #train, test

    @staticmethod
    def bestSentencesByPerplexity(M, testSet):
        ListOfSents = []
        for sent in testSet:
            ListOfSents.append((sent,M.entropyOfSentence(sent)))
        sorted = sorted(ListOfSents,key=lambda x: x[1])
        return  sorted #Sorts the list in ascending order

    @staticmethod
    def generateSentencesWithPerplexity(M, count):
        print("Starting the sentence generation")
        ListOfSents = []
        while len(ListOfSents) < 5:
            taggedSent, sent = M.createSentenceWithTags()
            perplexity = M.perplexity(taggedSent)
            if not utils.IsSentenceInBrownCorpus(sent):
                if 5 < len(sent.split(" ")) < 14:
                    print("Adding sentence: " + str(len(ListOfSents)+1))
                    ListOfSents.append((sent,perplexity))
        return sorted(ListOfSents,key=lambda x: x[1])[:count] #Sorts the list in descending order

    @staticmethod
    def parseArguments():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--smoothing", help="The smoothing method to be used")
        parser.add_argument("-g", "--gramCount", help="N in the n-Gram model", type=int)
        parser.add_argument("-f", "--trainingSetFractionSize", help="Set the fraction size of the training model", type=float)
        args = parser.parse_args()

        if args.smoothing == 'MLEProbDist':
            sm = probability.MLEProbDist
        elif args.smoothing == 'SimpleGoodTuringProbDist':
            sm = probability.SimpleGoodTuringProbDist
        elif args.smoothing == 'LaplaceProbDist':
            sm = probability.LaplaceProbDist
        elif args.smoothing == 'ELEProbDist':
            sm = probability.ELEProbDist
        else:
            sm = probability.MLEProbDist

        gr = default_gramCount if args.gramCount is None else args.gramCount
        tr = default_trainingSetFractionSize if args.trainingSetFractionSize is None else args.trainingSetFractionSize

        return gr,sm,tr


class nGramModel:

    def __init__(self, TaggedSentences, smoothing, n = 2):
        """
        Trains the model. Trains all models from n to 2
        :param TaggedSentences: List of tagged sentences
        :param smoothing: Smoothing technique
        :param n: gram count
        """
        self.n = n
        self.models = []
        if n > 1: ##bigrams

            self.addPseudo(TaggedSentences,1)
            tagged_bigram_tuples = self.listToTuples(TaggedSentences)
            biGrams = nltk.ngrams(tagged_bigram_tuples,2)
            cfdBiGrams = ConditionalFreqDist()
            for (gram1,gram2) in list(biGrams):
                cfdBiGrams[gram1][gram2] +=1
            self.models.append(ConditionalProbDist(cfdBiGrams, smoothing,len(cfdBiGrams)))
            self.model = self.models[0]

        if n > 2: #triGrams
            self.addPseudo(TaggedSentences,1)
            tagged_trigram_tuples = self.listToTuples(TaggedSentences)
            triGrams = nltk.trigrams(tagged_trigram_tuples)
            cfdTriGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3) in list(triGrams):
                cfdTriGrams[(gram1,gram2)][gram3] +=1
            self.models.append(ConditionalProbDist(cfdTriGrams, smoothing,len(cfdTriGrams)))
            self.model = self.models[1]
        if n > 3: #quadGrams
            self.addPseudo(TaggedSentences,1)
            TaggedTuples = self.listToTuples(TaggedSentences)
            quadGrams = nltk.ngrams(TaggedTuples,4)
            cfdQuadGrams = ConditionalFreqDist()
            for (gram1,gram2,gram3,gram4) in list(quadGrams):
                cfdQuadGrams[(gram1,gram2,gram3)][gram4] +=1
            self.models.append(ConditionalProbDist(cfdQuadGrams, smoothing,len(cfdQuadGrams)))
            self.model = self.models[2]
        print("Done training the model, moving onto more important things!")


    def addPseudo(self, TaggedSentences,count):
        """
        Adds start and end sentence pseudo tokens in the grams
        :param TaggedSentences: List of tagged sentences
        :param count: How many tokens we should add
        """
        for sent in TaggedSentences:
            self.addPseudoToSentence(sent,count)

    def addPseudoToSentence(self, sent, count):
        for i in range(count):
            sent.insert(0,("<s>","<s>"))
            sent.append(("</s>","</s>"))

    def listToTuples(self, sents):
        """
        Utility functions that returns the given list as a set of tuples
        :param sents: A list of sentences
        :return: The input list as a set of tuples
        """
        t = (item for sublist in sents for item in sublist)
        t = [(t1.lower(), t2) for t1, t2 in t if (t1.isalpha() or t1 in("<s>","</s>"))]
        return t

    def prob(self, word, context):
        """
        Returns the likelyhood of a word given it's context
        :param word: Word to check the probability of
        :param context: The context, in bigrams that would be the previous word
        :return: The likelyhood if this word give it's context
        """
        return self.model[context].prob(word)


    def generate(self,context, withTags = False):
        """
        A parent functions that returns the outcome from the recursive function
        :param context: The sentence up to this point.
        :return: The most probable next word
        """
        if withTags:
            return self.generateRec(context,self.n), self.generateRec(context,self.n)[0]
        else:
            return self.generateRec(context,self.n)[0]

    def generateRec(self, context,curNGram):
        """
        Recursive function that generates the next word given the context.
        If it does not find a word within the current nGram it falls back to a n-1Gram model
        :param context: The sentence up this point
        :param curNGram: Recursive parameter that accounts for which nGram model to find word from
        :return: The word that was most probable
        """
        if curNGram == 1:
            return ("</s>","</s>")
        tokenized = nltk.word_tokenize(context.lower())
        tagged = nltk.pos_tag(tokenized, tagset=tagSet)
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

    def createSentenceWithTags(self):
        """
        Creates sentences and return them with tags
        """
        taggedSent = []
        text = ""
        tk = ""
        while tk != "</s>":
            text += tk + " "
            tagged,tk = self.generate(text.strip(),True)
            taggedSent.append(tagged)
        return taggedSent, text.strip().capitalize()

    def entropyOfSentence(self, sentence):
        """
        Function that runs through all words in the sentence and calculates the entropy
        :param sentence: The sentence in clear text
        :return: the entropy of a given sentence
        """
        p = 0
        counter = 0;
        self.addPseudoToSentence(sentence,self.n-1)
        grams = nltk.ngrams(sentence,self.n)
        for gram in grams:
            context =  tuple(gram)[:self.n-1]
            word =  tuple(gram)[self.n-1]
            prob = self.prob(word,context)
            counter += 1
            if prob > 0:
                p += -log(prob)
        return p/counter

    def perplexity(self, sentence):
        """
        Calculates the perplexity of the model given a test set
        :param testSet: test set
        :return: perplexity
        """
        e = self.entropyOfSentence(sentence)
        return 2**e


##PLEASE CHANGE THE FOLLOWING VALUES IF YOU WANT
tagSet = "universal"
default_gramCount = 3
default_trainingSetFractionSize = 0.7
default_smoothingTechnique = probability.MLEProbDist
#STOP EDITING HERE :D


def main():

    gramCount, smoothingTechnique,trainingSetFractionSize = utils.parseArguments()


    print("Loading the corpus!")
    TaggedSent = [w for w in brown.tagged_sents(tagset=tagSet)]
    global brown_sets
    brown_sets = [set([i.lower() for i in s]) for s in brown.sents()]
    print("Starting the training of the model with the following parameters")
    print("N: " + str(gramCount))
    print("Smoothing: " + smoothingTechnique.__name__)
    print("trainingSetFractionSize: " + str(trainingSetFractionSize))

    train, test = utils.generateTrainAndTestSets(TaggedSent,trainingSetFractionSize);
    Mo = nGramModel(train, smoothingTechnique, gramCount)
    genSentences = utils.generateSentencesWithPerplexity(Mo,10)
    utils.createSentenceFile(genSentences,smoothingTechnique.__name__,smoothingTechnique.__name__+".txt")

if __name__ == "__main__":
   main()