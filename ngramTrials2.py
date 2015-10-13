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
    def bestSentencesByPerplexity(M, testSet,count):
        ListOfSents = []
        for sent in testSet:
            ListOfSents.append((sent,M.entropyOfSentence(sent)))
        sorted = sorted(ListOfSents,key=lambda x: x[1])
        return  sorted #Sorts the list in ascending order

    @staticmethod
    def generateSentencesWithPerplexityThreshold(M, count):
        ListOfSents = []
        while len(ListOfSents) < 200:
            taggedSent, sent = M.createSentenceWithTags()
            perplexity = M.perplexity(taggedSent)
            if 3 < len(sent.split(" ")) < 11:
                print("Adding sentence nr: " + str(len(ListOfSents)))
                ListOfSents.append((sent,perplexity))
        return sorted(ListOfSents,key=lambda x: x[1])[:count] #Sorts the list in descending order


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
        print("Training the model")
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
            triGrams = nltk.ngrams(tagged_trigram_tuples,3)
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

    def createSentences(self,count = 1):
        """
        Creates sentences
        :param count: The number of sentences to generate
        """
        for i in range(count):
            text = ""
            tk = ""
            while tk != "</s>":
                text += tk + " "
                tk = self.generate(text.strip())
        return text.strip().capitalize()

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

    def entropy(self, testSet):
        """
        Calculates the entopy of the model given a testset
        :param testSet: Trainingset
        :return: Entropy
        """
        print("Calculating entropy for a given testSet")
        p = 0
        counter = 0;
        self.addPseudo(testSet,self.n)
        t = self.listToTuples(testSet)
        grams = nltk.ngrams(t,self.n)
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

tagSet = "universal"
TaggedSent = [w for w in brown.tagged_sents(tagset=tagSet)]
gramCount = 3
testSent = TaggedSent
train, test = utils.generateTrainAndTestSets(TaggedSent,0.7);
Mo = nGramModel(train, probability.MLEProbDist, gramCount)
#print(utils.bestSentencesByPerplexity(model,test,10))
genSentences = utils.generateSentencesWithPerplexityThreshold(Mo,10)
utils.createSentenceFile(genSentences,"MLEProbDist","MLEProbDist.txt")




#utils.createSentenceFile(genSentences,"LaplaceProbDist","LaplaceProbDist.txt")
#model.entropyOfSentence()
#print(model7.perplexity(test))
#generateSentenceFromModel(model1,"MLEProbDist",20)
#utils.generateSentenceFromModel(model,"LaplaceProbDist",20)
#generateSentenceFromModel(model4,"ELEProbDist",20)
#utils.generateSentenceFromModel(model7,"SimpleGoodTuringProbDist",20)







'''
from tkinter import *


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):
    counter = 0
    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        #reference to the master widget, which is the tk window
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        createSentenceButton = Button(self, text="Create sentence",command=self.showText)

        # placing the button on my window
        createSentenceButton.place(x=0, y=0)

    def showText(self):
        print("creating sentence")
        text = Label(self, text=model.createSentences(1))
        text.pack()


    def client_exit(self):
        exit()

# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("800x600")

#creation of an instance
app = Window(root)

#mainloop
root.mainloop()
'''