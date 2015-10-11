from infGrammarNgram import nGramModel
from nltk.corpus import brown
from nltk.probability import ELEProbDist
from nltk.probability import SimpleGoodTuringProbDist
from math import log
import nltk

def main():

    ##NGRAM MODEL FOR GRAMMAR
    sents = brown.tagged_sents(categories='news', tagset='universal')
    sentences_of_tags = []
    #Pull out the tags and make sentences of just tags!
    for sentence in sents:
        sentence_tags = [tag for (word, tag) in sentence]
        sentences_of_tags.append(sentence_tags)
    testModelGrammar = generateModelFromSentences(sentences_of_tags, ELEProbDist, 3) #Create trigram of only grammar

    ##NGRAM MODEL FOR TAGS AND WORDS
    testModelwordtags = generateModelFromSentences(sents, ELEProbDist, 3, True)


    ## HERE BE DEBUGGING
    print(testModelwordtags.tagged)


def generateModelFromSentences(sents, smoothingF, n, isTagged=False):
    if isTagged:
        addPseudo(sents, n, True)
        return nGramModel(sents, smoothingF, n, True)
    else:
        sents = [[w.lower() for w in s if w.isalnum()] for s in sents]
        addPseudo(sents,n)
        return nGramModel(sents, smoothingF, n)

def entropy(model, test):
    p = 0
    n = 0
    for sent in test:
        n += len(sent)
        wPrev = sent.pop(0)
        for w in sent:
            p += -log(model.prob(w,wPrev),2)
    return p/n

def perplexity(model, test):
    e = entropy(model, test)
    return 2**e

#mutator
def addPseudo(sents, n, tag=False):
    """Modify sents by inserting start and end tokens to beginning and end of each sentence"""
    if tag:
        start_symbol = ('<s>', 'START')
        end_symbol = ('</s>', 'END')
        for s in sents:
            for _ in range(n-1):
                s.insert(0, start_symbol)
                s.append(end_symbol)
    else:
        for s in sents:
            for _ in range(n-1):
                s.insert(0,'<s>')
                s.append('</s>')

def infGrammarGenerate(grammar_model, word_tag_model, nrSents):
    #TODO: Pass in word tag pairs bigram as word model
    """Generate a given number of sentences using a model for grammar and a (word,tag) model of equal N"""
    cfd = nltk.ConditionalFreqDist(word_tag_model)
    #Create an empty list to hold our conditional tokens
    gram_prevTk = list()
    grammar_order = grammar_model.getOrder()

    for _ in range(nrSents): #Generate a sentence, nrSents times
        text = "" #Initialize empty string
        for _ in range(grammar_order-1): #Generate sentences on a given word/symbol (here it is the start symbol)
            gram_prevTk.append("<s>")
        gram_tk = "" #Initialize empty token string
        while(gram_tk != "</s>"): #Loop until we find an END token
            text += gram_tk + " "
            # print("Text: " + text) #debug
            gram_tk = grammar_model.generate(list(gram_prevTk))
            # print("gram_tk: " + str(gram_tk)) #debug
            print("gram_prevTk: " + str(gram_prevTk)) #debug
            tag = gram_prevTk.pop(0)
            # cfd[tag][tag]
            gram_prevTk.append(gram_tk)
            # tag = tag.upper()
            # if tag != "<S>":
            #     # print("Looking up tag: " + str(tag)) #debug
            #     word_list = []
            #     for (a,b) in word_model:
            #         if a[1] == tag:
            #             word_list.append(a[0])
            #
            #     fdist = nltk.FreqDist(word_list)
            #     print(fdist.max())
            #
            #     word_preceders = [a[1] for (a,b) in word_model if b[1] == tag.upper()]
            #     fdist = nltk.FreqDist(word_preceders)
            #     most_common = fdist.most_common()
            #     # most_common = [tag for (tag, _), in fdist.most_common()]
            #     print("word preceders: " + str(most_common))
            #
            # gram_prevTk.append(gram_tk)
            # gram_prevTk = list()
            # noun_preceders =
            # print(noun_preceders)
            # word_tk = word_model.generate(gram_tk)




def generateText(model, sents):
    prevTk = list()
    lN = model.getOrder()
    for _ in range(sents):
        text = ""
        for _ in range(lN-1):
            prevTk.append("<s>")
        tk = ""
        while(tk != "</s>"):
            text += tk + " "
            tk = model.generate(list(prevTk))
            print("tk: " + str(tk))
            prevTk.pop(0)
            prevTk.append(tk)

        print(text.strip())
        prevTk = list()

if __name__ == '__main__':
    main()
