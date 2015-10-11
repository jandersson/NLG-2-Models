from infGrammarNgram import nGramModel
from nltk.corpus import brown
from nltk.probability import ELEProbDist
from math import log

def main():
    testModel = generateModelFromSentences(brown.sents(categories='adventure'), ELEProbDist, 3)
    #p = perplexity(adventureModel,test)
    #p2 = perplexity(adventureModel,test2)
    # Generate cool shit
    generateText(testModel, 5)
    #generateText(religionModel, 5)
    #print(p, p2)


def generateModelFromSentences(sents, smoothingF, n):
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
def addPseudo(sents, n):
    for s in sents:
        for _ in range(n-1):
            s.insert(0,'<s>')    
            s.append('</s>')

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
            prevTk.pop(0)
            prevTk.append(tk)

        print(text.strip())
        prevTk = list()

if __name__ == '__main__':
    main()
