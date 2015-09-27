from ngramTrials import nGramModel
from nltk.corpus import brown
from nltk.probability import ELEProbDist
from math import log

def main():
    sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')]
    rSentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='religion')]

    addPseudo(rSentences)
    addPseudo(sentences) # Add pseudowords
    split = 9*len(sentences)//10
    train = sentences[:split]
    test = sentences[split:]


    test2 = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='religion')][:len(test)]
    addPseudo(test2) # Add pseudowords

    # Train models
    adventureModel = nGramModel(train,ELEProbDist)
    religionModel = nGramModel(rSentences,ELEProbDist)

    p = perplexity(adventureModel,test)
    p2 = perplexity(adventureModel,test2)

    # Generate cool shit
    generateText(adventureModel, 5)
    print()
    generateText(religionModel, 5)
    #print(p, p2)



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
def addPseudo(sents):
    for s in sents:
        s.insert(0,'<s>')
        s.append('</s>')

def generateText(model, sents):
    for _ in range(sents):
        text = ""
        prevTk = "<s>"
        tk = ""
        while(tk != "</s>"):
            text += tk + " "
            tk = model.generate(prevTk)
            prevTk = tk
        print(text.strip())

if __name__ == '__main__':
    main()
