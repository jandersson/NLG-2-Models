from infGrammarNgram import nGramModel
from nltk.corpus import brown
from nltk.probability import ELEProbDist
from nltk.probability import SimpleGoodTuringProbDist
from math import log
import random




def main():
    """
    Provide an entry point into program.
    :return: None
    """
    ##NGRAM MODEL FOR GRAMMAR
    sents_ = brown.tagged_sents(categories='adventure', tagset='universal')
    sents = list(sents_) #needs to be mutable to insert start/end tokens if working with tags
    sentences_of_tags = []
    #Pull out the tags and make sentences of just tags!
    for sentence in sents:
        sentence_tags = [tag for (word, tag) in sentence]
        sentences_of_tags.append(sentence_tags)
    testModelGrammar = generateModelFromSentences(sentences_of_tags, SimpleGoodTuringProbDist, 3) #Create trigram of only grammar

    ##NGRAM MODEL FOR TAGS AND WORDS
    testModelwordtags = generateModelFromSentences(sents, SimpleGoodTuringProbDist, 3, True)

    ## GENERATE TEXT
    infGrammarGenerate(testModelGrammar, testModelwordtags, 10)

    ## HERE BE DEBUGGING
    #TODO: Implement function to split corpus sentences into training and test set.
    training_sents, test_sents = split(sents_, 0.9)
    print("Training Set Length: " + str(len(training_sents)))
    print("Test Set Length: " + str(len(test_sents)))

    ## TESTS
    assert(testModelGrammar.tagged == False)
    assert(testModelwordtags.tagged == True)
    assert('START' in testModelwordtags.model)

def generateModelFromSentences(sents, smoothingF, n, isTagged=False):
    if isTagged:
        addPseudo(sents, n, True)
        return nGramModel(sents, smoothingF, n, True)
    else:
        sents = [[w.lower() for w in s if w.isalnum()] for s in sents]
        addPseudo(sents,n)
        return nGramModel(sents, smoothingF, n)

def entropy(model, test):
    """Calculate entropy given an ngram model and a list of test sentences."""
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
    """Generate a given number of sentences using a model for grammar and a (word,tag) model of equal N"""
    assert(grammar_model.getOrder() == word_tag_model.getOrder())
    #Create an empty list to hold our conditional tokens
    gram_prevTk = list()
    word_prevTk = list()
    grammar_order = grammar_model.getOrder()
    for _ in range(nrSents): #Generate a sentence, nrSents times
        text = "" #Initialize empty string
        for _ in range(grammar_order-1): #Generate sentences on a given word/symbol (here it is the start symbol)
            gram_prevTk.append("<s>")
            word_prevTk.append("START")
        gram_tk = "" #Initialize empty token string
        word_tk = ""
        while(gram_tk != "</s>"): #Loop until we find an END token
            text += word_tk + " "
            gram_tk = grammar_model.generate(list(gram_prevTk))
            wordgram = gram_tk.upper()
            word_tk = word_tag_model.generate(list(word_prevTk))
            gram_prevTk.pop(0)
            word_prevTk.pop(0)
            gram_prevTk.append(gram_tk)
            word_prevTk.append(wordgram)
        print(text.strip())
        gram_prevTk = list()
        word_prevTk = list()


def split(sentences, fraction):
    """
    Split a fraction of a list of sentences into a training and test set.
    :param sentences: Initial training as a list of sentences data to be split
    :param fraction: represents the fraction of sentences to place in a training data set, the remainder goes into test set
    :return: list of training sentences, list of test sentences
    """
    split_sentences = list(sentences)
    random.shuffle(split_sentences)
    break_point = int(len(split_sentences) * fraction)
    return split_sentences[:break_point], split_sentences[break_point:]



def generateText(model, sents):
    """
    Generate sentences of text based on a single model.
    :param model: NgramModel of words
    :param sents: Integer, number of sentences to generate
    :return: none
    """
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
