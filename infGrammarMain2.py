from infGrammarNgram2 import nGramModel
from nltk.corpus import brown
from nltk.probability import ELEProbDist
from nltk.probability import SimpleGoodTuringProbDist
from math import log
import nltk
import random
#import ngramTrials2
import sys

smoothing_string = "SimpleGoodTuringProbDist"

def main():
    """
    Provide an entry point into program.
    :return: None
    """
    print("Building models...")
    order = 4
    smoothing = SimpleGoodTuringProbDist
    ##NGRAM MODEL FOR GRAMMAR
    sents_ = brown.tagged_sents()
    sents = list(sents_) #needs to be mutable to insert start/end tokens if working with tags
    sents = remove_punctuation(sents)
    sentences_of_tags = []
    #Pull out the tags and make sentences of just tags!
    for sentence in sents:
        sentence_tags = [tag for (word, tag) in sentence if word.isalnum()]
        sentences_of_tags.append(sentence_tags)

    print("Building grammar model")
    testModelGrammar = generateModelFromSentences(sentences_of_tags, smoothing, order,False,"START","END") #Create trigram of only grammar
    ##NGRAM MODEL FOR TAGS AND WORDS
    #print(sents[0], " order: ", order)
    print("Building word model")
    testModelwordtags = generateModelFromSentences(sents, smoothing, order, True)


    print("Generating text...")
    ## GENERATE TEXT
    # infGrammarGenerate(testModelGrammar, testModelwordtags, 10)

    ## HERE BE DEBUGGING
    #TODO: Implement function to split corpus sentences into training and test set.
    #brown_sents_ = brown.tagged_sents()
    #brown_sents = list(brown_sents_)
    ##word_model = ngramTrials2.nGramModel(brown_sents, smoothing, order)
    infGrammarGenerate(testModelGrammar, testModelwordtags, None, 50)


    #Testing out other model
    # ngramTrials2.nGramModel(sents_,smoothing,order)

    #
    # print(perplexity(word_model, test_sent))

    ## TESTS
    assert(testModelGrammar.tagged == False)
    assert(testModelwordtags.tagged == True)
    #assert('START' in testModelwordtags.model)

def remove_punctuation(sentences):
    sents = []
    #Pull out the tags and make sentences of just tags!
    for sentence in sentences:
        nopunc_sentence = [(word, tag) for (word, tag) in sentence if word.isalnum()]
        sents.append(nopunc_sentence)
    return sents

def generateModelFromSentences(sents, smoothingF, n, isTagged=False, startChar="<s>", endChar="</s>"):
    if isTagged:
        addPseudo(sents, n, True)
        return nGramModel(sents, smoothingF, n, True)
    else:
        sents = [[w.lower() for w in s] for s in sents]
        addPseudo(sents,n,False, startChar, endChar)
        return nGramModel(sents, smoothingF, n)

def entropy(model, test):
    """Calculate entropy given an ngram model and a list of test sentences."""

    p = 0
    n = 0
    order = model.getOrder()
    if order == 2:
        for sent in test:
            n += len(sent)
            wPrev = sent.pop(0)
            for w in sent:
                p += -log(model.prob(w,wPrev),2)
        return p/n
    elif order == 3:
        for sent in test:
            n += len(sent)
            w1 = sent.pop(0)
            w2 = sent.pop(0)
            for w in sent:
                p += -log(model.prob_tri(w1, w2, w))
            return p/n



def perplexity(model, test):
    e = entropy(model, test)
    return 2**e

#mutator
def addPseudo(sents, n, tag=False, start="<s>", end="</s>"):
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
                s.insert(0,start)
                s.append(end)

def infGrammarGenerate(grammar_model, word_tag_model, word_model, nrSents):
    """Generate a given number of sentences using a model for grammar and a (word,tag) model of equal N"""
    assert(grammar_model.getOrder() == word_tag_model.getOrder())
    #Create an empty list to hold our conditional tokens
    gram_prevTk = list()
    word_prevTk = list()
    counter = 0
    tagged_sentence = list()
    best_sentence = ("", 999999)
    order = grammar_model.getOrder()
    for _ in range(nrSents): #Generate a sentence, nrSents times
        text = "" #Initialize empty string
        for _ in range(order-1): #Generate sentences on a given word/symbol (here it is the start symbol)
            word_prevTk.append("<s>")
            gram_prevTk.append("START")
        gram_tk = "" #Initialize empty token string
        word_tk = ""
        wordgram = ""
        while(wordgram != "END"): #Loop until we find an END token
            #print("GRAMTOKEN: ",gram_tk)
            text += word_tk + " "
            #print("text so far: ",text)
            tagged_sentence.append((word_tk, gram_tk))
            #print("gram prevtkn: ",list(gram_prevTk))
            gram_tk = grammar_model.generate(list(gram_prevTk))
            #print("gramtoken: ",gram_tk)
            wordgram = gram_tk.upper()
            #print("wordgram: ",wordgram)
            word_tk = word_tag_model.generate(list(word_prevTk), wordgram)
            #print("wdtoken", word_tk)
            while(word_tk == None): # No word found for w1,w2,tag
                gram_tk = grammar_model.generate(list(gram_prevTk)) # GET UPPERCASE
                wordgram = gram_tk.upper()
                #print("new gram_tk: ",gram_tk)
                word_tk = word_tag_model.generate(list(word_prevTk), wordgram)
                #sys.exit()
            #while(word_tk == "</s>"):
            #    word_tk = word_tag_model.generate(list(word_prevTk))
            gram_prevTk.pop(0)
            word_prevTk.pop(0)
            gram_prevTk.append(gram_tk)
            word_prevTk.append(word_tk)
        gram_prevTk = list()
        word_prevTk = list()
        counter += 1
        print(counter)
        if not validate_sentence(text, order):
            continue
        '''perplexity = word_model.perplexity(tagged_sentence)
        if perplexity < best_sentence[1]:
            best_sentence = (text.strip(), perplexity)
            write_to_file(best_sentence, smoothing_string)
            print(text.strip())
            # print(text.strip())'''
        write_to_file((text.strip(),0), smoothing_string)
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


def write_to_file(text, smoothingTechnique):
    model_name = "InferredGrammar"
    sentence = text[0]
    perplexity = float(text[1])
    print("text[1] = " + str(text[1]))
    filename = smoothingTechnique + ".txt"
    with open(filename, "a") as file:
        file.write("%s\t%s\t%s\t%.02f\n" % (model_name, smoothingTechnique, sentence, perplexity))
        file.close()

def validate_sentence(sentence, order):
    """
    Check if a sentence meets various constraints such as length.
    :param sentence:
    :return:
    """
    sent_length = len(sentence.split())
    maximum_phrase_length = 20
    minimum_phrase_length = 4

    #if
    if (sent_length >= maximum_phrase_length) or (order >= sent_length) or (sent_length <= minimum_phrase_length):
        print("Rejected sentence: ")
        return False
    else:
        return True


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
