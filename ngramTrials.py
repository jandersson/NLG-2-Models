import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist

class nGramModel():

	def __init__(self, samples, estimator, n = 2):
		cfd = ConditionalFreqDist()
		for sent in samples:
			sent.insert(0,'<s>')
			sent.append('</s>')
		for sent in samples:
			for (w1,w2) in list(zip(sent[:-1],sent[1:])):
				cfd[w1][w2] +=1
		self.model = ConditionalProbDist(cfd, estimator)




sentences = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:2]]

model = nGramModel(sentences,MLEProbDist).model

test = ["<s>","i", "can", "like", "apples","</s>"]
p = 1

for i in range(len(test)-1):
	print(test[i],test[i+1])
	print(model[test[i]].prob(test[i+1]))
	p *= model[test[i]].prob(test[i+1])