import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import MLEProbDist

cfd = ConditionalFreqDist()

s = [[w.lower() for w in s if w.isalnum()] for s in brown.sents(categories='adventure')[:2]]

for sent in s:
	sent.insert(0,'<s>')
	sent.append('</s>')

for sent in s:
	print("sent : ",sent)
	for (w1,w2) in list(zip(sent[:-1],sent[1:])):
		print("Bigram: ",w1,w2)
		cfd[w1][w2] +=1
cpd = ConditionalProbDist(cfd, MLEProbDist)

test = ["<s>","i", "can", "like", "apples","</s>"]
p = 1

for i in range(len(test)-1):
	print(test[i],test[i+1])
	print(cpd[test[i]].prob(test[i+1]))
	p *= cpd[test[i]].prob(test[i+1])