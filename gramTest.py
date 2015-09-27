import nltk
import nltk.collocations
import nltk.corpus
import collections
from nltk.corpus import gutenberg

corpus = gutenberg.words('austen-persuasion.txt');

def buildModel(iCorpus):
	'''
	Instead of using likelihood_ratio

	we could try and use any of these methods
	raw_freq - Scores ngrams by their frequency
	student_t - Scores ngrams using Student's t test with independence hypothesis for unigrams, as in Manning and Schutze 5.3.1.
	chi_sq - Scores ngrams using Pearson's chi-square as in Manning and Schutze 5.3.3.
	mi_like - Scores ngrams using a variant of mutual information. The keyword argument power sets an exponent (default 3) for the numerator. No logarithm of the result is calculated.
	pmi - Scores ngrams by pointwise mutual information, as in Manning and Schutze 5.4.
	likelihood_ratio - Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4.
	poisson_stirling - Scores ngrams using the Poisson-Stirling measure."
	jaccard - Scores ngrams using the Jaccard index.
	'''
	global scoredBigrams
	bgm    = nltk.collocations.BigramAssocMeasures()
	print("Splitting up words in the dataset")
	finder = nltk.collocations.BigramCollocationFinder.from_words(iCorpus)
	#We could also try and use TrigramCollocationFinder then we also need to change the TrigramAssocMeasures()
	print("Building the nGram scoresList")
	scoredBigrams = finder.score_ngrams( bgm.likelihood_ratio  )
	return scoredBigrams

def estimateWord(word, scoredBigrams, numberOfEstimates):
	print("Grouping all the bigram by first word")
	# Group bigrams by first word in bigram.                                        
	prefix_keys = collections.defaultdict(list)
	for key, scores in scoredBigrams:
	   prefix_keys[key[0]].append((key[1], scores))

	print("Sorting the bigrams")
	# Sort keyed bigrams by strongest association.                                  
	for key in prefix_keys:
	   prefix_keys[key].sort(key = lambda x: -x[1])
	return prefix_keys[word][:numberOfEstimates]




model = buildModel(corpus)
print("Most likely words to come after 'therefore'")
print(estimateWord('therefore', model,5))


