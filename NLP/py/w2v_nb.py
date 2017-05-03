import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import math
import nltk.data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from KaggleWord2VecUtility import KaggleWord2VecUtility

def review_to_words(raw_review, remove_stopwords=True):
	review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
	letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
	words = letters_only.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words('english'))
		# meaningful_words = [w for w in words if w not in stops]
		words = [w for w in words if w not in stops]
	return (words)
	# return ' '.join(meaningful_words)

def review_to_sentences(raw_review, tokenizer, remove_stopwords=True):
	raw_sentences = tokenizer.tokenize(raw_review.strip())
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(review_to_words(raw_sentence, remove_stopwords))
	return sentences

def cacu_nb(train_data, train_data_features):
	_train_data = np.array(train_data['sentiment'])
	m1 = np.zeros(train_data_features.shape[0], dtype = bool)
	m1[_train_data == 1] = True

	count1 = np.sum(train_data_features[m1], axis=0)
	counts = np.sum(train_data_features, axis=0)
	probs1 = np.divide(1.0*count1, counts)
	return probs1

def nb_predict(features, probs):
	tags = []
	for g in features:
		tag = -1
		positive_result = 0
		negative_result = 0
		for idx in range(len(g)):
			if probs[idx] != 0:
				positive_result += math.log(probs[idx]) * g[idx]
			if probs[idx] != 1:
				negative_result += math.log(1-probs[idx]) * g[idx]
		if positive_result > negative_result:
			tag = 1
		else:
			tag = 0
		tags.append(tag)
	return tags

if __name__ == '__main__':
	# train_data = pd.read_csv('.\\data\\labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
	# test_data = pd.read_csv('.\\data\\unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
	train_data = pd.read_csv('.\\data\\train.tsv', header=0, delimiter='\t', quoting=3)
	test_data = pd.read_csv('.\\data\\test.tsv', header=0, delimiter='\t', quoting=3)

	sentences = []
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	for review in train_data['review']:
		review = review.decode('utf-8')
		sentences += review_to_sentences(review, tokenizer, False)

	for review in test_data['review']:
		review = review.decode('utf-8')
		sentences += review_to_sentences(review, tokenizer, False)

	import logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s, level=logging.INFO')

	num_features = 300
	min_word_count = 40
	num_workers = 4
	context = 10
	downsampling =1e-3

	from gensim.models import word2vec
	model = word2vec.Word2Vec(sentences, workers=num_workers, \
							size=num_features, min_count=min_word_count,\
							window=context, sample=downsampling)
	model.init_sims(replace=True)
	# model_name = "demo1"
	# model.save(model_name)