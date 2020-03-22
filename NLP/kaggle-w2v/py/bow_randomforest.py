import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from KaggleWord2VecUtility import KaggleWord2VecUtility

def review_to_words(raw_review):
	review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
	letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
	words = letters_only.lower().split()
	stops = set(stopwords.words('english'))
	meaningful_words = [w for w in words if w not in stops]
	return ' '.join(meaningful_words)

if __name__ == '__main__':
	# train_data = pd.read_csv('.\\data\\labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
	ori_data = pd.read_csv('.\\data\\train.tsv', header=0, delimiter='\t', quoting=3)
	data_size = ori_data['id'].size
	split_size = data_size*9/10
	train_data = ori_data.loc[:split_size, :]
	# test_data = train_data
	test_data = ori_data.loc[split_size:, :]

	num_reviews = train_data['review'].size
	clean_train_reviews = []
	for i in xrange(0, num_reviews):
		if (i+1)%1000 == 0:
			print 'Review %d of %d\n' % (i+1, num_reviews)
		clean_train_reviews.append(review_to_words(train_data['review'][i]))

	print 'Starting Training...'
	vectorizer = CountVectorizer(analyzer = 'word', \
								tokenizer = None, \
								preprocessor = None, \
								stop_words = None, \
								max_features = 5000)
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()
	
	vocab = vectorizer.get_feature_names()
	dist = np.sum(train_data_features, axis=0)

	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(train_data_features, train_data['sentiment'])

	# Predict
	print 'Start Prediction...'
	# test_data = pd.read_csv('.\\data\\testData.tsv', header=0, delimiter='\t', quoting=3)
	# test_data = pd.read_csv('.\\data\\test.tsv', header=0, delimiter='\t', quoting=3)

	num_reviews = test_data['review'].size
	clean_test_reviews = []
	for g in test_data['review']:
		clean_test_reviews.append(review_to_words(g))
	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()

	result = forest.predict(test_data_features)
	# output = pd.DataFrame(data={'id':test_data['id'], 'sentiment':result})
	output = pd.DataFrame(data={'id':test_data['id'], 'review':test_data['review'], 'review2':clean_test_reviews, 'prediction':result, 'label':test_data['sentiment']})

	output.to_csv('Bag_of_Words_RF_model.csv', index=False, quoting=0, sep='\t')
