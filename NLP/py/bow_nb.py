"""
Bag-of-Words with Naive Bayes Model
Description:
- Control the number of words selected as the features
- Results:
- 1000 -> 0.83064
- 5000 -> 0.83832
"""

import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import logging
import re
import math
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def review_to_words(raw_review, remove_stopwords=True):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words('english'))
        meaningful_words = [w for w in words if w not in stops]
    return ' '.join(meaningful_words)

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

def get_accuracy(hat, tgts):
    L = len(hat)
    acc = 0
    ssum = L
    idx = 0
    for tgt in tgts:
        if hat[idx] == tgt:
            acc += 1
        idx += 1
    return acc, ssum, 1.0*acc/ssum

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python ./py/bow_nb.py [-d|-t]'
        print '\t-d\tdebug mode'
        print '\t-t\ttrianing mode'
        exit(0)
    elif sys.argv[1] == '-d':
        print 'debug mode'
        ori_data = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
        #ori_data = pd.read_csv('./data/tmp', header=0, delimiter='\t', quoting=3)
        data_size = ori_data['id'].size
        split_size = data_size*9/10
        train_data = ori_data.loc[:split_size, :]
        test_data = ori_data.loc[split_size:, :]
    elif sys.argv[1] == '-t':
        print 'training mode'
        train_data = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
        test_data = pd.read_csv('./data/testData.tsv', header=0, delimiter='\t', quoting=3)

    # Processing
    logging.warning("Starting processing...")
    num_reviews = train_data['review'].size
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        if (i+1)%1000 == 0:
            logging.warning('Review %d of %d' % (i+1, num_reviews))
        clean_train_reviews.append(review_to_words(train_data['review'][i]))

    # Training
    logging.warning('Starting Training...')
    vectorizer = CountVectorizer(analyzer = 'word', \
                                tokenizer = None, \
                                preprocessor = None, \
                                stop_words = None,\
                                max_features = 5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    probs_positive = cacu_nb(train_data, train_data_features)
    
    """
    vocab = vectorizer.get_feature_names()
    words_count = np.sum(train_data_features, axis=0)
    for tag, prob, count in zip(vocab, probs_positive, words_count):
        print tag, prob, count
    """

    # Predict
    logging.warning('Start Prediction...')
    num_reviews = test_data['review'].size
    clean_test_reviews = []
    for g in test_data['review']:
        clean_test_reviews.append(review_to_words(g))
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    result = nb_predict(test_data_features, probs_positive)

    if sys.argv[1] == '-d':
        acc, ssum, prob = get_accuracy(result, test_data['sentiment'])
        report = '%d\t%d\t%.3lf\n' % (acc, ssum, prob)
        output = pd.DataFrame(data={'id':test_data['id'], 'review':test_data['review'], 'review2':clean_test_reviews, 'prediction':result, 'label':test_data['sentiment']})

        with open('./result/Bow_NB_case', 'w') as f:
            f.write(report)
        output.to_csv('./result/Bow_NB_case', index=False, quoting=0, sep='\t', mode='a')
    elif sys.argv[1] == '-t':
        output = pd.DataFrame(data={'id':test_data['id'], 'sentiment':result})
        output.to_csv('./result/Bow_NB', index=False, quoting=3, sep=',')
