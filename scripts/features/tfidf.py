# requires train, test
# appends to features, test_features

"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

https://www.kaggle.com/abhishek/crowdflower-search-relevance/beating-the-benchmark/run/12217
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# do some lambda magic on text columns
traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(min_df=3, max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

# Fit TFIDF
tfv.fit(traindata)
features.append(tfv.transform(traindata))
test_features.append(tfv.transform(testdata))
