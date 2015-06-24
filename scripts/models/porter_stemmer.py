# appends to model, params

# from https://www.kaggle.com/gshguru/crowdflower-search-relevance/clubbing-2-benchmarks

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')
svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
scl = StandardScaler(copy=True, with_mean=True, with_std=True)
svm = SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
models.append(Pipeline([('v', v), ('svd', svd), ('scl', scl), ('svm', svm)]))

# hyperparams
params.append({
  'svd__n_components' : [200],
  'svm__C': [10]
})

