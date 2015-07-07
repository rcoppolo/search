# appends to model, params

"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

https://www.kaggle.com/abhishek/crowdflower-search-relevance/beating-the-benchmark/run/12217
"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Initialize SVD
svd = TruncatedSVD()

# Initialize the standard scaler
scl = StandardScaler()

# We will use SVM here..
svm = SVC()

# Create the pipeline
models.append(Pipeline([('svd', svd), ('scl', scl), ('svm', svm)]))

# hyperparams
params.append({
  'svd__n_components' : [400],
  'svm__C': [10]
})

