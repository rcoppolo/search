# creates model, params

"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""

from sklearn import pipeline
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
model = pipeline.Pipeline([('svd', svd), ('scl', scl), ('svm', svm)])

# hyperparams
params = {
  'svd__n_components' : [400],
  'svm__C': [10]
}

