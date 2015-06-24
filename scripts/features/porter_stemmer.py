# requires train, test
# appends to features, test_features

import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
some_features = []
some_test_features = []

def something(example):
  adjusted_query = (" ").join(["q"+ z for z in BeautifulSoup(example[0]).get_text(" ").split(" ")])
  adjusted_title = (" ").join(["z"+ z for z in BeautifulSoup(example[1]).get_text(" ").split(" ")])
  description = BeautifulSoup(example[2]).get_text(" ")
  s = adjusted_query + " " + adjusted_title + " " +  description
  s = re.sub("[^a-zA-Z0-9]"," ", s)
  return (" ").join([stemmer.stem(z) for z in s.split(" ")])

for i in range(len(train.values)):
  s = something(train.values[i])
  some_features.append(s)

for i in range(len(test.values)):
  s = something(test.values[i])
  some_test_features.append(s)

some_features = np.asarray(some_features)
features.append(some_features)
some_test_features = np.asarray(some_test_features)
test_features.append(some_test_features)

