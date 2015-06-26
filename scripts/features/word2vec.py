# requires train, test
# appends to features, test_features

import re
import logging
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

def clean_text(again_text):
  letters_only = re.sub("[^a-zA-Z]", " ", again_text)
  words = letters_only.lower().split()
  stops = set(stopwords.words("english"))
  meaningful_words = [w for w in words if not w in stops]
  # meaningful_words = words
  # return( " ".join( meaningful_words ))
  return(meaningful_words)

def setup_sentences(not_text):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  nice_text = BeautifulSoup(not_text).get_text()
  raw_sentences = tokenizer.tokenize(nice_text)
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      sentences.append(clean_text(raw_sentence))
  return sentences

# try w/o the description?
# traincombined = train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1)
traincombined = train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1)

sentences = []
for text in traincombined:
  sentences += setup_sentences(text)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# try fiddling
num_features = 400
min_word_count = 10
num_workers = 4
context = 10
downsampling = 1e-3

print "Training word2vec model..."
w2v = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features,
    min_count=min_word_count, window=context, sample=downsampling)

# forget what this does
w2v.init_sims(replace=True)

index2word_set = set(w2v.index2word)

train_values = []
test_values = []

for example in train.values:
  query = example[0]
  title = example[1]
  queryVec = np.zeros((num_features,),dtype="float32")
  titleVec = np.zeros((num_features,),dtype="float32")
  nqwords = 0.
  ntwords = 0.
  for word in clean_text(query):
    if word in index2word_set:
      nqwords = nqwords + 1.
      queryVec = np.add(queryVec, w2v[word])
  for word in clean_text(title):
    if word in index2word_set:
      ntwords = ntwords + 1.
      titleVec = np.add(titleVec, w2v[word])
  queryVec = np.divide(queryVec,nqwords)
  titleVec = np.divide(titleVec,ntwords)
  train_values.append(np.concatenate([queryVec, titleVec]))

for example in test.values:
  query = example[0]
  title = example[1]
  queryVec = np.zeros((num_features,),dtype="float32")
  titleVec = np.zeros((num_features,),dtype="float32")
  nqwords = 0.
  ntwords = 0.
  for word in clean_text(query):
    if word in index2word_set:
      nqwords = nqwords + 1.
      queryVec = np.add(queryVec, w2v[word])
  for word in clean_text(title):
    if word in index2word_set:
      ntwords = ntwords + 1.
      titleVec = np.add(titleVec, w2v[word])
  queryVec = np.divide(queryVec,nqwords)
  titleVec = np.divide(titleVec,ntwords)
  test_values.append(np.concatenate([queryVec, titleVec]))

features.append(np.nan_to_num(np.asarray(train_values)))
test_features.append(np.nan_to_num(np.asarray(test_values)))

