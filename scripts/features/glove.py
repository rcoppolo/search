# requires sentences as defined by word2vec.py
# appends to features, test_features

import subprocess
import os.path

# equal to SAVE_FILE in scripts/bash/glove.sh
vector_file = './tmp/glove_vectors.txt'

# only need this once to generate vector file
if not os.path.isfile(vector_file):
  f = open('./tmp/words_for_glove.txt', 'w')
  for sentence in sentences:
    f.write("%s " % " ".join(sentence))

  subprocess.call(["./scripts/bash/glove.sh"])

words = pd.read_csv(vector_file, sep=" ", header=None)
word_set = set(words[0])
words = np.asarray(words)
num_features = 50 # equal to VECTOR_SIZE in ./scripts/bash/glove.sh

train_values = []
test_values = []

def get_word_values(word):
  return words[np.where(words[:,0] == word)][:,range(1,num_features + 1)][0]

print("creating GloVe train features...")
for example in train.values:
  query = example[0]
  title = example[1]
  queryVec = np.zeros((num_features,),dtype="float32")
  titleVec = np.zeros((num_features,),dtype="float32")
  nqwords = 0.
  ntwords = 0.
  for word in clean_text(query):
    if word in word_set:
      if np.isinf(np.max(get_word_values(word))):
        print word
      nqwords = nqwords + 1.
      queryVec = np.add(queryVec, get_word_values(word))
  for word in clean_text(title):
    if word in word_set:
      if np.isinf(np.max(get_word_values(word))):
        print word
      ntwords = ntwords + 1.
      titleVec = np.add(titleVec, get_word_values(word))
  # try summing instead of averaging?
  queryVec = np.divide(queryVec,nqwords)
  titleVec = np.divide(titleVec,ntwords)
  train_values.append(np.concatenate([queryVec, titleVec]))

print("creating GloVe test features...")
for example in test.values:
  query = example[0]
  title = example[1]
  queryVec = np.zeros((num_features,),dtype="float32")
  titleVec = np.zeros((num_features,),dtype="float32")
  nqwords = 0.
  ntwords = 0.
  for word in clean_text(query):
    if word in word_set:
      nqwords = nqwords + 1.
      if np.isinf(np.max(get_word_values(word))):
        print word
      queryVec = np.add(queryVec, get_word_values(word))
  for word in clean_text(title):
    if word in word_set:
      if np.isinf(np.max(get_word_values(word))):
        print word
      ntwords = ntwords + 1.
      titleVec = np.add(titleVec, get_word_values(word))
  # try summing instead of averaging?
  queryVec = np.divide(queryVec,nqwords)
  titleVec = np.divide(titleVec,ntwords)
  test_values.append(np.concatenate([queryVec, titleVec]))

features.append(np.nan_to_num(np.asarray(train_values, dtype='float')))
test_features.append(np.nan_to_num(np.asarray(test_values, dtype='float')))
