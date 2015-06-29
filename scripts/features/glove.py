# requires sentences as defined by word2vec.py
# appends to features, test_features

import subprocess
import os.path

# only need this once to generate vector file
if not os.path.isfile('./tmp/glove_vectors.txt'):
  f = open('./tmp/words_for_glove.txt', 'w')
  for sentence in sentences:
    f.write("%s " % " ".join(sentence))

  subprocess.call(["./scripts/bash/glove.sh"])

