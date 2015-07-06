#!/bin/bash

CORPUS=./tmp/words_for_glove.txt
VOCAB_FILE=./tmp/vocab_for_glove.txt
COOCCURRENCE_FILE=./tmp/glove_cooccurrence.bin
COOCCURRENCE_SHUF_FILE=./tmp/glove_cooccurrence.shuf.bin
SAVE_FILE=./tmp/glove_vectors50_5
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5 # fiddle
VECTOR_SIZE=50 # equal to num_features in ./scripts/features/glove.py
MAX_ITER=25
WINDOW_SIZE=15
BINARY=0 # save vectors as txt
NUM_THREADS=8
X_MAX=10

GLOVE_PATH=../glove

"$GLOVE_PATH/vocab_count" -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  "$GLOVE_PATH/cooccur" -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    "$GLOVE_PATH/shuffle" -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       "$GLOVE_PATH/glove" -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
    fi
  fi
fi


