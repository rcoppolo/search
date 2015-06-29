Python scripts for Kaggle's [Crowdflower Search Results Relevance
competition](https://www.kaggle.com/c/crowdflower-search-relevance). Lots of
neat language stuff!

Expects compiled [GloVe](http://nlp.stanford.edu/projects/glove/) directory to
be at ../glove compared to this directory (or define `GLOVE_PATH` in
`./scripts/bash/glove.sh`).

Run `python scripts/main.py` for cross validation and grid search.

Run `python scripts/main.py --submit` to generate submission file.

### Todo

* Try out GloVe features/model
* Try out a NN model
* Use some more external data
* Think
