import os
import sys
import ml_metrics
import pandas as pd
from sklearn import metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

if __name__ == "__main__":

  # load data
  train = pd.read_csv('./downloaded/train.csv').fillna("")
  test = pd.read_csv('./downloaded/test.csv').fillna("")

  # drop ids but save for submission
  ids = test.id.values.astype(int)
  train = train.drop('id', axis=1)
  test = test.drop('id', axis=1)

  # save targets and score variance
  targets = train.median_relevance.values
  score_variance = train.relevance_variance.values
  train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

  # one model at a time, for now

  # define features
  execfile('./scripts/features/tfidf.py')

  # define model, params
  execfile('./scripts/models/beating_the_benchmark.py')

  # create grid search for best params
  scorer = metrics.make_scorer(ml_metrics.quadratic_weighted_kappa, greater_is_better = True)
  searcher = GridSearchCV(model, params, n_jobs=-1, verbose=10, scoring=scorer, cv=3)
  # searcher = RandomizedSearchCV(model, params, n_jobs=-1, verbose=10, scoring=scorer, cv=3)
  searcher.fit(features, targets)

  # print score and best params
  print("Best score... %0.3f" % searcher.best_score_)
  print("Best params...")
  best_params = searcher.best_estimator_.get_params()
  for param in sorted(params.keys()):
    print("\t%s: %r" % (param, best_params[param]))

  # require --submit flag to generate submission csv
  if "--submit" in sys.argv:

    # run best model on test data
    best = searcher.best_estimator_
    predictions = best.predict(test_features)

    # write submission file w/ git hash identifier
    submission = pd.DataFrame({"id": idx, "prediction": predictions})
    proc = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    submission_name = proc.stdout.read().strip()
    submission.to_csv("../submissions/" + submission_name + ".csv", index=False)
