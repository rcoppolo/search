import os
import sys
import subprocess
import ml_metrics
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
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

  # required variables
  features, test_features, models, params, best, ensemble_scores = [], [], [], [], [], []

  # my ugly way of trying out different features and models...

  # define features
  execfile('./scripts/features/word2vec.py')
  execfile('./scripts/features/variance.py')
  execfile('./scripts/features/tfidf.py')
  execfile('./scripts/features/porter_stemmer.py')

  # define model, params
  execfile('./scripts/models/word2vec.py')
  # execfile('./scripts/models/variance.py')
  execfile('./scripts/models/beating_the_benchmark.py')
  execfile('./scripts/models/porter_stemmer.py')

  # after all features have created, we create a mask to effectively "set aside"
  # some data to later validate an ensemble (later maybe loop and do this a few times?)

  cv_ensemble = 3 #5
  test_split = 0.3

  for _i in range(cv_ensemble):
    random = np.random.rand(train.shape[0])
    train_mask = np.where(random > test_split)
    test_mask = np.where(random <= test_split)

    # for each model, we grid search the best params using the masked data
    for i in range(len(models)):
      # just fit if we've already grid searched, to save time
      if len(best) == len(models): # and False:
        print("Fitting a model...")
        current_features = features[i]
        best[i].fit(current_features[train_mask], targets[train_mask])
      else:
        model = models[i]
        current_params = params[i]
        current_features = features[i]

        scorer = metrics.make_scorer(ml_metrics.quadratic_weighted_kappa, greater_is_better=True)
        searcher = GridSearchCV(model, current_params, n_jobs=1, verbose=10, scoring=scorer, cv=3)
        # searcher = RandomizedSearchCV(model, current_params, n_jobs=-1, verbose=10, scoring=scorer, cv=3)
        searcher.fit(current_features[train_mask], targets[train_mask])

        # print score and best params
        print("Best score... %0.3f" % searcher.best_score_)
        print("Best params...")
        best_params = searcher.best_estimator_.get_params()
        for param in sorted(current_params.keys()):
          print("\t%s: %r" % (param, best_params[param]))

        # save the trained model for later
        best.append(searcher.best_estimator_)

    # we then validate the ensemble on the set aside features and targets
    predictions = []
    for i in range(len(best)):
      model = best[i]
      predictions.append(model.predict(features[i][test_mask]))

    # just averaging for now, play with this later
    predictions = np.sum(predictions, axis=0) / len(predictions)
    score = ml_metrics.quadratic_weighted_kappa(targets[test_mask], predictions)
    ensemble_scores.append(score)
    print("Ensemble score... %0.3f" % score)

  print("CV ensemble score... %0.3f" % np.mean(ensemble_scores))

  # once the ensemble has been validated, we can fit each model with all the
  # training data and make predictions for the real test data, if we want to
  # generate a submission csv (requires --submit flag)
  if "--submit" in sys.argv:

    # run all models on all data
    predictions = []
    for i in range(len(best)):
      model = best[i]
      print("Thinking...")
      model.fit(features[i], targets)
      print("Predicting...")
      predictions.append(model.predict(test_features[i]))

    # just averaging for now, play with this later
    predictions = np.sum(predictions, axis=0) / len(predictions)

    # write submission file w/ git hash identifier
    submission = pd.DataFrame({"id": ids, "prediction": predictions})
    proc = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    submission_name = proc.stdout.read().strip()
    submission.to_csv("./submissions/" + submission_name + ".csv", index=False)
