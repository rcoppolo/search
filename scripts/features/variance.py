# DOESN'T append to features or test_features, just adds to word2vec features

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([('scl', StandardScaler()), ('m', SVR())])
current_params = {
  'm__C': [1],
  'm__gamma': [0.0]
}

var_features = features[-1] # word2vec stuff
var_test_features = test_features[-1] # word2vec stuff
var_targets = StandardScaler().fit_transform(score_variance)

searcher = GridSearchCV(model, current_params, n_jobs=1, verbose=10, cv=3)
# searcher = RandomizedSearchCV(model, current_params, n_jobs=-1, verbose=10, scoring=scorer, cv=3)
searcher.fit(var_features, var_targets)

# print score and best params
print("Best score... %0.3f" % searcher.best_score_)
print("Best params...")
best_params = searcher.best_estimator_.get_params()
for param in sorted(current_params.keys()):
  print("\t%s: %r" % (param, best_params[param]))

# save the trained model for later
predicted_score_variance = searcher.best_estimator_.predict(var_test_features)

# features.append(var_targets) # ?
# test_features.append(predicted_score_variance)

features[-1] = np.column_stack((features[-1], var_targets))
test_features[-1] = np.column_stack((test_features[-1], predicted_score_variance))
