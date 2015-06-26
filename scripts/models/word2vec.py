# appends to model, params

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scl = StandardScaler()
# m = RandomForestClassifier(n_estimators=100)
m = SVC()

models.append(Pipeline([('scl', scl), ('m', m)]))

params.append({
  # 'm__max_depth' : [None],
  # 'm__max_features' : [10], # maybe higher?
  # 'm__min_samples_split' : [3],
  # 'm__min_samples_leaf' : [1],
  # 'm__bootstrap' : [False],
  # 'm__criterion' : ["gini"],
  # SVC
  'm__C': [1000], # w/ 500 features, 100
  'm__gamma': [0.0001], # w/ 500 features, 0.001
  'm__class_weight': [None]
})
