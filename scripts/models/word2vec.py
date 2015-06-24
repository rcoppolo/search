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
  # 'm__max_depth' : [3, None],
  # 'm__max_features' : [1, 3, 10],
  # 'm__min_samples_split' : [1, 3, 10],
  # 'm__min_samples_leaf' : [1, 3, 10],
  # 'm__bootstrap' : [True, False],
  # 'm__criterion' : ["gini", "entropy"],
  'm__C': [1, 10, 100, 1000],
  'm__gamma': [0.001, 0.0001],
  'm__class_weight': ['auto', None]
})

