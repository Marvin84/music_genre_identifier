from __future__ import division
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

def get_best_estimator_by_cv(data, targets, folds, gamma=None, C =[2**i for i in range(0, 5)], kernel=['linear', 'rbf']):
    n, d = data.shape
    if gamma is None : gamma = [1/d]
    positives = len([y for y in targets if y==1])
    if folds>len(data) or positives == 0 or positives == len(data) : return SVC(C=1, gamma=1/d)
    params_grid = [
      {'C': C,
       'gamma': gamma,
       'kernel': kernel}
    ]
    gs = GridSearchCV(SVC(), params_grid, n_jobs=-1, cv=folds)
    gs.fit(data, targets)
    best_estimator = gs.best_estimator_
    return best_estimator

def get_positive_samples_ratio(targets):
    return len([y for y in targets if y==1.0])/len(targets)

def get_max_distance_between_samples(data):
    return max(euclidean_distances(data, data).ravel())  
