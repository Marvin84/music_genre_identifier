from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import numpy as np



def get_LinearSVC_best_estimator(xTrain, yTrain):

    model = SVC(kernel='linear', random_state=42)
    parameters = {'C': (0.1, 0.5, 1)}
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(xTrain, yTrain)
    return grid.best_estimator_

def get_my_best_estimator(X, y, folds, kernel):

    if kernel == "rbf":
        params_grid = [{
            'C': [2**i for i in range(-5, 10)],
            'gamma': [2**i for i in range(-10, 3)]}]
        gs = GridSearchCV(SVC(kernel='rbf'), params_grid, n_jobs=-1, cv=folds)


    else:
        params_grid = [{
            'C': [2 ** i for i in range(-5, 10)],
            'gamma': [2 ** i for i in range(-10, 3)],
            'degree': np.array([0, 1, 2, 3])}]
        gs = GridSearchCV(SVC(kernel='poly'), params_grid, n_jobs=-1, cv=folds)

    gs.fit(X, y)
    best_estimator = gs.best_estimator_
    return best_estimator