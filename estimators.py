from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm


def get_LinearSVC_best_estimator(xTrain, yTrain):

    model = svm.SVC(kernel='linear', random_state=42)
    parameters = {'C': (0.1, 0.5, 1)}
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(xTrain, yTrain)
    return grid.best_estimator_

def get_my_best_estimator(X, y, folds):

    params_grid = [{
        'C': [2**i for i in range(-5, 15)],
        'gamma': [2**i for i in range(-15, 3)]}]
    gs = GridSearchCV(SVC(), params_grid, n_jobs=-1, cv=folds)
    gs.fit(X, y)
    best_estimator = gs.best_estimator_
    return best_estimator