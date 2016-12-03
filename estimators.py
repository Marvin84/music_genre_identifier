from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import svm



def get_SGDC_best_estimator(xTrain, yTrain):

    model = Ridge(random_state=42)
    parameters = {'alpha': sp_rand()}
    rdmSearch = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100, random_state=42)
    rdmSearch.fit(xTrain, yTrain)
    #the pre defined alphas is worse!!
    #alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    #grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    #grid.fit(xTrain, yTrain)
    return rdmSearch.best_estimator_

def get_LinearSVC_best_estimator(xTrain, yTrain):

    model = svm.SVC(kernel='linear', random_state=42)
    parameters = {'C': (.1, .5, 1.0)}
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(xTrain, yTrain)
    return grid.best_estimator_

def get_my_best_estimator(data, targets, folds):
    n, d = data.shape

    params_grid = [{
        'C': [2**i for i in range(-10, 10)],
        'gamma': [2**i for i in range(-10, 10)],
        'kernel': ['linear', 'rbf']}]
    gs = GridSearchCV(SVC(), params_grid, n_jobs=-1, cv=folds)
    gs.fit(data, targets)
    best_estimator = gs.best_estimator_
    return best_estimator