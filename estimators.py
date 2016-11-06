from sklearn.linear_model import Ridge
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.grid_search import RandomizedSearchCV
from sklearn import svm



def get_SGDC_best_estimator(xTrain, yTrain):

    model = Ridge()
    parameters = {'alpha': sp_rand()}
    rdmSearch = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100)
    rdmSearch.fit(xTrain, yTrain)
    #the pre defined alphas is worse!!
    #alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    #grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    #grid.fit(xTrain, yTrain)
    return rdmSearch.best_estimator_

def get_LinearSVC_best_estimator(xTrain, yTrain):

    model = svm.LinearSVC()
    parameters = {'C': (.1, .5, 1.0)}
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(xTrain, yTrain)
    return grid.best_estimator_