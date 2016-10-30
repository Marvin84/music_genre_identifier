from dataStructures import *
from manageDataset import *
from lagrange.lagrangean_s3vm import *
import mySeed



#set the pairTarget list in get_lagrange_dataset
def launch_lagrange(training, test, percentageLabel, r):
    xtrain_l, ytrain_l, xtrain_u, xtest, ytest, l, u = get_lagrange_dataset(training, test, [2, 4], percentageLabel, True)
    best_estimator = get_best_estimator_by_cv(xtrain_l, ytrain_l, 3)
    model = lagrangian_s3vm_train(l, u, xtrain_l, ytrain_l, xtrain_u, C=best_estimator.C,
                                 gamma=best_estimator.gamma, kernel=best_estimator.kernel, r=r, rdm = mySeed.rdm)

    return model.score(xtest,ytest)