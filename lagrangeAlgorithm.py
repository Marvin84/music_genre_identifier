from dataStructures import *
from manageDataset import *
from lagrange.lagrangean_s3vm import *
import mySeed



#set the pairTarget list in get_lagrange_dataset
def launch_lagrange(training, test, percentageLabel, r):
    xtrain_l, ytrain_l, xtrain_u, xtest, ytest, l, u = get_lagrange_dataset(training, test, [1, 3], percentageLabel, True)
    best_estimator = get_best_estimator_by_cv(xtrain_l, ytrain_l, 3)
    print best_estimator
    s3vc = lagrangian_s3vm_train(l,
                                 u,
                                 xtrain_l,
                                 ytrain_l,
                                 xtrain_u,
                                 C=best_estimator.C,
                                 gamma=best_estimator.gamma,
                                 kernel=best_estimator.kernel,
                                 r=r,
                                 rdm = mySeed.rdm
                                 )
    return s3vc.score(xtest,ytest)