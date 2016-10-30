from dataStructures import *
from manageDataset import *
from lagrange.lagrangean_s3vm import *


#set the pairTarget list in get_lagrange_dataset
def launch_lagrange(dataset, percentageLabel, rdm, r):
    xtrain_l, ytrain_l, xtrain_u, xtest, ytest, l, u = get_lagrange_dataset(dataset, [1, 3], percentageLabel, True)
    best_estimator = get_best_estimator_by_cv(xtrain_l, ytrain_l, 3)
    s3vc = lagrangian_s3vm_train(l,
                                 u,
                                 xtrain_l,
                                 ytrain_l,
                                 xtrain_u,
                                 C=best_estimator.C,
                                 gamma=best_estimator.gamma,
                                 kernel=best_estimator.kernel,
                                 r=r,
                                 rdm=rdm
                                 )