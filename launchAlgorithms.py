from sklearn.multiclass import OneVsRestClassifier
from manageDataset import *
from lagrange.lagrangean_s3vm import *
from qn.qns3vm import *
from qn.examples import *
import qn.randomGenerator
import mySeed



#set the pairTarget list if you want the pairwise version
def launch_lagrange(training, test, percentageLabel, r, pairwise = False, pairTarget = None):
    if pairwise:
        xTrainL, yTrainL, xTrainU, xTest, yTest, l, u = get_lagrange_dataset(training, test, percentageLabel, pairTarget, pairwise)
    else:
        xTrainL, yTrainL, xTrainU, xTest, yTest, l, u = get_lagrange_dataset(training, test, percentageLabel, pairTarget, pairwise)

    best_estimator = get_best_estimator_by_cv(xTrainL, yTrainL, 3)
    model = lagrangian_s3vm_train(l, u, xTrainL, yTrainL, xTrainU, C=best_estimator.C,
                                 gamma=best_estimator.gamma, kernel=best_estimator.kernel, r=r, rdm = mySeed.rdm)

    if pairwise: print "The score is: ", model.score(xTest,yTest)
    return model


def launch_qn_algorithm(training, test, percentageLabel, r):
    xTrainL, yTrainL, xTrainU, xTest, yTest = get_qn_dataset(training, test, percentageLabel)
    model = QN_S3VM(xTrainL, yTrainL, xTrainU, qn.randomGenerator.my_random_generator, lam=0.0009765625, lamU=1,
                    kernel_type="RBF", sigma=0.5, estimate_r=r )
    model.train()
    preds = model.getPredictions(xTest)
    error = classification_error(preds,yTest)
    print "Classification error of QN-S3VM: ", error
    return model


def launch_oneVsRest_lagrange(training, test, targets, targetsDic, percentageLabel, r):

    models = {}
    multiclassTraining = get_multiclass_dataset(training, targets, targetsDic)

    #training models
    for i in range(len(targets)):
        models[targets[i]] = launch_lagrange(multiclassTraining[targets[i]], test, percentageLabel, r)
        #print models[targets[i]]
















