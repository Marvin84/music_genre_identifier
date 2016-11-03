from __future__ import division
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
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
        xTrainL, yTrainL, xTrainU, l, u = get_lagrange_dataset(training, test, percentageLabel, pairTarget, pairwise)

    best_estimator = get_best_estimator_by_cv(xTrainL, yTrainL, 3)
    model = lagrangian_s3vm_train(l, u, xTrainL, yTrainL, xTrainU, C=best_estimator.C,
                                 gamma=best_estimator.gamma, kernel=best_estimator.kernel, r=r, rdm = mySeed.rdm)

    if pairwise: print "The score is: ", model.score(xTest,yTest)
    return model


def launch_qn_algorithm(training, test, percentageLabel, r, pairwise = False, pairTarget = None):
    xTrainL, yTrainL, xTrainU, xTest, yTest = get_qn_dataset(training, test, percentageLabel, pairwise, pairTarget)
    model = QN_S3VM(xTrainL, yTrainL, xTrainU, qn.randomGenerator.my_random_generator, lam=0.0009765625, lamU=1,
                    kernel_type="RBF", sigma=0.5, estimate_r=r )
    model.train()
    preds = model.getPredictions(xTest)
    error = classification_error(preds,yTest)
    print "Classification error of QN-S3VM: ", error
    return model


def launch_oneVsRest_lagrange(training, test, targets, targetsDic, percentageLabel, r):

    models = []
    multiclassTraining = get_multiclass_dataset(training, targets, targetsDic)

    #testset
    xTest, yTest = get_data_target_lists(test)
    xTestArray = MinMaxScaler(feature_range=(0, 1)).transform(np.array(xTest))
    yTestArray = np.array(yTest)
    decisions = []

    #training models
    for i in range(len(targets)):
       models.append(launch_lagrange(multiclassTraining[targets[i]], test, percentageLabel, r))
    #getting the distances
    distPointList = []
    for i in range(len(targets)):
        decisions.append(models[i].decision_function(xTestArray))

    #distPointList is a list of lists where every list has the distance from n models of every datapoint
    for i in range(len(decisions[0])):
        distPoint = []
        for j in range(len(decisions)):
         distPoint.append(decisions[j][i])
        distPointList.append(distPoint)

    #get the highest distance
    predictions=[]
    for i in range(len(distPointList)):
        predictions.append(max(distPointList[i]))
        #maxElement = max(distPointList[i])
        #if maxElement >= 0:
         #   predictions.append(distPointList[i].index(maxElement)+1)
        #else: predictions.append(100)

    #n = predictions.count(100)
    #m = len(predictions) - n
    #rightPercent = ((np.count_nonzero(np.array(predictions) == yTestArray))/m)*100
    #print m, np.count_nonzero(np.array(predictions))
    #print "there are ", n, "undefined and ", rightPercent, "accurary on defined"










'''    creating two lists of predictions and real labels
    targetsNum = get_numbered_classes(targets)
    #labels = count_labels(yTest, targetsNum)
    #predictions = []

    #for i in range(len(models)):
    #   predictions.append(np.count_nonzero(models[i].predict(xTestArray) == 1))

    #print yTestArray

    #classifier.estimators_ = models
    #classifier.classes_ = targets
    #classifier.fit(xTrainL, yTrainL)
    #classifier.label_binarizer_ = False
    #print classifier.score(xTestArray, yTestArray)
    print get_numbered_classes(targets)
    best_estimator = get_best_estimator_by_cv(xTrainL, yTrainL, 3)

    svc = SVC(C = best_estimator.C, gamma = best_estimator.gamma, kernel = best_estimator.kernel)
    classifier2 = OneVsRestClassifier(svc)
    #classifier2.estimators_ = models
    #classifier2.classes_ = get_numbered_classes(targets)
    classifier2.fit(xTrainArray, yTrainArray)
    print classifier2.predict(xTestArray)
    #trainset for the estimator
    xTrain, yTrain = get_data_target_lists(training)
    l, u = get_l_u(training, percentageLabel)
    xTrainL = np.array(xTrain[:l])
    yTrainL = np.array(yTrain[:l])
    xTrainArray = np.array(xTrain)
    yTrainArray = np.array(yTrain)
'''



























