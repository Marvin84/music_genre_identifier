from __future__ import division
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from manageDataset import *
from lagrange.lagrangean_s3vm import *
from qn.qns3vm import *
from qn.examples import *
import qn.randomGenerator
import mySeed





#set the pairTarget list if you want the pairwise version
def launch_lagrange(training, test, testScaler, percentageLabel, r, pairwise = False, pairTarget = None):
    if pairwise:
        xTrainL, yTrainL, xTrainU, xTest, yTest, l, u = \
            get_lagrange_pairwise(training, test, testScaler, percentageLabel, pairTarget)
    else:
        xTrainL, yTrainL, xTrainU, l, u = get_lagrange_oneVsAll(training, percentageLabel)

    best_estimator = get_best_estimator_by_cv(xTrainL, yTrainL, 3)
    model = lagrangian_s3vm_train(l, u, xTrainL, yTrainL, xTrainU, C=best_estimator.C,
                                 gamma=best_estimator.gamma, kernel=best_estimator.kernel, r=r, rdm = mySeed.rdm)

    if pairwise: print "The score is: ", model.score(xTest,yTest)
    return model




def launch_oneVsRest_lagrange(training, test, testScaler, targets, targetsDic, percentageLabel, r):

    models = []
    multiclassTraining = get_multiclass_dataset(training, targets, targetsDic)
    #testset
    xTest, yTest = get_data_target_lists(test)
    xTestArray = testScaler.transform(np.array(xTest))
    yTestArray = np.array(yTest)
    decisions = []
    #training models
    for i in range(len(targets)):
       models.append(launch_lagrange(multiclassTraining[targets[i]], test, testScaler, percentageLabel, r))
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
        maxElement = max(distPointList[i])
        predictions.append(distPointList[i].index(maxElement)+1)

    print "langrange's right prediction percentage with oneVsRest strategy: ",get_predition_percentage(predictions, yTest)
    return models


def launch_qn(training, test, percentageLabel, r, pairwise = False, pairTarget = None):
    xTrainL, yTrainL, xTrainU, xTest, yTest = get_qn_dataset(training, test, percentageLabel, pairwise, pairTarget)
    model = QN_S3VM(xTrainL, yTrainL, xTrainU, qn.randomGenerator.my_random_generator, lam=0.0009765625, lamU=1,
                    kernel_type="RBF", sigma=0.5, estimate_r=r )
    model.train()
    preds = model.getPredictions(xTest)
    error = classification_error(preds,yTest)
    print "Classification error of QN-S3VM: ", error
    return model

def launch_KNN (training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    model = KNeighborsClassifier().fit(xTrain, yTrain)

    if crossValid:
        print "KNN's score with 10-fold cross validation: ",cross_val_score(model, xTrain, yTrain, cv=10)

    else:
        predictions = model.predict(np.array(xTest))
        print "KNN's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model

def launch_SVM_oneVsone (training, test, kernel, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    if kernel == 'rbf':
        model = svm.SVC(kernel='rbf').fit(xTrain, yTrain)
    elif kernel == 'linear':
        model = svm.SVC(kernel='linear').fit(xTrain, yTrain)
    else: model = svm.SVC(kernel='poly', degree=3).fit(xTrain, yTrain)

    if crossValid:
        print cross_val_score(model, xTrain, yTrain, cv=10)

    else:
        predictions = model.predict(np.array(xTest))
        print kernel, "kernel SVM's right prediction percentage with oneVsone strategy: ", get_predition_percentage(predictions, yTest)
    return model



























