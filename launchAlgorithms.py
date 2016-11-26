from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from manageDataset import *
from utilities import *
from estimators import *
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

    if pairwise:
        return model, model.score(xTest,yTest)
    else: return model



#use the launch_lagrange as subroutine for training evry model, pairwise is with default value False
def launch_oneVsRest_lagrange(training, test, testScaler, targets, targetsDic, percentageLabel, r):

    models = []
    multiclassTraining = get_multiclass_dataset(training, targets, targetsDic)

    #transformation of testset with the pre-calculated test scaler
    xTest, yTest = get_data_target_lists(test)
    xTestArray = testScaler.transform(np.array(xTest))

    #training models
    for i in range(len(targets)):
       models.append(launch_lagrange(multiclassTraining[targets[i]], test, testScaler, percentageLabel, r))

    predictions = []
    for i in range(len(targets)):
        predictions.append(models[i].predict(xTestArray))

    #getting the predicted class by taking the highest value of the decision function
    decisions =[]
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

    print "langrange's accuracy percentage with oneVsRest strategy: ", accuracy_score(predictions, yTest)
    return models

#use the launch_lagrange as subroutine for training evry model, setting pairwise True
def launch_oneVsone_lagrange(training, test, testScaler, targets, percentageLabel, r):

    scores = []
    for i in range(len(targets)):
        for j in range(len(targets)):
            if i!=j:
                scores.append(launch_lagrange(training, test, testScaler, percentageLabel, r, True, [i+1,j+1])[1])
            else: pass
    return get_average(scores)


def launch_qn(training, test, percentageLabel, r, pairTarget = None):
    xTrainL, yTrainL, xTrainU, xTest, yTest = get_qn_dataset(training, test, percentageLabel, pairTarget)
    model = QN_S3VM(xTrainL, yTrainL, xTrainU, qn.randomGenerator.my_random_generator, lam=0.0009765625, lamU=1,
                    kernel_type="RBF", sigma=0.5, estimate_r=r )
    model.train()
    preds = model.getPredictions(xTest)
    error = classification_error(preds,yTest)
    return model, error


#use the launch_lagrange as subroutine for training evry model, setting pairwise True
def launch_oneVsone_qn(training, test, targets, percentageLabel, r):

    errors = []
    for i in range(len(targets)):
        for j in range(len(targets)):
            if i!=j:
                errors.append(launch_qn(training, test, percentageLabel, r, [i+1,j+1])[1])
            else: pass
    return reduce(lambda x, y: x + y, errors) / len(errors)


#
#starting with supervised algorithm
#

def launch_KNN (training, test, testScaler, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    model = KNeighborsClassifier(n_neighbors=7).fit(xTrain, yTrain)
    xTestArray = testScaler.transform(np.array(xTest))

    if crossValid:
     print"todo "#todo
    predictions = model.predict(xTestArray)
    print "KNN's accuracy: ", accuracy_score(predictions, np.array(yTest))*100
    return model


def launch_SVM_SVC (training, test, testScaler, kernel, crossValid = False):
    #here the multiclass is supported by one vs one
    #gamma must be set only for rbf and poly
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_best_estimator_by_cv (xTrain, yTrain, 10)

    if kernel == 'rbf':
        model = svm.SVC(C=bestEstimator.C, decision_function_shape = 'ovo',
                        gamma=bestEstimator.gamma, random_state=42, verbose=False).fit(xTrain, yTrain)
    elif kernel == 'linear':
        model = svm.SVC(C=bestEstimator.C, decision_function_shape = 'ovo',
                        kernel='linear', random_state=42, verbose=False).fit(xTrain, yTrain)
    else: 'unvalid value for kernel'

    predictionsDec = []
    #decisions = model.decision_function(xTest).tolist()
    #for item in decisions:
    #    maxElement = max(item)
    #    predictionsDec.append(item.index(maxElement)+1)
    #score = model.score(xTest, yTest)
    #print "Accuracy of ", kernel, "kernel SVC using onevsRest decision function : ",\
    #    accuracy_score(predictionsDec, yTest),"and the score is: ", score

    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(xTest)
    bestModel = get_best_model(xTrain, yTrain)
    bestPredictions = bestModel.predict(xTest)
    print kernel, "kernel SVC's accuracy using predict function: ", accuracy_score(predictions, yTest)*100
    print "and best model: ", accuracy_score(bestPredictions, yTest)
    return model


#using the oneVsRestClassifier of SVM
def launch_SVM_oneVsall (training, test, testScaler, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_best_estimator_by_cv(xTrain, yTrain, 10)
    model = OneVsRestClassifier(svm.SVC(C=bestEstimator.C, kernel='rbf', gamma=bestEstimator.gamma, random_state=42)).fit(xTrain, yTrain)

    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(np.array(xTest))
    print "oneVsRestClassifier SVM's accuracy: ", accuracy_score(predictions, yTest)*100
    return model


#using the oneVsOneClassifier of SVM
def launch_SVM_oneVsone(training, test, testScaler, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_best_estimator_by_cv(xTrain, yTrain, 10)
    model = OneVsOneClassifier(svm.SVC(C=bestEstimator.C, kernel='linear', random_state=42)).fit(xTrain, yTrain)
    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(np.array(xTest))
    print "oneVoneClassifier SVM's accurary: ", accuracy_score(predictions, np.array(yTest))*100
    return model

#using the SVCLinear
def launch_SVCLinear(training, test, testScaler, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_LinearSVC_best_estimator(xTrain, yTrain)
    xTestArray = testScaler.transform(np.array(xTest))
    model = LinearSVC(C=bestEstimator.C, random_state=42).fit(xTrain, yTrain)
    if crossValid:
     print"todo "#todo
    predictions = model.predict(xTestArray)
    print "SVCLinear's accurary: ", accuracy_score(predictions, yTest)*100
    return model

#using the stochastic
def launch_SGDClassifier(training, test, testScaler, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_SGDC_best_estimator(xTrain, yTrain)
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=bestEstimator.alpha, random_state=42).fit(xTrain, yTrain)

    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(xTest)
    print "SGDClassifier's accuracy: ", accuracy_score(predictions, np.array(yTest))*100
    return model



def launch_gradientBoost (training, test, testScaler, crossValid):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.0829819533259, loss='deviance', max_depth=4,
              max_features='sqrt', max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=946, presort='auto', random_state=4,
              subsample=1.0, verbose=0, warm_start=False).fit(xTrain, yTrain)

    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(xTest)
    print "gradientBoost's accuracy: ", accuracy_score(predictions, np.array(yTest))*100
    return model


def launch_extraTrees (training, test, testScaler, crossValid):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    model = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
           max_depth=None, max_features=0.667097169797,
           max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=5, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=364, n_jobs=1,
           oob_score=False, random_state=3, verbose=False,
           warm_start=False).fit(xTrain, yTrain)

    if crossValid:
     print"todo "#todo
    xTest = testScaler.transform(xTest)
    predictions = model.predict(xTest)
    print "ExtraTrees's accuracy: ", accuracy_score(predictions, np.array(yTest))*100
    return model

def get_best_model(xTrain, yTrain):
    model = SVC(C=96959.5216309, cache_size=512, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=1, gamma=0.188129827352,
        kernel='rbf', max_iter=599598507.0, probability=False, random_state=0,
        shrinking=False, tol=0.00873620944293, verbose=False)
    #model = svm.SVC(C=156.419002877, cache_size=512, class_weight=None, coef0=0.0,
    #    decision_function_shape=None, degree=1, gamma='auto', kernel='linear',
    #    max_iter=22073714.0, probability=False, random_state=3, shrinking=True,
    #    tol=0.000212752230313, verbose=False)
    #model = SVC(C=68.1615839242, cache_size=512, class_weight=None, coef0=1081.71929367,
    #    decision_function_shape=None, degree=5.0, gamma=109.368508147,
    #    kernel='poly', max_iter=285102549.0, probability=False, random_state=4,
    #    shrinking=False, tol=0.000970574688949, verbose=False)

    model.fit(xTrain, yTrain)
    return model


