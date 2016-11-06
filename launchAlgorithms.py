from __future__ import division
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from manageDataset import *
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

    print "langrange's right prediction percentage with oneVsRest strategy: ",get_predition_percentage(predictions, yTest)
    return models

#use the launch_lagrange as subroutine for training evry model, setting pairwise True
def launch_oneVsone_lagrange(training, test, testScaler, targets, percentageLabel, r):

    scores = []
    for i in range(len(targets)):
        for j in range(len(targets)):
            if i!=j:
                scores.append(launch_lagrange(training, test, testScaler, percentageLabel, r, True, [i+1,j+1])[1])
            else: pass
    return reduce(lambda x, y: x + y, scores) / len(scores)


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
#starting with supervised algorithms
#


def launch_KNN (training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    model = KNeighborsClassifier().fit(xTrain, yTrain)

    if crossValid:
        print "KNN's score with 5-fold cross validation: ",cross_val_score(model, xTrain, yTrain, cv=5)

    else:
        predictions = model.predict(np.array(xTest))
        print "KNN's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model


def launch_SVM_SVC (training, test, kernel, crossValid = False):
    #here the multiclass is supported by one vs one
    #gamma must be set only for rbf and poly
    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    best_estimator = get_best_estimator_by_cv(xTrain, yTrain, 5)

    if kernel == 'rbf':
        model = svm.SVC(C=best_estimator.C, decision_function_shape = 'ovr',
                        gamma=best_estimator.gamma).fit(xTrain, yTrain)
    elif kernel == 'linear':
        model = svm.SVC(C=best_estimator.C, decision_function_shape = 'ovr',
                        gamma=best_estimator.gamma, kernel='linear').fit(xTrain, yTrain)
    else: model = svm.SVC(C=best_estimator.C, decision_function_shape = 'ovr',
                          kernel='poly', degree=3, gamma=best_estimator.gamma).fit(xTrain, yTrain)

    predictionsDec = []
    decisions = model.decision_function(xTest).tolist()
    for item in decisions:
        maxElement = max(item)
        predictionsDec.append(item.index(maxElement)+1)

    percentage = get_predition_percentage(predictionsDec, yTest)
    score = model.score(xTest, yTest)
    print "percentage of ", kernel, "kernel SVC using onevsRest decision function : ", percentage, "and the score is: ", score

    if crossValid:
        print kernel, "kernel SVM's score with 5-fold cross validation: ", cross_val_score(model, xTrain, yTrain, cv=5)
    else:
        predictions = model.predict(np.array(xTest))
        print kernel, "kernel SVC's right prediction percentage using predict function: ", \
            get_predition_percentage(predictions, yTest)
    return model


#using the oneVsRestClassifier of SVM
def launch_SVM_oneVsall (training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    best_estimator = get_best_estimator_by_cv(xTrain, yTrain, 5)
    model = OneVsRestClassifier(svm.SVC(C=best_estimator.C, kernel='linear', gamma=best_estimator.gamma)).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))

    if crossValid:
        print "oneVsRestClassifier SVM's right prediction percentage and 5-fold cross validation: ", \
            cross_val_score(model, xTrain, yTrain, cv=5)
    else:
        print "oneVsRestClassifier SVM's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model


#using the oneVsOneClassifier of SVM
def launch_SVM_oneVsone(training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    best_estimator = get_best_estimator_by_cv(xTrain, yTrain, 5)
    model = OneVsOneClassifier(svm.SVC(C=best_estimator.C, kernel='linear', gamma=best_estimator.gamma)).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))
    if crossValid:
        print "oneVoneClassifier SVM's right prediction percentage and 5-fold cross validation: ", \
            cross_val_score(model, xTrain, yTrain, cv=5)
    else:
        print "oneVoneClassifier SVM's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model

#using the SVCLinear
def launch_SVCLinear(training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    bestEstimator = get_LinearSVC_best_estimator(xTrain, yTrain)
    #the best estimator with SVC() model is worse!!
    #bestEstimator = get_best_estimator_by_cv(xTrain, yTrain, 3)
    model = LinearSVC(C=bestEstimator.C, random_state=1).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))
    if crossValid:
        print "SVCLinear's right prediction percentage and 5-fold cross validation: ", \
            cross_val_score(model, xTrain, yTrain, cv=5)
    else:
        print "SVCLinear's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model

#using the stochastic
def launch_SGDClassifier(training, test, crossValid = False):

    xTrain, yTrain, xTest, yTest = get_supervisedAlgorithm_dataset(training, test)
    bestEstimator = get_SGDC_best_estimator(xTrain, yTrain)
    xTrain, xTest = scale_stochastic_dataset(xTrain, xTest)
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=bestEstimator.alpha, random_state=42).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))
    if crossValid:
        print "SGDClassifier's right prediction percentage and 5-fold cross validation: ", \
            cross_val_score(model, xTrain, yTrain, cv=5)
    else:
        print "SGDClassifier's right prediction percentage: ", get_predition_percentage(predictions, yTest)
    return model
