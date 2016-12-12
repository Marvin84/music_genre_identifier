from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from manageDataset import *
from utilities import *
from estimators import *
from lagrange.lagrangean_s3vm import *
from qn.qns3vm import *
from qn.examples import *
import qn.randomGenerator
import mySeed


#
# starting with supervised algorithm
#

def launch_KNN(dataset, training, test, testScaler, crossValid=False):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(np.array(xTest))

    scores = []
    models = []
    for i in range(1, 8):
        model = KNeighborsClassifier(n_neighbors=i).fit(xTrain, yTrain)
        models.append(model)
        predictions = model.predict(xTest)
        scores.append(accuracy_score(predictions, np.array(yTest)))
    bestIndex = scores.index(np.amax(np.array(scores)))

    if crossValid:
        print "KNN 10-fold cross validation: ", \
            get_average(cross_val_score(KNeighborsClassifier(n_neighbors=bestIndex + 1), X, y, cv=10)) * 100

    print "KNN SVM 10-fold cross validation on training set: ", \
        get_average(cross_val_score(models[bestIndex], xTrain, yTrain, cv=10)), \
        "KNN's accuracy: ", scores[bestIndex] * 100, "with ", bestIndex+1, "neighbors"
    return model


def launch_SVM_SVC(dataset, training, test, testScaler, kernel, crossValid=False):
    # here the multiclass is supported by one vs one
    # gamma must be set only for rbf and poly

    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(xTest)

    if kernel == 'rbf':
        model = SVC(C=bestEstimator.C, decision_function_shape='ovo',
                        gamma=bestEstimator.gamma, verbose=False).fit(xTrain, yTrain)
    elif kernel == 'linear':
        model = SVC(C=bestEstimator.C, decision_function_shape='ovo',
                        kernel='linear', verbose=False).fit(xTrain, yTrain)

    else:
        'unvalid value for kernel'

    predictions = model.predict(xTest)
    if crossValid:
        print "SVM 10-fold cross validation: ", \
            get_average(cross_val_score(model, X, y, cv=10)) * 100

    print kernel, "kernel SVM 10-fold cross validation on training set: ", \
        get_average(cross_val_score(model, xTrain, yTrain, cv=10)) * 100
    print kernel, "kernel SVM's accuracy using predict function: ", accuracy_score(predictions, yTest) * 100


# using the oneVsRestClassifier of SVM
def launch_SVM_oneVsall(dataset, training, test, testScaler, crossValid=False):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(xTest)

    model = OneVsRestClassifier(SVC(C=bestEstimator.C, kernel='rbf', gamma=bestEstimator.gamma)).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))

    if crossValid:
        print "oneVsRestClassifier SVM 10-fold cross validation: ", \
            get_average(cross_val_score(model, X, y, cv=10)) * 100
    print "oneVsRestClassifier SVM 10-fold cross validation on training set: ", \
        get_average(cross_val_score(model, xTrain, yTrain, cv=10)), \
        "oneVsRestClassifier SVM's accuracy: ", accuracy_score(predictions, yTest) * 100
    return model


# using the oneVsOneClassifier of SVM
def launch_SVM_oneVsone(dataset, training, test, testScaler, crossValid=False):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(xTest)

    model = OneVsOneClassifier(SVC(C=bestEstimator.C, gamma=bestEstimator.gamma)).fit(xTrain, yTrain)
    predictions = model.predict(np.array(xTest))

    if crossValid:
        print "oneVsoneClassifier SVM 10-fold cross validation: ", \
            get_average(cross_val_score(model, X, y, cv=10)) * 100

    print "oneVsoneClassifier SVM 10-fold cross validation on training set: ", \
        get_average(cross_val_score(model, xTrain, yTrain, cv=10)), \
        "oneVoneClassifier SVM's accurary: ", accuracy_score(predictions, np.array(yTest)) * 100
    return model


# using the SVCLinear
def launch_SVCLinear(dataset, training, test, testScaler, crossValid=False):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_LinearSVC_best_estimator(xTrain, yTrain)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(np.array(xTest))

    model = LinearSVC(C=bestEstimator.C, random_state=42).fit(xTrain, yTrain)
    predictions = model.predict(xTest)

    if crossValid:
        print "SVCLinear 10-fold cross validation: ", get_average(cross_val_score(model, X, y, cv=10)) * 100

    print "SVCLinear 10-fold cross validation on training set: ", get_average(
        cross_val_score(model, xTrain, yTrain, cv=10)), \
        "SVCLinear's accurary: ", accuracy_score(predictions, yTest) * 100
    return model

def launch_svm(dataset, training, test, testScaler, crossValid=False):

    X = dataset[:, :-1]
    y = dataset[:, -1]
    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
    trainScaler = MinMaxScaler(feature_range=(-1, 1))
    X = trainScaler.fit_transform(X)
    xTest = testScaler.transform(xTest)

    model = SVC(C=bestEstimator.C, gamma=bestEstimator.gamma).fit(xTrain, yTrain)
    predictions = model.predict(xTest)

    if crossValid:
        print "SVM 10-fold cross validation: ", get_average(cross_val_score(model, X, y, cv=10)) * 100

    print "SVM 10-fold cross validation on training set: ", \
        get_average(cross_val_score(model, xTrain, yTrain, cv=10))
    print "SVM's accuracy using predict function: ", accuracy_score(predictions, yTest) * 100
    return model






'''
Starting with semi-supervised
'''

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








