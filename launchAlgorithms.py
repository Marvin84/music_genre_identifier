
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_score
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from manageDataset import *
from utilities import *
from estimators import *




#
# starting with supervised algorithm
#

def launch_KNN(training, test, testScaler):

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    xTest = testScaler.transform(np.array(xTest))



    scores = []
    models = []
    for i in range(1, 5):
        model = KNeighborsClassifier(n_neighbors=i).fit(xTrain, yTrain)
        models.append(model)
        predictions = model.predict(xTest)
        print(np.unique(yTest))
        scores.append(f1_score(yTest, predictions, average='weighted', labels=np.unique(predictions)))

    bestIndex = scores.index(np.amax(np.array(scores)))
    print("KNN's f1: ", scores[bestIndex], "with ", bestIndex + 1, "neighbors")



def launch_SVM_SVC(training, test, testScaler, kernel):
    # here the multiclass is supported by one vs one
    # gamma must be set only for rbf and poly

    xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
    bestEstimator = get_my_best_estimator(xTrain, yTrain, 5, kernel)
    xTest = testScaler.transform(xTest)

    if kernel == 'rbf':
        model = SVC(C=bestEstimator.C, decision_function_shape='ovo',
                        gamma=bestEstimator.gamma, verbose=False).fit(xTrain, yTrain)
        predictions = model.predict(xTest)
        print (kernel, "kernel SVM's f1 using predict function: ", f1_score(predictions, yTest, average='weighted', labels=np.unique(predictions)))
    elif kernel == 'linear':
        model = SVC(C=bestEstimator.C, decision_function_shape='ovo',
                        kernel='linear', verbose=False).fit(xTrain, yTrain)
        predictions = model.predict(xTest)
        print (kernel, "kernel SVM's f1 using predict function: ", f1_score(predictions, yTest, average='weighted', labels=np.unique(predictions)))

    else:
        'unvalid value for kernel'


#
#
#
#
# # using the oneVsRestClassifier of SVM
# def launch_SVM_oneVsall(dataset, training, test, testScaler, crossValid=False):
#     X = dataset[:, :-1]
#     y = dataset[:, -1]
#     xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
#     bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
#     trainScaler = MinMaxScaler(feature_range=(-1, 1))
#     xTest = testScaler.transform(xTest)
#
#     model = OneVsRestClassifier(SVC(C=bestEstimator.C, kernel='rbf', gamma=bestEstimator.gamma)).fit(xTrain, yTrain)
#     predictions = model.predict(np.array(xTest))
#
#     if crossValid:
#         print "oneVsRestClassifier SVM 10-fold cross validation: ", \
#             get_average(cross_val_score(model, X, y, cv=10)) * 100
#     print "oneVsRestClassifier SVM 10-fold cross validation on training set: ", \
#         get_average(cross_val_score(model, xTrain, yTrain, cv=10)), \
#         "oneVsRestClassifier SVM's accuracy: ", accuracy_score(predictions, yTest) * 100
#     return model
#
#
# # using the oneVsOneClassifier of SVM
# def launch_SVM_oneVsone(dataset, training, test, testScaler, crossValid=False):
#     X = dataset[:, :-1]
#     y = dataset[:, -1]
#     xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
#     bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
#     trainScaler = MinMaxScaler(feature_range=(-1, 1))
#     X = trainScaler.fit_transform(X)
#     xTest = testScaler.transform(xTest)
#
#     model = OneVsOneClassifier(SVC(C=bestEstimator.C, gamma=bestEstimator.gamma)).fit(xTrain, yTrain)
#     predictions = model.predict(np.array(xTest))
#
#     if crossValid:
#         print "oneVsoneClassifier SVM 10-fold cross validation: ", \
#             get_average(cross_val_score(model, X, y, cv=10)) * 100
#
#     print "oneVsoneClassifier SVM 10-fold cross validation on training set: ", \
#         get_average(cross_val_score(model, xTrain, yTrain, cv=10)), \
#         "oneVoneClassifier SVM's accurary: ", accuracy_score(predictions, np.array(yTest)) * 100
#     return model
#
#
# # using the SVCLinear
# def launch_SVCLinear(dataset, training, test, testScaler, crossValid=False):
#     X = dataset[:, :-1]
#     y = dataset[:, -1]
#     xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
#     bestEstimator = get_LinearSVC_best_estimator(xTrain, yTrain)
#     trainScaler = MinMaxScaler(feature_range=(-1, 1))
#     X = trainScaler.fit_transform(X)
#     xTest = testScaler.transform(np.array(xTest))
#
#     model = LinearSVC(C=bestEstimator.C, random_state=42).fit(xTrain, yTrain)
#     predictions = model.predict(xTest)
#
#     if crossValid:
#         print "SVCLinear 10-fold cross validation: ", get_average(cross_val_score(model, X, y, cv=10)) * 100
#
#     print "SVCLinear 10-fold cross validation on training set: ", get_average(
#         cross_val_score(model, xTrain, yTrain, cv=10)), \
#         "SVCLinear's accurary: ", accuracy_score(predictions, yTest) * 100
#     return model
#
# def launch_svm(dataset, training, test, testScaler, crossValid=False):
#
#     X = dataset[:, :-1]
#     y = dataset[:, -1]
#     xTrain, yTrain, xTest, yTest = get_supervised_dataset(training, test)
#     bestEstimator = get_my_best_estimator(xTrain, yTrain, 10)
#     trainScaler = MinMaxScaler(feature_range=(-1, 1))
#     X = trainScaler.fit_transform(X)
#     xTest = testScaler.transform(xTest)
#
#     model = SVC(C=bestEstimator.C, gamma=bestEstimator.gamma).fit(xTrain, yTrain)
#     predictions = model.predict(xTest)
#
#     if crossValid:
#         print "SVM 10-fold cross validation: ", get_average(cross_val_score(model, X, y, cv=10)) * 100
#
#     print "SVM 10-fold cross validation on training set: ", \
#         get_average(cross_val_score(model, xTrain, yTrain, cv=10))
#     print "SVM's accuracy using predict function: ", accuracy_score(predictions, yTest) * 100
#     return model














