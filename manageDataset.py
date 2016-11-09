from __future__ import division
from dataStructures import *
import numpy as np
import mySeed

def get_lagrange_pairwise(training, test, testScaler, percentageLabel, pairTarget):
    #pair target is a list of 2 classes
    #percentage of labeled datapoints
    #use testScaler for transforming the test set to the valid one

    pairTrain, pairTest, l, u = get_pair_dataset(training, test, percentageLabel, pairTarget)
    trainArray = np.array(pairTrain)
    testArray = np.array(pairTest)
    datapointsTrain = trainArray[:, :(len(trainArray[0]) - 1)]
    targetsTrain = trainArray[:, len(trainArray[0]) - 1]
    datapointsTest = testArray[:, :(len(trainArray[0]) - 1)]
    targetsTest = testArray[:, len(trainArray[0]) - 1]

    #transforms the test set
    datapointsTest = testScaler.transform(datapointsTest)

    return datapointsTrain[:l], targetsTrain[:l], datapointsTrain[l:l + u], datapointsTest, targetsTest, l, u


#it gets the prepared training where one label is set to 1 and others to -1
def get_lagrange_oneVsAll(trainingSet, percentageLabel):

    training = copy.copy(trainingSet)
    l, u = get_l_u(training, percentageLabel)
    trainArray = np.array(training)
    datapointsTrain = trainArray[:, :(len(trainArray[0]) - 1)]
    targetsTrain = trainArray[:, len(trainArray[0]) - 1]

    return datapointsTrain[:l], targetsTrain[:l], datapointsTrain[l:l + u], l, u


def get_qn_dataset(training, test, percentageLabel, pairTarget):

    #you must have a list of numpy arrays
    pairTrain, pairTest, l, u = get_pair_dataset(training, test, percentageLabel, pairTarget)
    trainArray = np.array(pairTrain)
    testArray = np.array(pairTest)
    #create arrays for labels qn needs int
    xTrainArrayL = trainArray[:l, :(len(trainArray[0]) - 1)]
    y = trainArray[:l,(len(trainArray[0]) - 1)]
    yTrainArrayL = y.astype(int)
    xTrainArrayU = trainArray[l:, :(len(trainArray[0]) - 1)]
    xTestArray = testArray[:, :(len(trainArray[0]) - 1)]
    y = testArray[:, len(testArray[0]) - 1]
    yTestArray = y.astype(int)
    #qn needs a list of arrays
    xTrainL = append_array_to_new_list(xTrainArrayL)
    yTrainL = append_array_to_new_list(yTrainArrayL)
    xTrainU = append_array_to_new_list(xTrainArrayU)
    xTest = append_array_to_new_list(xTestArray)
    yTest = append_array_to_new_list(yTestArray)

    return xTrainL, yTrainL, xTrainU, xTest, yTest

#It can be used for all supervised algorithms and it returns arrays of train and test and relative targets
def get_supervisedAlgorithm_dataset (training, test):

    xTrain, yTrain = get_data_target_lists(training)
    xTest, yTest = get_data_target_lists(test)
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    return xTrain, yTrain, xTest, yTest

#creates a dictionary where the key is a class and the value is the training set with that label
#set to 1 and others set to -1
def get_multiclass_dataset(dataset, targets, targetsDic):
    #mySeed.rdm.shuffle(dataset)
    dataDic = {}
    for i in range(len(targetsDic)):
        data = []
        for item in dataset:
            datapoint = copy.copy(item)
            if datapoint[-1] == targetsDic[targets[i]]:
                datapoint[-1] = 1
            else: datapoint[-1] = -1
            data.append(datapoint)

            dataDic[targets[i]] = data
    return dataDic


