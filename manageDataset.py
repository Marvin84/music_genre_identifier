from __future__ import division
from dataStructures import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import mySeed

def get_lagrange_dataset(training, test, percentageLabel, pairTarget, pairwise = False):
    #pair target is a list of 2 classes, set None if you are not using pairwise
    #percentage of labeled datapoints
    #set True if you want to do a pairwise comparation


    if pairwise:
        pairTrain, pairTest, l, u = get_pair_dataset(training, test, percentageLabel, pairTarget)
        set = pairTrain+pairTest
        dataArray = np.array(set)
        targets = dataArray[:, len(dataArray[0]) - 1]
        targets = np.array([1 if y == targets[0] else -1 for y in targets])

    else:
        #here the training and test are already ready
        l, u = get_l_u(training, percentageLabel)
        dataArray = np.array(training)
        targets = dataArray[:, len(dataArray[0]) - 1]


    datapoints = dataArray[:, :(len(dataArray[0]) - 1)]
    datapoints = MinMaxScaler(feature_range=(0, 1)).fit_transform(datapoints)



    if pairwise:
        return datapoints[:l], targets[:l], datapoints[l:l + u], datapoints[l + u:], targets[l + u:], l, u
    else: return datapoints[:l], targets[:l], datapoints[l:l + u], l, u




def get_qn_dataset(training, test, percentageLabel, pairwise, pairTarget):

    #you must have a list of numpy arrays
    if pairwise:
        pairTrain, pairTest, l, u = get_pair_dataset(training, test, percentageLabel, pairTarget)
        trainArray = np.array(pairTrain)
        testArray = np.array(pairTest)
    else: print "not pairwise version"

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
