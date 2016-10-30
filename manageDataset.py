from __future__ import division
from dataStructures import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import mySeed

def get_lagrange_dataset(training, test, pairTarget, percentageLabel, pairwise = False):
    #dataset = all training set
    #pair target is a list of 2 classes, set None if you are not using pairwise
    #percentage of labeled datapoints
    #set True if you want to do a pairwise comparation


    if pairwise:
        pairTrain = [datapoint for datapoint in training if datapoint[-1] in pairTarget]
        pairTest = [datapoint for datapoint in test if datapoint[-1] in pairTarget]
        l, u = get_l_u(pairTrain, percentageLabel)
        data = pairTrain+pairTest
        dataArray = np.array(data)
        return get_pairwise_dataset(dataArray, l, u)
    else: #todo call others
        print 'others'

def get_pairwise_dataset(dataset, l, u):

    data = dataset[:, :(len(dataset[0]) - 1)]
    data = MinMaxScaler(feature_range=(0,1)).fit_transform(data)
    targets = dataset[:, len(dataset[0]) - 1 ]
    targets = np.array([1 if y==targets[0] else -1 for y in targets])
    return data[:l], targets[:l], data[l:l+u], data[l+u:], targets[l+u:], l, u

def get_qn_dataset(training, test, percentageLabel):

    #you must have a list of numpy arrays
    l, u = get_l_u(training, percentageLabel)
    trainArray = np.array(training)
    testArray = np.array(test)

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
