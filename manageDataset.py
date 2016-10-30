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

