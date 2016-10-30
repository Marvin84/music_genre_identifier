from __future__ import division
from dataStructures import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def get_lagrange_dataset(dataset, pairTarget, percentageLabel, pairwise = False):


    if pairwise:
        pairDataset = [datapoint for datapoint in dataset if datapoint[-1] in pairTarget]
        l, u = get_l_u(pairDataset, percentageLabel)
        dataArray = np.array(pairDataset)
        return get_pairwise_dataset(dataArray, l, u)
    else: #todo call others
        print 'others'

def get_pairwise_dataset(dataset, l, u):


    rdm = np.random.RandomState()
    rdm.shuffle(dataset)
    data = dataset[:, :(len(dataset[0]) - 1)]
    data = MinMaxScaler(feature_range=(0,1)).fit_transform(data)
    targets = dataset[:, len(dataset[0]) - 1 ]
    targets = np.array([1 if y==targets[0] else -1 for y in targets])
    return data[:l], targets[:l], data[l:l+u], data[l+u:], targets[l+u:], l, u

