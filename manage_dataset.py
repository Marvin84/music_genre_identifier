import numpy as np
from __future__ import division
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_lagrange_dataset(dataset, classes, l, u, pairwise = False):

    if pairwise:
        get_pairwise_dataset(dataset, classes, l, u)
    else: #todo call others
        print 'othe'

def get_pairwise_dataset(dataset, targets, l, u):
    rdm = np.random.RandomState()
    #training data
    training_set = []
    #todo
    #test data
    test_set = []
    #merging sets to allow custom resplit
    samples = np.concatenate((training_set, test_set))
    rdm.shuffle(samples)
    data = samples[:, 1:]
    data = MinMaxScaler(feature_range=(0,1)).fit_transform(data)
    targets = samples[:, 0]
    targets = np.array([1 if y==targets[0] else -1 for y in targets])
    return data[:l], targets[:l], data[l:l+u], data[l+u:], targets[l+u:]

