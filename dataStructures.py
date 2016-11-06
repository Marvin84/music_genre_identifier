from __future__ import division
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.grid_search import RandomizedSearchCV
from sklearn import svm
import numpy as np
import os
import csv
import copy
import mySeed


#the dataset must be a csv file with first raw containing all attributes
#the class attribute like the last attribute of the raw
def read_file(filename):
 #create a set of tuples which contain the raws
 with open(os.path.dirname(os.path.abspath(__file__)) + filename) as file:
  datalist=[list(line) for line in csv.reader(file)]
  #created the tuples without the first raw which contains the attributes
  attributes = datalist[0]
  del datalist[0]
 #creates a dictionary with 'key:values' where key is the attribute and value its relatives values
 with open(os.path.dirname(os.path.abspath(__file__)) + filename, 'r') as csvin:
  reader=csv.DictReader(csvin)
  coloumns={k:[v] for k,v in reader.next().items()}
  datadomain={k:[v] for k,v in reader.next().items()}
  for line in reader:
   for k,v in line.items():
    coloumns[k].append(v)
    if v not in datadomain[k]:
     datadomain[k].append(v)
 return datalist, attributes, coloumns

#return a list with all targets
def unrepeated_values(list):
    unique = []
    for item in list:
     if item not in unique:
      unique.append(item)
    return unique

#creates a new dataset where the class attribute is replaced with
# a number from 1 to n where n is number of classes
def switch_label(dataset, classes):
    n = len(classes)
    newDataset = []
    for item in dataset:
        datapoint = copy.copy(item)
        for i in range(n):
            if datapoint[-1] == classes[i]:
                datapoint[-1]= str(i + 1)
                break


        newDataset.append(datapoint)
    return newDataset

#convert a list of lists of strings to float
def string_to_float(dataset):
 dataset = [[float(item) for item in raws] for raws in dataset]
 for item in dataset:
     item[-1] = int(item[-1])
 return dataset

#creates a csv file named filename from a list
def get_csv(dataset, filename):
    with open(filename + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

#split a generic list in 2 lists. n is the length of the first list
def splitter(myList, n):
    firstList = myList[0:n]
    secondList = myList[n:]
    return (firstList, secondList)

#split datasets in training and test wrt a user defined percentage for training
#user can decide to have as test set the first part of dataset by setting swap to true

def split_dataset(dataset, trainPercentage, swap = False):

    #randomize the raws
    mySeed.rdm.shuffle(dataset)
    #training percentage
    m = get_number_from_percentage(len(dataset), trainPercentage)
    if swap:
        test, training = splitter(dataset, len(dataset) - m)
    else: training, test = splitter(dataset, m)
    return (training, test)


def get_number_from_percentage(listLength, percentage):
    return int((listLength* percentage) / 100)


def get_l_u(dataset, percentageLabel):
    l = get_number_from_percentage(len(dataset), percentageLabel)
    u = len(dataset) - l
    return (l,u)

def append_array_to_new_list(myArray):
    list = []
    for item in myArray:
        list.append(item)
    return list

#it takes a list of classes refered to the class coloumn of dataset
#returns 2 values a dictionary with key : value as name of class : a number
# and the list of number associetaed with the classes
def  get_classes(classes):
    targets = sorted(unrepeated_values(classes))
    d = {}
    for i in range(len(targets)):
        d[targets[i]] = i+1
    return d, targets

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

def get_data_target_lists(dataset):
    xTest = []
    yTest = [int(y[len(dataset[0])-1]) for y in dataset]
    for item in dataset:
        x = copy.copy(item)
        del x[len(item)-1]
        xTest.append(x)
    return xTest, yTest

def count_labels(dataset, targets):
    count = [0] * len(targets)
    for i in range(len(dataset)):
        count[dataset[i]-1] += 1
    return count

def get_numbered_classes(targets):
    numberTargets = []
    for i in range(len(targets)):
        numberTargets.append(i+1)
    return numberTargets

#it creates a new list and change the labels to 1 and -1
def binary_targets(dataset, targets):
    binaryDataset = copy.copy(dataset)
    for item in binaryDataset:
        if item[-1] == targets[0]:
            item[-1] = 1
        else: item[-1] = -1
    return binaryDataset



def get_pair_dataset(training, test, percentageLabel, pairTarget):


    pairTrain = [copy.copy(datapoint) for datapoint in training if datapoint[-1] in pairTarget]
    pairTrain = binary_targets(pairTrain, pairTarget)
    pairTest = [copy.copy(datapoint) for datapoint in test if datapoint[-1] in pairTarget]
    pairTest = binary_targets(pairTest, pairTarget)

    l, u = get_l_u(pairTrain, percentageLabel)

    return pairTrain, pairTest, l, u



# getting the scaled train and test sets and the test scaler
# training was fit and transformed the test only fit
def get_scaleDataset_and_scalers(train, test):
    training = copy.copy(train)
    test = copy.copy(test)
    # instantiate the MinMax instance for training and for test
    mmTrain = MinMaxScaler(feature_range=(0,1))
    mmTest = MinMaxScaler(feature_range=(0,1))
    trainArray = np.array(training)
    testArray = np.array(test)
    trainData, trainTarget = trainArray[:, :(len(trainArray[0]) - 1)], trainArray[:, len(trainArray[0]) - 1]
    trainData = mmTrain.fit_transform(trainData)
    mmTest.fit(testArray[:, :(len(testArray[0]) - 1)])
    trainArray = np.c_[trainData, trainTarget]
    training = trainArray.tolist()
    for item in training:
        item[-1] = int(item[-1])


    return training, mmTrain, mmTest

#two lists
def get_predition_percentage(predictions, test):
    n = len([i for i, j in zip(predictions, test) if i == j])
    return (n / len(predictions)) * 100

def get_copy_lists(training, test):
    return copy.copy(training), copy.copy(test)


def scale_stochastic_dataset(xTrain, xTest):
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    return xTrain, xTest

def get_SGDC_best_estimator(xTrain, yTrain):

    model = Ridge()
    parameters = {'alpha': sp_rand()}
    rdmSearch = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100)
    rdmSearch.fit(xTrain, yTrain)
    #the pre defined alphas is worse!!
    #alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    #grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    #grid.fit(xTrain, yTrain)
    return rdmSearch.best_estimator_

def get_LinearSVC_best_estimator(xTrain, yTrain):

    model = svm.LinearSVC()
    parameters = {'C': (.1, .5, 1.0)}
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(xTrain, yTrain)
    return grid.best_estimator_






