from __future__ import division
import os
import csv
import copy
import mySeed
import random


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
    mySeed.rdm.shuffle(dataset)
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


# training, test = split_dataset(dataset, 50)
# trainingArray = np.array([datapoint for datapoint in training if datapoint[-1] in targets])
# testArray = np.array([datapoint for datapoint in test if datapoint[-1] in targets])
# dataset = np.concatenate((trainingArray, testArray))





