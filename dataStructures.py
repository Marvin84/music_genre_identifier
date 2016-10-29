import os
import csv
import copy
import random
import numpy as np

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

def get_csv(dataset):
    with open("newDataset.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)