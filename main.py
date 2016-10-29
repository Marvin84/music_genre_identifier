from dataStructures import *
from lagrangean_s3vm import *
from utils import *



if __name__ == "__main__":

    #datasetList is a list of lists of datas,
    #attributes is the first line
    #coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    #extract which are the classes and order it alphabetically
    targets = coloumns['class']
    classes_string = sorted(unrepeated_values(targets))
    x= switch_label(datasetList, classes_string)
    dataset = string_to_float(x)
    get_csv(dataset)