from dataStructures import *
from manage_dataset import *





if __name__ == "__main__":

    #datasetList is a list of lists of datas,
    #attributes is the first line
    #coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    #extract which are the classes and order it alphabetically
    targets = coloumns['class']
    classes_string = sorted(unrepeated_values(targets))
    dataset = string_to_float(switch_label(datasetList, classes_string))
    training, test = split_dataset(dataset, 50)
    get_csv(training, 'train')
    get_csv(test, 'test')