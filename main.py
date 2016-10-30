from lagrangeAlgorithm import *





if __name__ == "__main__":


    #datasetList is a list of lists of datas,
    #attributes is the first line
    #coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    #extract which are the classes and order it alphabetically
    targets = coloumns['class']
    classes_string = sorted(unrepeated_values(targets))
    dataset = string_to_float(switch_label(datasetList, classes_string))
    choice = input("Insert 1 if you split the whole dataset in labaled/unlabaled, unless 2\n")
    if (choice == 1 or choice == 2):
        if choice == 1:
            training = dataset
        else:
            percentageTrain = input("Insert percentage of the training set\n")
            training, test = split_dataset(dataset, get_number_from_percentage(len(dataset), percentageTrain))

        percentageLabel = input("Insert the percentage of labeled data\n")
        #call a function for a specific algorithm
        scoreLagrange = launch_lagrange(dataset, percentageLabel)






    else: ("invalid input, please try again")









