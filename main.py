from launchAlgorithms import *
import copy
import mySeed

if __name__ == "__main__":

    # datasetList is a list of lists of datas,
    # attributes is the first line
    # coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    # extract dictionary of classes orederd alphabetically and associated with a number
    #targets are the list of numbers associated to the ordered classes
    targetsDic, targets = get_classes(coloumns['class'])
    dataset = string_to_float(switch_label(datasetList, targets))
    choice = input("Insert 1 if you have a seperate test set, unless 2\n")
    if (choice == 1 or choice == 2):
        if choice == 1:
            training = dataset
            filename = input("insert the path to test set")
            test, attributes, coloumns = read_file('/dataset.csv')

        else:
            percentageTrain = input("Insert percentage of the training set\n")
            training, test = split_dataset(dataset, percentageTrain)

        percentageLabel = input("Insert the percentage of labeled data\n")

        # call a function for a specific algorithm
        # remember to chose r
        models = {}

        fitTransformedTraining, trainScaler, testScaler = get_scaleDataset_and_scalers(training, test)
        models['lagrangePairwise'] = launch_lagrange(fitTransformedTraining, test, testScaler,
                                                     percentageLabel, .5, True, [2, 5])

        fitTransformedTraining, trainScaler, testScaler = get_scaleDataset_and_scalers(training, test)
        models['lagrangeOneVsAll']= launch_oneVsRest_lagrange(fitTransformedTraining, test, testScaler,
                                                              targets, targetsDic, percentageLabel, .1)

        models['qn'] = launch_qn_algorithm(training, test, percentageLabel, .0, True, [2,5])







    else:
        ("invalid input, please try again")