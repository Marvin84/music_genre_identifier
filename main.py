from launchAlgorithms import *


if __name__ == "__main__":

    # datasetList is a list of lists of datas,
    # attributes is the first line
    # coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    # extract dictionary of classes orederd alphabetically and associated with a number
    #targets are the list of numbers associated to the ordered classes
    targetsDic, targets = get_classes(coloumns['class'])
    dataset = string_to_float(switch_label(datasetList, targets))
    models = {}
    choice = input("Insert 1 if you have a seperate test set, otherwise 2\n")
    if (choice == 1 or choice == 2):
        if choice == 1:
            training = dataset
            filename = input("insert the path to test set")
            test, attributes, coloumns = read_file('/dataset.csv')

        else:
            percentageTrain = input("Insert percentage of the training set\n")
            training, test = split_dataset(dataset, percentageTrain)

        typeAlgoChoice = input("chose 1 for semi-supervised and 2 for supervised\n")
        if (typeAlgoChoice == 1 or typeAlgoChoice == 2):
            if typeAlgoChoice == 1:
                # call a function for a specific algorithm
                # remember to chose r
                trainingSet, testSet = get_copy_lists(training, test)
                percentageLabel = input("Insert the percentage of labeled data\n")
                fitTransformedTraining, trainScaler, testScaler = get_scaleDataset_and_scalers(trainingSet, testSet)
                models['lagrangePairwise'] = launch_lagrange(fitTransformedTraining, testSet, testScaler,
                                                             percentageLabel, .5, True, [2, 5])
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, trainScaler, testScaler = get_scaleDataset_and_scalers(trainingSet, testSet)
                models['lagrangeOneVsAll'] = launch_oneVsRest_lagrange(fitTransformedTraining, testSet, testScaler,
                                                                       targets, targetsDic, percentageLabel, .1)
                trainingSet, testSet = get_copy_lists(training, test)
                models['qn'] = launch_qn(trainingSet, testSet, percentageLabel, .0, [2, 5])
            else:
                cvChoice = input("Insert 1 you use 10-fold cross validation, otherwise 2\n")
                if (cvChoice == 1 or cvChoice == 2):
                    if cvChoice == 1:
                        crossValid = True
                    else: crossValid = False
                else:
                    "unvalid choice"
                    sys.exit()
                kernelChoice = input("choose the kernel for svm: 1 for rbf, 2 for linear and 3 for poly\n")
                if (kernelChoice == 1 or kernelChoice == 2 or kernelChoice==3):
                    if kernelChoice == 1:
                        kernel = 'rbf'
                    elif kernelChoice == 2:
                        kernel = 'linear'
                    else: kernel = 'poly'
                else:
                    "unvalid choice"
                    sys.exit()

                #launch supervised algorithms
                models['knn'] = launch_KNN(training, test, crossValid)
                models['svmSVC'] = launch_SVM_SVC(training, test, kernel, crossValid)
                models['svmOnevsAll'] = launch_SVM_oneVsall(training, test, crossValid)
                models['svmOnevsOne'] = launch_SVM_oneVsone(training, test, crossValid)
                models['SVCLinear'] = launch_SVCLinear(training, test, crossValid)
                models['SGDC']= launch_SGDClassifier(training, test, crossValid)
        else:
            "unvalid choice"
            sys.exit()
    else:
        ("invalid input, please try again")
        sys.exit()