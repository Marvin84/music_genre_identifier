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
            test, testAttributes, testColoumns = read_file('/dataset.csv')

        else:
            percentageTrain = input("Insert percentage of the training set\n")
            training, test = split_dataset(dataset, percentageTrain)

        typeAlgoChoice = input("choose 1 for semi-supervised and 2 for supervised\n")
        if (typeAlgoChoice == 1 or typeAlgoChoice == 2):
            if typeAlgoChoice == 1:
                percentageLabel = input("Insert the percentage of labeled data\n")
                #call a function for a specific algorithm
                #remember to chose r
                #
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, scaler= get_minmax_scaled_dataset_and_scaler(trainingSet, testSet)
                pairTarget = [2, 4]
                print "lagrange pairwise evaluation on targets: ", pairTarget
                models['lagrangePairwise'], pairScore = launch_lagrange(fitTransformedTraining, testSet, scaler,
                                                             percentageLabel, .5, True, pairTarget)
                print "The lagrange's score for pair targets ", pairTarget, "is: ", pairScore
                print "--------------"
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet, testSet)
                print "lagrange doing one vs all organized manually"
                models['lagrangeOneVsAll'] = launch_oneVsRest_lagrange(fitTransformedTraining, testSet, scaler,
                                                                       targets, targetsDic, percentageLabel, .1)
                print "--------------"
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet, testSet)
                print "lagrange doing one vs one organized manually"
                oneVsOneScore = launch_oneVsone_lagrange(fitTransformedTraining, test, scaler,
                                                         targets, percentageLabel, .1)
                print "The lagrange's score with one vs one strategy is: ", oneVsOneScore
                print "--------------"
                trainingSet, testSet = get_copy_lists(training, test)
                print "qn pairwise evaluation on targets: ", pairTarget
                models['qn'], qnError = launch_qn(trainingSet, testSet, percentageLabel, .0, pairTarget)
                print "Classification error of QN-S3VM: ", qnError
                print "--------------"
                trainingSet, testSet = get_copy_lists(training, test)
                print "Qn doing one vs one organized manually"
                oneVsOneError = launch_oneVsone_qn(trainingSet, testSet, targets, percentageLabel, .0)
                print "The qn's average classification error with one vs one strategy is: ", oneVsOneError


            else:
                cvChoice = input("Insert 1 you use 5-fold cross validation, otherwise 2\n")
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
                #print "Knn evaluation"
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet, testSet)
                models['knn'] = launch_KNN(fitTransformedTraining, test, scaler, crossValid)
                print "--------------"
                print "SVC svm evaluation with ", kernel, " kernel"
                trainingSet, testSet = get_copy_lists(training, test)
                models['svmSVC'] = launch_SVM_SVC(trainingSet, testSet, kernel, crossValid)
                print "--------------"
                print "OnevsAllClassifier evaluation"
                trainingSet, testSet = get_copy_lists(training, test)
                models['svmOnevsAll'] = launch_SVM_oneVsall(trainingSet, testSet, crossValid)
                print "--------------"
                print "OnevsOneClassifier evaluation"
                trainingSet, testSet = get_copy_lists(training, test)
                models['svmOnevsOne'] = launch_SVM_oneVsone(trainingSet, testSet, crossValid)
                print "--------------"
                print "SVCLinear evaluation"
                trainingSet, testSet = get_copy_lists(training, test)
                fitTransformedTraining, scaler = standard_scale_dataset(trainingSet, testSet)
                models['SVCLinear'] = launch_SVCLinear(fitTransformedTraining, testSet, scaler, crossValid)
                print "--------------"
                print "SGDCClassifier evaluation"
                trainingSet, testSet = get_copy_lists(training, test)
                models['SGDC']= launch_SGDClassifier(trainingSet, testSet, crossValid)
        else:
            "unvalid choice"
            sys.exit()
    else:
        ("invalid input, please try again")
        sys.exit()