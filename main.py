from launchAlgorithms import *
from utilities import *
from AutoregModel import *
import sys


if __name__ == "__main__":



    trainFilename = "train.csv"
    testFilename = "test.csv"
    tdatasetList, tattributes, tcoloumns = read_file(trainFilename)
    ttargetsDic, ttargets = get_classes(tcoloumns['label'])
    training = string_to_float(switch_label(tdatasetList, ttargets))

    datasetList, attributes, coloumns = read_file(testFilename)
    targetsDic, targets = get_classes(coloumns['label'])
    test = string_to_float(switch_label(datasetList, targets))

    models = {}
    kernel = "poly"


    # launch supervised algorithms

    trainingSet, testSet = get_copy_lists(training, test)
    fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet)

    launch_KNN(fitTransformedTraining, test, scaler)
    print("--------------")
    launch_SVM_SVC(fitTransformedTraining, testSet, scaler, kernel)
    print ("--------------")
    # print "OnevsAllClassifier evaluation"
    # trainingSet, testSet = get_copy_lists(training, test)
    # fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet)
    # models['svmOnevsAll'] = launch_SVM_oneVsall(fitTransformedTraining, testSet, scaler)
    # # print models['svmOnevsAll']
    # print "--------------"
    # print "OnevsOneClassifier evaluation"
    # trainingSet, testSet = get_copy_lists(training, test)
    # fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet)
    # models['svmOnevsOne'] = launch_SVM_oneVsone(data, fitTransformedTraining, testSet, scaler, crossValid)
    # # print models['svmOnevsOne']
    # print "--------------"
    # print "SVCLinear evaluation"
    # trainingSet, testSet = get_copy_lists(training, test)
    # fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet)
    # models['SVCLinear'] = launch_SVCLinear(data, fitTransformedTraining, testSet, scaler, crossValid)
    # print models['SVCLinear']
    # print "--------------"
    # print "Svm evaluation"
    # trainingSet, testSet = get_copy_lists(training, test)
    # fitTransformedTraining, scaler = get_minmax_scaled_dataset_and_scaler(trainingSet)
    # models['svm'] = launch_svm(data, fitTransformedTraining, testSet, scaler, crossValid)
    # # print models['svm']
    # print "--------------"


