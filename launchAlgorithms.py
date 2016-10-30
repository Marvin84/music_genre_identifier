from manageDataset import *
from lagrange.lagrangean_s3vm import *
from qn.qns3vm import *
from qn.examples import *
import mySeed
import qn.randomGenerator



#set the pairTarget list if you want the pairwise version
def launch_lagrange(training, test, percentageLabel, r, parTarget = None):
    xTrainL, yTrainL, xTrainU, xTest, yTest, l, u = get_lagrange_dataset(training, test, parTarget, percentageLabel, True)
    best_estimator = get_best_estimator_by_cv(xTrainL, yTrainL, 3)
    model = lagrangian_s3vm_train(l, u, xTrainL, yTrainL, xTrainU, C=best_estimator.C,
                                 gamma=best_estimator.gamma, kernel=best_estimator.kernel, r=r, rdm = mySeed.rdm)

    print "The score is: ", model.score(xTest,yTest)


def launch_qn_algorithm(training, test, percentageLabel, r):
    xTrainL, yTrainL, xTrainU, xTest, yTest = get_qn_dataset(training, test, percentageLabel)
    model = QN_S3VM(xTrainL, yTrainL, xTrainU, qn.randomGenerator.my_random_generator, lam=0.0009765625, lamU=1,
                    kernel_type="RBF", sigma=0.5, estimate_r=r )
    model.train()
    preds = model.getPredictions(xTest)
    error = classification_error(preds,yTest)
    print "Classification error of QN-S3VM: ", error


if __name__ == "__main__":

    # datasetList is a list of lists of datas,
    # attributes is the first line
    # coloumns are the coloumns of vlaues refered to attributes
    datasetList, attributes, coloumns = read_file('/dataset.csv')
    # extract which are the classes and order it alphabetically
    targets = coloumns['class']
    classes_string = sorted(unrepeated_values(targets))
    dataset = string_to_float(switch_label(datasetList, classes_string))
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

        launch_lagrange(training, test, percentageLabel, .5, [2, 4])
        launch_qn_algorithm(training, test, percentageLabel, .0)





    else:
        ("invalid input, please try again")
