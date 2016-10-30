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
    choice = input("insert 1 if you split the whole dataset in labaled/unlabaled, unless 2\n")
    if (choice == 1 or choice == 2):
        if choice == 1:
            training = dataset
        else:
            percentageTrain = input("percentage of the training set\n")
            training, test = split_dataset(dataset, get_number_from_percentage(len(dataset), percentageTrain))

        percentageLabel = input("please insert the percentage of labeled data\n")

        #calling the specific function from manage_dataset
        # for having the right format of dataset for the desired algorithm
        '''
        xtrain_l, ytrain_l, xtrain_u, xtest, ytest, l, u = get_lagrange_dataset(training, [1,3], percentageLabel, True)
        np.savetxt('data[:l].csv', xtrain_l)
        np.savetxt('targets[:l].csv', ytrain_l)
        np.savetxt('data[l:l+u].csv', xtrain_u)
        np.savetxt('data[l+u:].csv', xtest)
        np.savetxt('targets[l+u:].csv', ytest)
        '''




    else: ("invalid input, please try again")









