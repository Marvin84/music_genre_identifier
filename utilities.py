import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_minibatches(X,y,mb_size=20):
    n_samples = X.shape[0]
    n_batches = np.ceil(1.*n_samples/mb_size).astype(np.int32)
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    for b in range(n_batches):
        left_i  = b*mb_size
        right_i = min((b+1)*mb_size, n_samples)
        indices = sorted(all_indices[left_i:right_i])
        yield X[indices, :], y[indices][:,None]


def standard_scale_dataset(training):

    scaler = StandardScaler()
    trainArray = np.array(training)
    trainData, trainTarget = trainArray[:, :(len(trainArray[0]) - 1)], trainArray[:, len(trainArray[0]) - 1]
    scaler.fit(trainData)
    trainData = scaler.transform(trainData)
    trainArray = np.c_[trainData, trainTarget]
    training = trainArray.tolist()
    for item in training:
        item[-1] = int(item[-1])
    return training, scaler

def scale_stochastic_dataset(xTrain, xTest):
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    return xTrain, xTest


# getting the scaled train and test sets and the test scaler
# training was fit and transformed the test only fit
def get_minmax_scaled_dataset_and_scaler(training):

    # instantiate the MinMax instance for training and for test
    scaler = MinMaxScaler(feature_range=(0,1))
    trainArray = np.array(training)
    trainData, trainTarget = trainArray[:, :(len(trainArray[0]) - 1)], trainArray[:, len(trainArray[0]) - 1]
    scaler.fit(trainData)
    trainData = scaler.transform(trainData)
    trainArray = np.c_[trainData, trainTarget]
    training = trainArray.tolist()
    for item in training:
        item[-1] = int(item[-1])
    return training, scaler