import statsmodels.api as sm
import scipy
import numpy as np
#import bigfloat
#bigfloat.exp(5000,bigfloat.precision(100))


def get_coeff_datapoint(datapoint):


    model = sm.tsa.ARIMA(np.array(datapoint), order=(3,0,0)).fit(method='css')

    return model.params.tolist()

def get_coeffs(dataset):
    new = []
    for item in dataset:

        new.append(get_coeff_datapoint(item))
    return new

