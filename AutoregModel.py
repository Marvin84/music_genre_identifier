import statsmodels.tsa as st
from GridSearchArima import *
import matplotlib.pyplot as plt

def get_coeffs(filename):

    coeffDataset=[]
    dataset = pd.read_csv(filename)
    M = np.array(dataset)

    for item in M:
        parameters = []
        p = choose_model(item)['params']
        for list_ in p:
            for value in list_:
                parameters.append(value)
        coeffDataset.append(parameters)
    return coeffDataset


#it gets the result of fitting a arima model and plot
#normally result = bestModelIdentities['results'][pIndex]
def plot_model(result):

    # autocorrelation function of residuals
    result.plot_predict()
    plt.show()
    acValues = st.stattools.acf(result.resid, unbiased=False, nlags=50, qstat=False, fft=False, alpha=None)
    plt.plot(acValues, 'm-', label='pac')
    plt.show()
    plt.hist(result.resid)
    plt.show()





