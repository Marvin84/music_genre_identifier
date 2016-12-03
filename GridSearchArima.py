import statsmodels.api as sm
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError


def get_mtFeatures(data, mtWin, mtStep, stWin, stStep):
    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    stFeatures = data
    numOfFeatures = 34
    numOfStatistics = 2

    mtFeatures = []
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):  # for each of the short-term features:
        curPos = 0
        N = 1200
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(np.mean(curStFeatures))
            mtFeatures[i + numOfFeatures].append(np.std(curStFeatures))
            # mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
            curPos += mtStepRatio
    return mtFeatures


def get_datapoint_with_window(sPoint, mW, mS, sW, sS):
    #which datapoint
    #datapoint is a list of 34 short features with length 1200
    dataPoint = []
    for i in range(34):
        dataPoint.append(sPoint[i*1200:(i+1)*1200])
    return get_mtFeatures(dataPoint, mW, mS, sW, sS)



def choose_model(observation):

    parameters = {'p': [1, 2, 3, 4, 5], 'd': [0, 1, 2], 'q': [0, 1, 2]}
    windows = [[1.0, 0.50], [0.70, 0.35], [0.50, 0.25]]
    grid = []
    for p in parameters['p']:
        for d in parameters['d']:
            for q in parameters['q']:
                grid.append([p, d, q])

    bestModelIdentities = {'params': [], 'models': [], 'results': [], 'window': []}
    # range(34) for mean values and range(34,68) for standard deviation
    for featureIndex in range(34):
        models = []
        results = []
        bests = []
        differentWindows = {'1': [], '2': [], '3': []}

        for i in range(3):
            m = get_datapoint_with_window(observation, windows[i][0], windows[i][1], 0.050, 0.025)
            errors = []
            aic = []
            for value in grid:
                try:
                    model = sm.tsa.ARIMA(m[featureIndex], order=value)
                    result = model.fit(method='css')
                except (ValueError, LinAlgError):
                    print "error with", value, "with window size:", windows[i]
                    pass

                e = result.resid ** 2
                errors.append(np.sum(e) / len(e))
                aic.append(result.aic)
                models.append(model)
                results.append(result)

            index = errors.index(np.amin(np.array(errors)))
            bests.append(aic[index])
            differentWindows[str(i + 1)].append(models[index])
            differentWindows[str(i + 1)].append(results[index])

        bestIndex = bests.index(np.amin(np.array(bests)))
        bestModelIdentities['params'].append(differentWindows[str(bestIndex + 1)][1].params.tolist())
        bestModelIdentities['models'].append(differentWindows[str(bestIndex + 1)][0])
        bestModelIdentities['results'].append(differentWindows[str(bestIndex + 1)][1])
        bestModelIdentities['window'].append(bestIndex)

    return bestModelIdentities





