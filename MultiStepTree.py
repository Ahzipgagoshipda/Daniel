import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

predictions = np.array([1.74, 2.17, 2.56, 2.91]) / 100
maturities = np.array([0.5, 1, 1.5, 2])
givenP0 = np.array([97.8925, 96.1462, 94.1011])
sigma = 0.0173
dt = 0.5
predDiffArr = np.diff(predictions)
nj = maturities.shape[0]
ni = maturities.shape[0]

rateTree = np.zeros([nj, ni])
rateTree = -np.tile(np.arange(nj), nj).reshape(-1, nj).T * 2 * sigma * np.sqrt(dt)
rateTree[0, :] = (np.insert(predDiffArr + sigma * np.sqrt(dt), 0, predictions[0])).cumsum()
rateTree[1:, :] = rateTree[1:, :] + rateTree[0, :]
rateTree[np.tril_indices_from(rateTree, -1)] = 0

print(pd.DataFrame(rateTree))

# bond prices
rateTree[np.tril_indices_from(rateTree, -1)] = 1e+5
bondPrices = (np.exp(-rateTree * dt) * 100)
print(pd.DataFrame(bondPrices))

Pli = []
print(bondPrices)

for n in range(2, bondPrices.shape[0] + 1):

    def mimErr(x):
        P0XMat = copy.deepcopy(bondPrices[:n, :n])
        for i in range(P0XMat.shape[1] - 1, 0, -1):
            if i == P0XMat.shape[1] - 1:
                thisP = x[0]
            else:
                thisP = Pli[i - 1]
            for j in range(P0XMat.shape[1] - 1):
                if j < i:
                    P0XMat[j, i - 1] = (P0XMat[j, i] * thisP + P0XMat[j + 1, i] * (1 - thisP)) * np.exp(-rateTree[j, i - 1] * dt)
        return (P0XMat[0, 0] - givenP0[n - 2]) ** 2


    res = minimize(mimErr, 0.5, method='SLSQP', bounds=[(0, 1)])

    Pli.append(res.x[0])

print(bondPrices)
