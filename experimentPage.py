import numpy as np
import pandas as pd
import copy
from scipy.optimize import minimize

# given Params
r0 = 0.0174
sigma = 0.01730
dt = 0.5
P0X = [97.8925, 96.1462]

maturities = np.array([0.5, 1])
thetasLi = []
sigmas = np.array([0.01730, 0.01730])
n = len(maturities)
rateTree = np.zeros([n + 1, n + 1])
rateTree[0, 0] = r0

for k in range(2, len(P0X) + 2):
    rateTreeCopy = copy.deepcopy(rateTree[:k, :k])
    bondPrice = copy.deepcopy(rateTree[:k, :k])
    def minErr(x):

        tempTheta = x[0]

        for i in range(1, rateTreeCopy.shape[0]):
            for j in range(rateTreeCopy.shape[0]):
                if j <= i:
                    if j == 0:
                        if i == rateTreeCopy.shape[0] - 1:
                            rateTreeCopy[j, i] = rateTreeCopy[j, i - 1] + tempTheta * dt + sigmas[i - 1] * np.sqrt(dt)
                        else:
                            rateTreeCopy[j, i] = rateTreeCopy[j, i - 1] + thetasLi[i-1] * dt + sigmas[i - 1] * np.sqrt(dt)
                    else:
                        rateTreeCopy[j, i] = rateTreeCopy[0, i] - 2 * j * sigmas[i - 1] * np.sqrt(dt)

                    bondPrice[j, i] = np.exp(-rateTreeCopy[j, i] * dt)

        for l in range(rateTreeCopy.shape[0] - 1, 0, -1):
            for m in range(rateTreeCopy.shape[0] - 1):
                bondPrice[m, l - 1] = np.exp(-rateTreeCopy[m, l - 1] * dt) * (bondPrice[m, l] + bondPrice[m + 1, l]) * 0.5

        return (bondPrice[0, 0] * 100 - P0X[k-2]) ** 2


    ret = minimize(minErr, 0.02, method='BFGS')
    thetasLi.append(ret.x[0])
print(thetasLi)




# draw rate tree
thetas=np.array([0.02,0.02])
diffusionTerm = sigma * np.sqrt(dt)
rateTree = np.zeros([n])
rateTree[:] = thetas * dt + diffusionTerm
rateTree[0] = r0
rateTree = rateTree.cumsum()

sigmaLI = np.full(n, sigma)
diffMinusDF = np.zeros([n, n])
diffMinusDF[:, :] = -(2 * sigmaLI * np.sqrt(dt)).reshape(-1, 1)
diffMinusDF[0, :] = rateTree[:]
diffMinusDF = diffMinusDF.cumsum(axis=0)
diffMinusDF[np.tril_indices_from(diffMinusDF, -1)] = 0

# first
notional = 100
discountFactor = np.exp(-diffMinusDF * dt)
bondPrice = discountFactor * notional
bondPrice[np.tril_indices_from(bondPrice, -1)] = 0

#time axis, starts from the end
for i in range(bondPrice.shape[0] - 1, 0, -1):
    # price axis, starts from the front
    for j in range(bondPrice.shape[0] - 1):
        if j < i:
            bondPrice[j, i - 1] = (bondPrice[j, i] + bondPrice[j + 1, i]) * 0.5 * discountFactor[j, i - 1]

P02 = 97.8925
P03 = 96.1462

# make variables
givenP0X = [P02, P03]
dfNaive = np.exp(-diffMinusDF * dt) * 100
Pli = []


print(np.linspace(0.5,5,10))