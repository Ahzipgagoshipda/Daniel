from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy

from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from scipy.optimize import minimize


class IRtreePricer(ABC):

    def __init__(self, dt, r0, notional, maturities):
        self.dt = dt
        self.r0 = r0
        self.notional = notional
        self.maturities = maturities

    @abstractmethod
    def rateCalcFormulas(self):
        pass

    def bondPriceCalc(self, rateTreeCopy, bondPrice):
        bondFunc = self.rateCalcFormulas()[3]
        for i in range(rateTreeCopy.shape[0] - 1, 0, -1):
            for j in range(rateTreeCopy.shape[0] - 1):
                # bondPrice[m, l - 1] = np.exp(-rateTreeCopy[m, l - 1] * self.dt) * (
                #         bondPrice[m, l] + bondPrice[m + 1, l]) * 0.5
                bondPrice[j, i - 1] = bondFunc(rateTreeCopy, j, i-1, self.dt) * (
                        bondPrice[j, i] + bondPrice[j + 1, i]) * 0.5

        return bondPrice

    def solver(self, sigmaLI, givenP0X):

        n = len(self.maturities)

        if isinstance(sigmaLI, (int, float)):
            sigmaLI = np.full(n, sigmaLI)

        thetasLi = []
        rateTree = np.zeros([n + 1, n + 1])
        rateTree[0, 0] = self.r0

        for k in range(2, len(givenP0X) + 2):

            rateTreeCopy = rateTree[:k, :k]
            bondPrice = np.zeros(rateTreeCopy.shape)
            topLineFunc = self.rateCalcFormulas()[0]
            pastRateFunc = self.rateCalcFormulas()[1]
            minus2SigFunc = self.rateCalcFormulas()[2]
            bondFunc = self.rateCalcFormulas()[3]

            def minErr(x):
                tempTheta = x[0]
                for i in range(1, rateTreeCopy.shape[0]):
                    for j in range(rateTreeCopy.shape[0]):
                        if j <= i:
                            if j == 0:
                                if i == rateTreeCopy.shape[0] - 1:
                                    rateTreeCopy[j, i] = topLineFunc(rateTreeCopy, j, i, tempTheta, self.dt, sigmaLI)

                                else:
                                    rateTreeCopy[j, i] = pastRateFunc(rateTreeCopy, j, i, thetasLi, self.dt, sigmaLI)

                            else:
                                rateTreeCopy[j, i] = minus2SigFunc(rateTreeCopy, j, i, self.dt, sigmaLI)

                            bondPrice[j, i] = bondFunc(rateTreeCopy, j, i, self.dt)

                nowBondPrice = self.bondPriceCalc(rateTreeCopy, bondPrice)[0, 0] * self.notional

                return (nowBondPrice - givenP0X[k - 2]) ** 2

            ret = minimize(minErr, np.array([0.5]), method='BFGS')
            thetasLi.append(ret.x[0])

        if isinstance(self, simpleBDT):
            rateTree = np.exp(rateTree)
            rateTree = np.triu(rateTree)

        return thetasLi, rateTree

    def statePrice(self, rateTree):
        statePrice: ndarray[Any, dtype[floating[_64Bit] | float_]] = np.zeros(rateTree.shape)
        statePrice[0, 0] = 1
        for i in range(1, statePrice.shape[1]):
            for j in range(statePrice.shape[0]):
                if j <= i:
                    if j == 0:
                        statePrice[j, i] = statePrice[j, i - 1] * np.exp(-rateTree[j, i - 1] * self.dt) * 0.5
                    else:
                        statePrice[j, i] = 0.5 * (
                                statePrice[j - 1, i - 1] * np.exp(-rateTree[j - 1, i - 1] * self.dt) +
                                statePrice[j, i - 1] * np.exp(-rateTree[j, i - 1] * self.dt))

        return statePrice

    def discountFactor(self, sigmaLI, givenP0X):
        thetasLi, rateTree = self.solver(sigmaLI, givenP0X)
        statePrice = self.statePrice(rateTree)

        endMaturity = len(thetasLi) * self.dt
        matCol = np.linspace(self.dt, endMaturity, len(thetasLi))
        df_df = pd.DataFrame(np.sum(statePrice[:, 1:], axis=0))
        df_df.index = [matCol]
        df_df.columns = ["Discount Factor"]
        return df_df

    def structuredZeroPricer(self, strikePrice, sigmaLi, givenP0X, maturityTenor, tradeNotional):
        thetaLi, rateTree = self.solver(sigmaLi, givenP0X)
        matColNum: int = int((maturityTenor / self.dt))

        bpDf = copy.deepcopy(rateTree)
        bpDf[:, matColNum] = np.where(rateTree[:, matColNum] * tradeNotional > strikePrice,
                                      rateTree[:, matColNum] * tradeNotional, strikePrice)

        bondPrice = self.bondPriceCalc(rateTree, bpDf)
        bondPrice = np.triu(bondPrice)

        return bondPrice


class hoLee(IRtreePricer):

    def __init__(self, dt, r0, notional, maturities):
        super().__init__(dt, r0, notional, maturities)

    def rateCalcFormulas(self):
        topLineFunc = lambda tree, j, i, the, dt, sig: tree[j, i - 1] + the * dt + sig[i - 1] * np.sqrt(dt)
        pastRateFunc = lambda tree, j, i, theLi, dt, sig: tree[j, i - 1] + theLi[i - 1] * dt + sig[i - 1] * np.sqrt(dt)
        minus2SigFunc = lambda tree, j, i, dt, sig: tree[0, i] - 2 * j * sig[i - 1] * np.sqrt(dt)
        bondFunc = lambda tree, j, i, dt: np.exp(-tree[j, i] * dt)

        return [topLineFunc, pastRateFunc, minus2SigFunc, bondFunc]


class simpleBDT(IRtreePricer):

    def __init__(self, dt, r0, notional, maturities):
        super().__init__(dt, r0, notional, maturities)

    def rateCalcFormulas(self):
        topLineFunc = lambda tree, j, i, the, dt, sig: tree[j, i - 1] + the * dt + sig[i - 1] * np.sqrt(dt)
        pastRateFunc = lambda tree, j, i, theLi, dt, sig: tree[j, i - 1] + theLi[i - 1] * dt + sig[i - 1] * np.sqrt(dt)
        minus2SigFunc = lambda tree, j, i, dt, sig: tree[0, i] - 2 * j * sig[i - 1] * np.sqrt(dt)
        bondFunc = lambda tree, j, i, dt: np.exp((-np.exp(tree[j, i]) * dt))

        return [topLineFunc, pastRateFunc, minus2SigFunc, bondFunc]
