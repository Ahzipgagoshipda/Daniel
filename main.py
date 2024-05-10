import copy

from IRtreePricer import *
import numpy as np
import pandas as pd


def main():
    r0 = 0.0174
    sigma = 0.01730
    dt = 0.5
    notional = 100
    P0X = [97.8925, 96.1462, 94.1011, 91.7136, 89.2258, 86.8142, 84.5016, 82.1848, 79.7718, 77.433900]
    maturities = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])

    hoLeeInst = hoLee(dt, r0, notional, maturities)
    holee_thetaLi, holee_RateTree = hoLeeInst.solver(sigma, P0X)
    DF = hoLeeInst.discountFactor(sigma, P0X)

    strikePrice = 94
    maturityTenor = 5
    tradeNotional = 1100
    structPrice = hoLeeInst.structuredZeroPricer(strikePrice, sigma, P0X, maturityTenor, tradeNotional)

    sigmaBDT = 0.2142
    bdtInst = simpleBDT(dt, np.log(r0), notional, maturities)
    bdt_thetaLi, bdt_RateTree = bdtInst.solver(sigmaBDT, P0X)

    return holee_RateTree, bdt_RateTree


if __name__ == "__main__":
    holee_RateTree, bdt_RateTree = main()

    print(pd.DataFrame(holee_RateTree))
    print(pd.DataFrame(bdt_RateTree))
