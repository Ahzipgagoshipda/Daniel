import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

dt = 0.5
T = 1
ni = int(T / dt)
r0 = 0.0174
ru = 0.0339
rd = 0.0095

strike = 99
notional = 100
P02 = 97.8925  # given value of zero bond price maturing in a year

# draw tree
tree = np.zeros([ni, ni])
tree[0, 0] = r0
tree[:, 1] = [ru, rd]

# draw bond prices
bpTree = np.exp(-tree * dt) * notional
bpTree[0:, 0] = 0
bpTree[0, 0] = P02

# draw option value tree (calculate up prop p)
optValTree = np.where(bpTree - strike > 0, bpTree - strike, 0)
p = (P02 * np.exp(r0 * dt) - bpTree[1, 1]) / (bpTree[0, 1] - bpTree[1, 1])
optPrice = (optValTree[0, 1] * p + optValTree[1, 1] * (1 - p)) * np.exp(-r0 * dt)

# opt price
strikLI = np.linspace(97.3, 99.5, 11)

priceli = []
for k in strikLI:
    optValTree = np.where(bpTree - k > 0, bpTree - k, 0)
    p = (P02 * np.exp(r0 * dt) - bpTree[1, 1]) / (bpTree[0, 1] - bpTree[1, 1])
    optPrice = (optValTree[0, 1] * p + optValTree[1, 1] * (1 - p)) * np.exp(-r0 * dt)
    priceli.append(optPrice)

plt.plot(strikLI, priceli)
plt.show()

# swap price
swapRate = 0.02
swapValTree = notional * ((tree - swapRate) * dt)
swapValTree[:, 0] = 0
rnProb = np.array([p, 1 - p])
swapVal = (swapValTree[:, 1] @ rnProb) * np.exp(-r0 * dt)

swapRateLi = np.linspace(0.005, 0.04, 15)
swapPriceLi = []
for sr in swapRateLi:
    swapValTree = notional * ((tree - sr) * dt)
    swapValTree[:, 0] = 0
    swapVal = (swapValTree[:, 1] @ rnProb) * np.exp(-r0 * dt)
    swapPriceLi.append(swapVal)

plt.plot(swapRateLi, swapPriceLi)
plt.axhline(0, c='r')
plt.show()
