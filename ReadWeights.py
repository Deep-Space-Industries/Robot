import numpy as np
import pickle

fitness = np.load("/FT.npy")
historyWeightsIH = np.load("/IH.npy")
historyBiasIHH1 = np.load("/BIHH.npy")
historyWeightsHH = np.load("/HH.npy")
historyBiasIHHn = np.load("/BIHHn.npy")
historyWeightsHO = np.load("/HO.npy")
historyBiasHO = np.load("/BHO.npy")

print(fitness)
print(historyWeightsIH)
print(historyBiasIHH1)
print(historyWeightsHH)
print(historyBiasIHHn)
print(historyWeightsHO)
print(historyBiasHO)
