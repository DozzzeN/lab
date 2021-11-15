from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat

rawData = loadmat('Scenario3-Mobile/data_mobile_1.mat')
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dctCSIa1 = dct(CSIa1Orig)
dctCSIb1 = dct(CSIb1Orig)


print("CSIa1Orig", CSIa1Orig)
print("CSIb1Orig", CSIb1Orig)
print("dctCSIb1", dctCSIb1)
print("dctCSIb1", dctCSIb1)

plt.figure()
plt.plot(range(len(dctCSIa1)), dctCSIa1, color="red", label="a")
plt.plot(range(len(dctCSIb1)), dctCSIb1, color="blue", label="b")
plt.legend(loc='upper left')
plt.show()
